"""Post-hoc audit sensitivity analyses for the frozen-FM pool paper.

This script is intentionally CPU-only. It reuses the released per-case Dice
traces to quantify artifact-level robustness checks that do not require new
training:

* functional-floor sweep,
* Pearson-vs-ICC(A,1) metric sensitivity,
* leave-one-FM-out sensitivity,
* task-level pooled meta-summary,
* random-label permutation negative control.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmpool.estimators import functional_floor_filter, within_cross_rbar  # noqa: E402

import compute_all as ca  # noqa: E402


TASKS = ca.TASKS
DINOV2_FAMILY_PAIRS = ca.DINOV2_FAMILY_PAIRS
DEFAULT_FLOORS = (0.0, 0.20, 0.25, 0.30, 0.35, 0.40)


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_float(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _metric_row(per_case: dict[tuple[str, int], np.ndarray], stat: str) -> dict:
    point = within_cross_rbar(
        per_case,
        family_pairs_within=DINOV2_FAMILY_PAIRS,
        stat=stat,
    )
    return {
        "stat": stat,
        "within": _clean_float(point["within"]),
        "cross": _clean_float(point["cross"]),
        "gap": _clean_float(point["gap"]),
        "n_within_pairs": int(point["n_within_pairs"]),
        "n_cross_pairs": int(point["n_cross_pairs"]),
    }


def _fms(per_case: dict[tuple[str, int], np.ndarray]) -> list[str]:
    return sorted({fm for fm, _seed in per_case})


def _filter(
    per_case_all: dict[tuple[str, int], np.ndarray],
    floor: float,
) -> dict[tuple[str, int], np.ndarray]:
    if floor <= 0:
        return dict(per_case_all)
    return functional_floor_filter(per_case_all, min_mean_dice=floor, scope="per_fm")


def _shuffle_family_labels(
    per_case: dict[tuple[str, int], np.ndarray],
    rng: random.Random,
) -> dict[tuple[str, int], np.ndarray]:
    keys = sorted(per_case)
    labels = [fm for fm, _seed in keys]
    rng.shuffle(labels)
    occurrence: dict[str, int] = {}
    out: dict[tuple[str, int], np.ndarray] = {}
    for old_key, new_fm in zip(keys, labels, strict=True):
        occurrence[new_fm] = occurrence.get(new_fm, 0) + 1
        out[(new_fm, occurrence[new_fm])] = per_case[old_key]
    return out


def _permutation_control(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    n_perm: int,
    seed: int,
) -> dict:
    observed = _metric_row(per_case, "pearson")["gap"]
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(n_perm):
        shuffled = _shuffle_family_labels(per_case, rng)
        gap = _metric_row(shuffled, "pearson")["gap"]
        if gap is not None:
            gaps.append(float(gap))
    if observed is None or not gaps:
        return {
            "observed_gap": observed,
            "n_perm": int(n_perm),
            "valid_permutations": len(gaps),
            "null_mean": None,
            "null_sd": None,
            "null_p_ge_observed": None,
            "null_p_abs_ge_observed": None,
        }
    arr = np.asarray(gaps, dtype=np.float64)
    obs = float(observed)
    return {
        "observed_gap": obs,
        "n_perm": int(n_perm),
        "valid_permutations": int(arr.size),
        "null_mean": float(arr.mean()),
        "null_sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "null_q025": float(np.quantile(arr, 0.025)),
        "null_q975": float(np.quantile(arr, 0.975)),
        "null_p_ge_observed": float((1 + np.sum(arr >= obs)) / (arr.size + 1)),
        "null_p_abs_ge_observed": float(
            (1 + np.sum(np.abs(arr) >= abs(obs))) / (arr.size + 1)
        ),
    }


def _task_analysis(
    results_root: Path,
    task: str,
    *,
    floors: tuple[float, ...],
    n_perm: int,
    seed: int,
) -> dict:
    fm_jsons, _nnunet_jsons = ca._collect_task_jsons(results_root, task)
    if not fm_jsons:
        raise FileNotFoundError(f"no per-case Dice JSONs found for task {task!r}")
    ca._validate_alignment(fm_jsons)
    per_case_all = ca._as_per_case_map(fm_jsons)
    floor_rows: list[dict] = []
    for floor in floors:
        per_case = _filter(per_case_all, floor)
        row = _metric_row(per_case, "pearson")
        row.update(
            {
                "floor": float(floor),
                "n_members": len(per_case),
                "fms_kept": _fms(per_case),
            }
        )
        floor_rows.append(row)

    primary = _filter(per_case_all, 0.30)
    metric_rows = [_metric_row(primary, "pearson"), _metric_row(primary, "icc")]
    lofo_rows: list[dict] = []
    for fm in _fms(primary):
        reduced = {k: v for k, v in primary.items() if k[0] != fm}
        row = _metric_row(reduced, "pearson")
        row.update({"left_out_fm": fm, "n_members": len(reduced), "fms_kept": _fms(reduced)})
        lofo_rows.append(row)

    return {
        "task": task,
        "n_test": int(len(fm_jsons[0].test_ids)),
        "source_files": ca._rel_paths([j.path for j in fm_jsons], results_root),
        "available_fms": _fms(per_case_all),
        "available_members": len(per_case_all),
        "floor_sweep": floor_rows,
        "metric_sensitivity_floor_0.30": metric_rows,
        "leave_one_fm_out_floor_0.30": lofo_rows,
        "permutation_negative_control_floor_0.30": _permutation_control(
            primary,
            n_perm=n_perm,
            seed=seed,
        ),
    }


def _meta_summary(task_rows: dict[str, dict]) -> dict:
    primary = []
    for task, payload in task_rows.items():
        row = next(
            x for x in payload["floor_sweep"]
            if abs(float(x["floor"]) - 0.30) < 1e-12
        )
        primary.append((task, float(row["gap"])))
    gaps = np.asarray([gap for _task, gap in primary], dtype=np.float64)
    loto = []
    for task, _gap in primary:
        kept = np.asarray([gap for t, gap in primary if t != task], dtype=np.float64)
        loto.append({"left_out_task": task, "mean_gap": float(kept.mean())})
    return {
        "tasks": [task for task, _gap in primary],
        "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps.max()),
        "leave_one_task_out": loto,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--floors", nargs="*", type=float, default=list(DEFAULT_FLOORS))
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    task_rows = {
        task: _task_analysis(
            results_root,
            task,
            floors=tuple(args.floors),
            n_perm=args.n_perm,
            seed=args.seed,
        )
        for task in args.tasks
    }
    out = {
        "schema_version": ca.SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(results_root),
        "floors": [float(x) for x in args.floors],
        "permutation_seed": int(args.seed),
        "n_perm": int(args.n_perm),
        "task_rows": task_rows,
        "primary_floor_0.30_meta_summary": _meta_summary(task_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
