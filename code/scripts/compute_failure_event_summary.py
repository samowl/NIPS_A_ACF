#!/usr/bin/env python3
"""Five-task binary failure-event co-failure diagnostic.

The primary paper estimator uses Pearson correlation of per-case Dice traces.
This CPU-only diagnostic asks the complementary event question on the same
released traces: after the Table-1 functional floor, do binary failure events
co-occur more strongly for same-bundle/recipe pairs than for Table-1 cross
pairs?

For each task, failure thresholds are empirical percentiles over all
model-case Dice values in the functional pool. For each member pair we compute
the binary-event Pearson/phi correlation and the joint-failure rate, then
average those quantities under the same Table-1 same/cross grouping rule.
Case bootstrap intervals resample aligned cases with fixed thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmpool.estimators import functional_floor_filter  # noqa: E402

import compute_all as ca  # noqa: E402


SCHEMA_VERSION = "fmpool_failure_event_summary_v1"
PERCENTILES = (25, 33, 50)
DEFAULT_BOOTSTRAP_B = 1000
DEFAULT_BOOTSTRAP_SEED = 0


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(payload), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def _is_same_group(fm_a: str, fm_b: str) -> bool:
    return fm_a == fm_b or frozenset({fm_a, fm_b}) in ca.DINOV2_FAMILY_PAIRS


def _mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _ci95(values: list[float]) -> list[float | None]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


def _event_stats(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    threshold: float,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    keys = sorted(per_case)
    data = np.stack([np.asarray(per_case[k], dtype=np.float64) < threshold for k in keys])
    if indices is not None:
        data = data[:, indices]
    fail = data.astype(np.float64, copy=False)
    n_members, n_cases = fail.shape
    pair_i, pair_j = np.triu_indices(n_members, k=1)
    same_mask = np.asarray(
        [_is_same_group(keys[i][0], keys[j][0]) for i, j in zip(pair_i, pair_j, strict=True)],
        dtype=bool,
    )

    centered = fail - fail.mean(axis=1, keepdims=True)
    ss = np.sum(centered * centered, axis=1)
    denom = np.sqrt(np.outer(ss, ss))
    with np.errstate(invalid="ignore", divide="ignore"):
        phi_matrix = (centered @ centered.T) / denom
    phi_matrix[denom <= 1e-12] = np.nan
    joint_matrix = (fail @ fail.T) / float(n_cases)

    pair_phi = phi_matrix[pair_i, pair_j]
    pair_joint = joint_matrix[pair_i, pair_j]
    same_phi = pair_phi[same_mask].tolist()
    cross_phi = pair_phi[~same_mask].tolist()
    same_joint = pair_joint[same_mask].tolist()
    cross_joint = pair_joint[~same_mask].tolist()

    same_phi_mean = _mean(same_phi)
    cross_phi_mean = _mean(cross_phi)
    same_joint_mean = _mean(same_joint)
    cross_joint_mean = _mean(cross_joint)
    return {
        "same_fail_phi": same_phi_mean,
        "cross_fail_phi": cross_phi_mean,
        "gap_fail_phi": (
            same_phi_mean - cross_phi_mean
            if math.isfinite(same_phi_mean) and math.isfinite(cross_phi_mean)
            else float("nan")
        ),
        "same_joint_fail_rate": same_joint_mean,
        "cross_joint_fail_rate": cross_joint_mean,
        "joint_fail_rate_ratio": (
            same_joint_mean / cross_joint_mean
            if math.isfinite(same_joint_mean) and cross_joint_mean > 0
            else float("nan")
        ),
        "n_same_pairs": int(np.sum(same_mask)),
        "n_cross_pairs": int(np.sum(~same_mask)),
        "n_same_valid_phi_pairs": int(sum(math.isfinite(v) for v in same_phi)),
        "n_cross_valid_phi_pairs": int(sum(math.isfinite(v) for v in cross_phi)),
    }


def _summarize_task(
    results_root: Path,
    task: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    fm_jsons, _nnunet = ca._collect_task_jsons(results_root, task)
    ca._validate_alignment(fm_jsons)
    per_case_all = ca._as_per_case_map(fm_jsons)
    per_case = functional_floor_filter(per_case_all, ca.FUNCTIONAL_FLOOR)
    if len(per_case) < 2:
        raise RuntimeError(f"{task}: fewer than two traces after functional floor")

    all_values = np.concatenate([np.asarray(v, dtype=np.float64) for v in per_case.values()])
    thresholds = {
        percentile: float(np.percentile(all_values, percentile))
        for percentile in PERCENTILES
    }

    rng = np.random.default_rng(seed)
    n_cases = len(next(iter(per_case.values())))
    rows: list[dict[str, Any]] = []
    for percentile, threshold in thresholds.items():
        point = _event_stats(per_case, threshold=threshold)
        boot_gaps: list[float] = []
        boot_ratios: list[float] = []
        for _ in range(n_boot):
            indices = rng.integers(0, n_cases, n_cases)
            boot = _event_stats(per_case, threshold=threshold, indices=indices)
            gap = float(boot["gap_fail_phi"])
            ratio = float(boot["joint_fail_rate_ratio"])
            if math.isfinite(gap):
                boot_gaps.append(gap)
            if math.isfinite(ratio):
                boot_ratios.append(ratio)
        rows.append(
            {
                "percentile": int(percentile),
                "threshold": threshold,
                **point,
                "case_bootstrap_95ci_gap_fail_phi": _ci95(boot_gaps),
                "case_bootstrap_95ci_joint_fail_rate_ratio": _ci95(boot_ratios),
                "bootstrap": {
                    "B_requested": int(n_boot),
                    "B_completed_gap": int(len(boot_gaps)),
                    "B_completed_ratio": int(len(boot_ratios)),
                    "seed": int(seed),
                },
            }
        )

    source_files = ca._rel_paths([j.path for j in fm_jsons], results_root)
    return {
        "task": task,
        "functional_floor": float(ca.FUNCTIONAL_FLOOR),
        "threshold_definition": (
            "empirical percentile over all model-case Dice values in the "
            "functional pool retained after Dice-floor filtering"
        ),
        "fms_after_floor": sorted({fm for fm, _seed in per_case}),
        "n_members_after_floor": int(len(per_case)),
        "n_cases": int(n_cases),
        "rows": rows,
        "source_files": source_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/diagnostics/failure_event_summary.json"),
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    task_rows = {
        task: _summarize_task(
            args.results_root,
            task,
            n_boot=args.n_boot,
            seed=args.seed + task_index * 1009,
        )
        for task_index, task in enumerate(ca.TASKS)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "same_group_rule": "same FM plus DINOv2-B/S, matching the Table-1 grouping",
        "description": (
            "Binary failure-event same/cross diagnostic computed from released "
            "per-case Dice traces after the Table-1 functional floor."
        ),
        "percentiles": list(PERCENTILES),
        "bootstrap_B": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
        "task_rows": task_rows,
        "summary": {
            "all_p33_gap_cis_exclude_zero": bool(
                all(
                    (row := task_rows[task]["rows"][1])[
                        "case_bootstrap_95ci_gap_fail_phi"
                    ][0]
                    is not None
                    and row["case_bootstrap_95ci_gap_fail_phi"][0] > 0
                    for task in ca.TASKS
                )
            ),
            "p33_gap_range": [
                float(min(task_rows[task]["rows"][1]["gap_fail_phi"] for task in ca.TASKS)),
                float(max(task_rows[task]["rows"][1]["gap_fail_phi"] for task in ca.TASKS)),
            ],
            "p33_joint_fail_ratio_range": [
                float(min(task_rows[task]["rows"][1]["joint_fail_rate_ratio"] for task in ca.TASKS)),
                float(max(task_rows[task]["rows"][1]["joint_fail_rate_ratio"] for task in ca.TASKS)),
            ],
        },
    }
    _write_json(args.out, payload)
    for task in ca.TASKS:
        row = task_rows[task]["rows"][1]
        ci = row["case_bootstrap_95ci_gap_fail_phi"]
        print(
            f"{task}: P33 phi gap={row['gap_fail_phi']:.3f} "
            f"ci=[{ci[0]:.3f},{ci[1]:.3f}] "
            f"joint ratio={row['joint_fail_rate_ratio']:.2f}"
        )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
