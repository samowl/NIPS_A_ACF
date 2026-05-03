#!/usr/bin/env python3
"""Quality-controlled pairwise-correlation diagnostic for the artifact bundle.

The primary paper estimator compares the mean Pearson correlation of same-FM
decoder-seed pairs against Table-1 cross-family pairs. A natural alternative
concern is that the gap could be mostly a quality artifact: similarly accurate
members may have similarly shaped per-case Dice traces. This script therefore
fits, for each primary task row, a pair-level OLS model

    r_ab ~ same_group_ab
           + pair_mean_dice_ab
           + abs_mean_dice_diff_ab
           + avg_case_variance_ab
           + abs_case_variance_diff_ab

where ``same_group`` uses the exact Table-1 grouping rule: same FM plus the
DINOv2-B/S within-family pair. Case bootstrap intervals resample test units and
refit the same model. The diagnostic uses only released per-case Dice JSONs.
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

from fmpool.estimators import functional_floor_filter, within_cross_rbar  # noqa: E402

import compute_all as ca  # noqa: E402


SCHEMA_VERSION = "fmpool_quality_controlled_pairs_v1"
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


def _ci95(values: list[float]) -> list[float | None]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


def _is_same_group(fm_a: str, fm_b: str) -> bool:
    return fm_a == fm_b or frozenset({fm_a, fm_b}) in ca.DINOV2_FAMILY_PAIRS


def _pair_table(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    keys = sorted(per_case)
    if indices is None:
        data = {k: np.asarray(per_case[k], dtype=np.float64) for k in keys}
    else:
        data = {k: np.asarray(per_case[k], dtype=np.float64)[indices] for k in keys}

    rows: list[list[float]] = []
    records: list[dict[str, Any]] = []
    feature_names = [
        "intercept",
        "same_group",
        "pair_mean_dice",
        "abs_mean_dice_diff",
        "avg_case_variance",
        "abs_case_variance_diff",
    ]
    for i, key_a in enumerate(keys):
        arr_a = data[key_a]
        mean_a = float(np.mean(arr_a))
        var_a = float(np.var(arr_a))
        for key_b in keys[i + 1:]:
            arr_b = data[key_b]
            if arr_a.size < 2 or float(np.std(arr_a)) < 1e-8 or float(np.std(arr_b)) < 1e-8:
                continue
            r = float(np.corrcoef(arr_a, arr_b)[0, 1])
            if not math.isfinite(r):
                continue
            mean_b = float(np.mean(arr_b))
            var_b = float(np.var(arr_b))
            same_group = 1.0 if _is_same_group(key_a[0], key_b[0]) else 0.0
            rows.append(
                [
                    r,
                    1.0,
                    same_group,
                    (mean_a + mean_b) / 2.0,
                    abs(mean_a - mean_b),
                    (var_a + var_b) / 2.0,
                    abs(var_a - var_b),
                ]
            )
            records.append(
                {
                    "a": {"fm": key_a[0], "seed": key_a[1]},
                    "b": {"fm": key_b[0], "seed": key_b[1]},
                    "pearson_r": r,
                    "same_group": bool(same_group),
                    "pair_mean_dice": (mean_a + mean_b) / 2.0,
                    "abs_mean_dice_diff": abs(mean_a - mean_b),
                    "avg_case_variance": (var_a + var_b) / 2.0,
                    "abs_case_variance_diff": abs(var_a - var_b),
                }
            )
    if not rows:
        raise RuntimeError("no valid pair rows after functional filtering")
    return np.asarray(rows, dtype=np.float64), records, feature_names


def _ols(pair_rows: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    y = pair_rows[:, 0]
    x = pair_rows[:, 1:]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = float("nan") if ss_tot <= 1e-24 else float(1.0 - ss_res / ss_tot)
    return {
        "formula": (
            "pearson_r ~ same_group + pair_mean_dice + abs_mean_dice_diff + "
            "avg_case_variance + abs_case_variance_diff"
        ),
        "feature_names": feature_names,
        "coefficients": {name: float(value) for name, value in zip(feature_names, beta)},
        "r2": r2,
        "n_pair_rows": int(len(y)),
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
        raise RuntimeError(f"{task}: fewer than two traces after floor")

    point = within_cross_rbar(
        per_case,
        family_pairs_within=ca.DINOV2_FAMILY_PAIRS,
        stat="pearson",
    )
    pair_rows, records, feature_names = _pair_table(per_case)
    fit = _ols(pair_rows, feature_names)
    same = pair_rows[:, 2].astype(bool)

    beta_boot: list[float] = []
    rng = np.random.default_rng(seed)
    n_cases = len(next(iter(per_case.values())))
    for _ in range(n_boot):
        indices = rng.integers(0, n_cases, n_cases)
        try:
            sampled_rows, _sampled_records, sampled_features = _pair_table(
                per_case,
                indices=indices,
            )
            sampled_fit = _ols(sampled_rows, sampled_features)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
        beta = sampled_fit["coefficients"]["same_group"]
        if math.isfinite(beta):
            beta_boot.append(float(beta))

    source_files = ca._rel_paths([j.path for j in fm_jsons], results_root)
    return {
        "task": task,
        "functional_floor": float(ca.FUNCTIONAL_FLOOR),
        "n_cases": int(n_cases),
        "n_members_after_floor": int(len(per_case)),
        "n_fms_after_floor": int(len({fm for fm, _seed in per_case})),
        "n_same_group_pairs": int(np.sum(same)),
        "n_cross_group_pairs": int(np.sum(~same)),
        "raw_within_cross": point,
        "raw_gap_from_pair_rows": float(
            np.mean(pair_rows[same, 0]) - np.mean(pair_rows[~same, 0])
        ),
        "quality_control_model": fit,
        "same_group_beta": float(fit["coefficients"]["same_group"]),
        "same_group_beta_ci95_case_bootstrap": _ci95(beta_boot),
        "bootstrap": {
            "B_requested": int(n_boot),
            "B_completed": int(len(beta_boot)),
            "seed": int(seed),
            "unit": "test case / slice; slice-derived tasks are case-bootstrap diagnostics",
        },
        "pair_records": records,
        "source_files": source_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/diagnostics/quality_controlled_pairs.json"),
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    rows = [
        _summarize_task(
            args.results_root,
            task,
            n_boot=args.n_boot,
            seed=args.seed + task_index * 1009,
        )
        for task_index, task in enumerate(ca.TASKS)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "script": "code/scripts/compute_quality_controlled_pairs.py",
        "description": (
            "Pair-level OLS diagnostic controlling for marginal quality and "
            "per-case variance before estimating the same-FM/Table-1 same-group "
            "association with pairwise Pearson correlation."
        ),
        "same_group_rule": "same FM plus DINOv2-B/S, matching the Table-1 grouping",
        "task_rows": {row["task"]: row for row in rows},
        "summary": {
            "same_group_beta_min": float(min(row["same_group_beta"] for row in rows)),
            "same_group_beta_max": float(max(row["same_group_beta"] for row in rows)),
            "all_same_group_betas_positive": bool(all(row["same_group_beta"] > 0 for row in rows)),
            "all_same_group_beta_cis_exclude_zero": bool(
                all(
                    (ci := row["same_group_beta_ci95_case_bootstrap"])[0] is not None
                    and ci[0] > 0
                    for row in rows
                )
            ),
        },
    }
    _write_json(args.out, payload)
    for row in rows:
        ci = row["same_group_beta_ci95_case_bootstrap"]
        print(
            f"{row['task']}: beta_same={row['same_group_beta']:.3f} "
            f"ci=[{ci[0]:.3f},{ci[1]:.3f}] raw_gap={row['raw_within_cross']['gap']:.3f}"
        )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
