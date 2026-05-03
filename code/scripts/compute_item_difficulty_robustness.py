#!/usr/bin/env python3
"""CPU-only item-difficulty robustness diagnostics.

This script uses only released per-case Dice JSONs. It computes two artifact
checks that do not require raw images, feature caches, checkpoints, or GPUs:

1. Functional-floor grid over the prespecified floor values.
2. Cross-fitted item-difficulty residualization. For each FM trace, item
   difficulty is estimated from traces outside that FM's broad upstream family;
   the trace is residualized against that held-out estimate before recomputing
   the within/cross gap.
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


SCHEMA_VERSION = "fmpool_item_difficulty_robustness_v1"
TASKS = ca.TASKS
DEFAULT_FLOORS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PRIMARY_FLOOR = 0.30
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0

BROAD_FAMILY = {
    "dinov2_vitb14": "dinov2",
    "dinov2_vits14": "dinov2",
    "dinov2_vitl14": "dinov2",
    "dinov2_vitg14": "dinov2",
    "clip_vitb16": "openai_clip",
    "clip_vitb32": "openai_clip",
    "clip_vitl14": "openai_clip",
    "resnet18": "resnet",
    "resnet34": "resnet",
    "resnet50": "resnet",
    "resnet101": "resnet",
    "convnext_tiny": "convnext",
    "convnext_small": "convnext",
    "convnext_base": "convnext",
    "efficientnet_b0": "efficientnet",
    "efficientnet_b1": "efficientnet",
    "efficientnet_b2": "efficientnet",
    "efficientnet_b3": "efficientnet",
    "mae_vitb16": "mae",
    "mae_vitl16": "mae",
    "deit_vitb16": "deit",
    "deit_vits16": "deit",
    "deit_vitt16": "deit",
    "biomedclip": "biomedclip",
    "retfound_vitl16": "retfound",
    "medsam_vitb16": "medsam",
}


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


def _write_json(path: Path, payload: dict) -> None:
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


def _fm_mean(per_case: dict[tuple[str, int], np.ndarray]) -> dict[str, float]:
    by_fm: dict[str, list[np.ndarray]] = {}
    for (fm, _seed), arr in per_case.items():
        by_fm.setdefault(fm, []).append(np.asarray(arr, dtype=np.float64))
    return {
        fm: float(np.mean([float(np.mean(arr)) for arr in arrs]))
        for fm, arrs in by_fm.items()
    }


def _point(per_case: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    stats = within_cross_rbar(
        per_case,
        family_pairs_within=ca.DINOV2_FAMILY_PAIRS,
        stat="pearson",
    )
    return {
        "within_rbar": float(stats["within"]),
        "cross_rbar": float(stats["cross"]),
        "gap": float(stats["gap"]),
        "n_within_pairs": int(stats["n_within_pairs"]),
        "n_cross_pairs": int(stats["n_cross_pairs"]),
    }


def _family(fm: str) -> str:
    return BROAD_FAMILY.get(fm, fm)


def _residualize_one(y: np.ndarray, difficulty: np.ndarray) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if y.shape != difficulty.shape:
        raise ValueError(f"residualization shape mismatch: {y.shape} vs {difficulty.shape}")
    if y.size < 2 or float(np.std(difficulty)) < 1e-12:
        residual = y - float(np.mean(y))
        return residual, {"intercept": float(np.mean(y)), "slope": 0.0, "r2": 0.0}
    x = np.column_stack([np.ones_like(difficulty), difficulty])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    r2 = 0.0 if ss_tot <= 1e-24 else max(0.0, 1.0 - ss_res / ss_tot)
    return residual, {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(r2),
    }


def _crossfit_residualize(
    per_case: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    keys = sorted(per_case)
    residualized: dict[tuple[str, int], np.ndarray] = {}
    fits: dict[str, dict] = {}
    skipped: list[str] = []
    for key in keys:
        fm, seed = key
        fam = _family(fm)
        outside = [
            np.asarray(per_case[k], dtype=np.float64)
            for k in keys
            if _family(k[0]) != fam
        ]
        if not outside:
            skipped.append(f"{fm}:{seed}")
            continue
        difficulty = np.mean(np.stack(outside, axis=0), axis=0)
        residual, fit = _residualize_one(np.asarray(per_case[key], dtype=np.float64), difficulty)
        residualized[key] = residual
        fits[f"{fm}:{seed}"] = {
            **fit,
            "heldout_family": fam,
            "n_heldout_traces": int(len(outside)),
        }
    if len(residualized) < 2:
        raise RuntimeError("not enough traces after cross-fitted residualization")
    r2_values = [fit["r2"] for fit in fits.values()]
    return residualized, {
        "n_residualized_traces": int(len(residualized)),
        "skipped_traces": skipped,
        "fit_r2_mean": float(np.mean(r2_values)) if r2_values else float("nan"),
        "fit_r2_median": float(np.median(r2_values)) if r2_values else float("nan"),
        "fit_r2_range": [
            float(np.min(r2_values)) if r2_values else float("nan"),
            float(np.max(r2_values)) if r2_values else float("nan"),
        ],
        "trace_fits": fits,
    }


def _attenuation(raw_gap: float, residual_gap: float) -> float:
    if not math.isfinite(raw_gap) or abs(raw_gap) < 1e-12:
        return float("nan")
    return float(residual_gap / raw_gap)


def _item_difficulty_analysis(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    residualized, fit_summary = _crossfit_residualize(per_case)
    common_keys = sorted(residualized)
    raw_common = {k: np.asarray(per_case[k], dtype=np.float64) for k in common_keys}
    raw = _point(raw_common)
    residual = _point(residualized)
    raw_gap = float(raw["gap"])
    residual_gap = float(residual["gap"])

    boot_raw: list[float] = []
    boot_residual: list[float] = []
    boot_attenuation: list[float] = []
    rng = np.random.default_rng(seed)
    n = len(next(iter(raw_common.values())))
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sampled = {k: arr[idx] for k, arr in raw_common.items()}
        try:
            sampled_residual, _fit = _crossfit_residualize(sampled)
        except RuntimeError:
            continue
        sampled_raw_gap = float(_point(sampled)["gap"])
        sampled_residual_gap = float(_point(sampled_residual)["gap"])
        if math.isfinite(sampled_raw_gap):
            boot_raw.append(sampled_raw_gap)
        if math.isfinite(sampled_residual_gap):
            boot_residual.append(sampled_residual_gap)
        att = _attenuation(sampled_raw_gap, sampled_residual_gap)
        if math.isfinite(att):
            boot_attenuation.append(att)

    return {
        "description": (
            "Each FM trace is residualized against per-case difficulty estimated "
            "from traces outside that FM's broad upstream family; the original "
            "within/cross family-pair rule is then recomputed on residuals."
        ),
        "difficulty_family_rule": "broad upstream family exclusion",
        "within_cross_family_rule": "paper primary rule: same FM plus DINOv2-B/S pair",
        "raw": raw,
        "residualized": residual,
        "residualized_gap_minus_raw_gap": float(residual_gap - raw_gap),
        "attenuation_ratio_residual_over_raw": _attenuation(raw_gap, residual_gap),
        "bootstrap": {
            "B": int(n_boot),
            "seed": int(seed),
            "raw_gap_95ci": _ci95(boot_raw),
            "residualized_gap_95ci": _ci95(boot_residual),
            "attenuation_ratio_95ci": _ci95(boot_attenuation),
        },
        "fit_summary": fit_summary,
    }


def _floor_grid(
    per_case_all: dict[tuple[str, int], np.ndarray],
    *,
    task: str,
    unit_ids: tuple[str, ...],
    floors: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_subject_units = len(set(unit_ids)) < len(unit_ids)
    for floor in floors:
        if floor <= 0:
            per_case = dict(per_case_all)
        else:
            per_case = functional_floor_filter(
                per_case_all,
                min_mean_dice=float(floor),
                scope="per_fm",
            )
        fms = sorted({fm for fm, _seed in per_case})
        row: dict[str, Any] = {
            "floor": float(floor),
            "n_fms": int(len(fms)),
            "n_members": int(len(per_case)),
            "fms_kept": fms,
            "fm_mean_dice": _fm_mean(per_case),
            "case_level": _point(per_case) if len(per_case) >= 2 else None,
        }
        if has_subject_units and len(per_case) >= 2:
            aggregated, units = ca._aggregate_by_unit(per_case, unit_ids)
            row["subject_level"] = {
                "n_units": int(len(units)),
                **_point(aggregated),
            }
        rows.append(row)
    return rows


def _task_analysis(
    results_root: Path,
    task: str,
    *,
    floors: tuple[float, ...],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    fm_jsons, _nnunet_jsons = ca._collect_task_jsons(results_root, task)
    if not fm_jsons:
        raise FileNotFoundError(f"no per-case Dice JSONs found for task {task!r}")
    ca._validate_alignment(fm_jsons)
    per_case_all = ca._as_per_case_map(fm_jsons)
    meta = ca._case_metadata(task, fm_jsons[0].test_ids)
    unit_ids = tuple(str(x) for x in meta["unit_ids"])

    primary = functional_floor_filter(
        per_case_all,
        min_mean_dice=PRIMARY_FLOOR,
        scope="per_fm",
    )
    out: dict[str, Any] = {
        "task": task,
        "n_test": int(len(fm_jsons[0].test_ids)),
        "case_unit": meta["case_unit"],
        "cluster_unit": meta["cluster_unit"],
        "n_units": int(meta["n_units"]),
        "source_files": ca._rel_paths([j.path for j in fm_jsons], results_root),
        "available_fms": sorted({fm for fm, _seed in per_case_all}),
        "available_members": int(len(per_case_all)),
        "floor_grid": _floor_grid(
            per_case_all,
            task=task,
            unit_ids=unit_ids,
            floors=floors,
        ),
        "cross_fitted_item_difficulty_floor_0.30": _item_difficulty_analysis(
            primary,
            n_boot=n_boot,
            seed=seed,
        ),
    }
    if meta["cluster_unit"] is not None:
        aggregated, units = ca._aggregate_by_unit(primary, unit_ids)
        out["subject_level_cross_fitted_item_difficulty_floor_0.30"] = {
            "n_units": int(len(units)),
            **_item_difficulty_analysis(
                aggregated,
                n_boot=n_boot,
                seed=seed,
            ),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--floors", nargs="*", type=float, default=list(DEFAULT_FLOORS))
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    floors = tuple(float(x) for x in args.floors)
    task_rows = {
        task: _task_analysis(
            results_root,
            task,
            floors=floors,
            n_boot=int(args.n_boot),
            seed=int(args.seed),
        )
        for task in args.tasks
    }
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(results_root),
        "primary_floor": PRIMARY_FLOOR,
        "functional_floor_grid": [float(x) for x in floors],
        "bootstrap_B": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
        "broad_family_rule": BROAD_FAMILY,
        "task_rows": task_rows,
    }
    _write_json(args.out, out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
