#!/usr/bin/env python3
"""Recompute the released M15 single-seed 5-fold CV summary."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "fmpool_m15_cv_summary_v1"
TASKS = ("acdc_lv", "riga_cup")
FMS = (
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "clip_vitl14",
    "convnext_tiny",
    "efficientnet_b0",
    "resnet50",
    "resnet18",
)
FOLDS = (0, 1, 2, 3, 4)
SEED = 42


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
    return value


def _corr_matrix(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    std = data.std(axis=1)
    valid = std >= eps
    out = np.full((data.shape[0], data.shape[0]), np.nan, dtype=np.float64)
    if valid.sum() < 2:
        return out
    centered = data[valid] - data[valid].mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
    corr = (centered @ centered.T) / (denom @ denom.T)
    idx = np.where(valid)[0]
    out[np.ix_(idx, idx)] = corr
    return out


def _load_fold(results_root: Path, task: str, fold: int) -> tuple[dict[str, np.ndarray], list[str], int]:
    rows: dict[str, np.ndarray] = {}
    sources: list[str] = []
    expected_ids: tuple[str, ...] | None = None
    for fm in FMS:
        path = results_root / "per_case_dice_m15" / task / fm / f"fold_{fold}_seed_{SEED}.json"
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ids = tuple(str(x) for x in payload["val_ids"])
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"val_ids mismatch for {path}")
        rows[fm] = np.asarray(payload["per_case_dice"], dtype=np.float64)
        sources.append(str(path.relative_to(results_root)))
    if expected_ids is None:
        raise ValueError(f"no M15 traces for {task} fold {fold}")
    return rows, sorted(sources), len(expected_ids)


def _task_summary(results_root: Path, task: str) -> dict[str, Any]:
    per_fold: list[dict[str, Any]] = []
    all_sources: list[str] = []
    for fold in FOLDS:
        rows, sources, n_val = _load_fold(results_root, task, fold)
        fms = sorted(rows)
        matrix = _corr_matrix(np.stack([rows[fm] for fm in fms], axis=0))
        pair_vals = matrix[np.triu_indices(len(fms), k=1)]
        pair_vals = pair_vals[~np.isnan(pair_vals)]
        mean_dice = {fm: float(rows[fm].mean()) for fm in fms}
        per_fold.append(
            {
                "fold": int(fold),
                "n_val": int(n_val),
                "cross_bundle_rbar": float(np.mean(pair_vals)),
                "n_cross_bundle_pairs": int(len(pair_vals)),
                "pool_mean_dice": float(np.mean(list(mean_dice.values()))),
                "mean_dice_by_fm": mean_dice,
            }
        )
        all_sources.extend(sources)
    rvals = [row["cross_bundle_rbar"] for row in per_fold]
    dice_vals = [row["pool_mean_dice"] for row in per_fold]
    return {
        "task": task,
        "protocol": "8 bundles x 5 folds x one decoder seed; per-fold cross-bundle correlation on the held-out fold",
        "seed": SEED,
        "fms": list(FMS),
        "folds": list(FOLDS),
        "per_fold": per_fold,
        "cross_bundle_rbar_mean": float(np.mean(rvals)),
        "cross_bundle_rbar_std": float(np.std(rvals, ddof=1)),
        "pool_mean_dice_mean": float(np.mean(dice_vals)),
        "pool_mean_dice_std": float(np.std(dice_vals, ddof=1)),
        "source_files": sorted(all_sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "tasks": {task: _task_summary(args.results_root, task) for task in TASKS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(out), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
