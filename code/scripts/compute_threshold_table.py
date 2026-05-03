#!/usr/bin/env python3
"""Recompute the RIGA Cup failure-threshold appendix table."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "fmpool_threshold_table_v1"
TASK = "riga_cup"
PRIMARY_FMS = (
    "biomedclip",
    "clip_vitb16",
    "clip_vitl14",
    "convnext_tiny",
    "deit_vitb16",
    "dinov2_vitb14",
    "dinov2_vits14",
    "efficientnet_b0",
    "mae_vitb16",
    "resnet18",
    "resnet50",
)
SEEDS = (42, 43, 44, 45)
PERCENTILES = (25, 33, 50)
FUNCTIONAL_FLOOR = 0.30


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


def _load(results_root: Path) -> tuple[dict[tuple[str, int], np.ndarray], list[str]]:
    per_case: dict[tuple[str, int], np.ndarray] = {}
    source_files: list[str] = []
    expected_ids: tuple[str, ...] | None = None
    for fm in PRIMARY_FMS:
        for seed in SEEDS:
            path = results_root / "per_case_dice" / TASK / fm / f"seed_{seed}.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            ids = tuple(str(x) for x in payload["test_ids"])
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"test_ids mismatch for {path}")
            per_case[(fm, seed)] = np.asarray(payload["per_case_dice"], dtype=np.float64)
            source_files.append(str(path.relative_to(results_root)))
    means = {
        fm: float(np.mean([arr.mean() for (f, _s), arr in per_case.items() if f == fm]))
        for fm in sorted({f for f, _s in per_case})
    }
    keep = {fm for fm, mean in means.items() if mean >= FUNCTIONAL_FLOOR}
    return {k: v for k, v in per_case.items() if k[0] in keep}, sorted(source_files)


def _corr_binary(a: np.ndarray, b: np.ndarray) -> float:
    ax = a.astype(np.float64)
    bx = b.astype(np.float64)
    ax = ax - ax.mean()
    bx = bx - bx.mean()
    denom = float(np.sqrt(np.sum(ax * ax) * np.sum(bx * bx)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(ax * bx) / denom)


def _mean_pair_corr(arrays: list[np.ndarray]) -> float:
    vals: list[float] = []
    for i, a in enumerate(arrays):
        for b in arrays[i + 1:]:
            value = _corr_binary(a, b)
            if not math.isnan(value):
                vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def _rows(per_case: dict[tuple[str, int], np.ndarray]) -> list[dict[str, Any]]:
    fms = sorted({fm for fm, _seed in per_case})
    thresholds = {
        p: float(np.percentile(np.concatenate(list(per_case.values())), p))
        for p in PERCENTILES
    }
    rows: list[dict[str, Any]] = []
    for percentile, threshold in thresholds.items():
        mono_corrs: list[float] = []
        mono_all_fail: list[float] = []
        for fm in fms:
            fail_arrays = [
                per_case[(fm, seed)] < threshold
                for seed in SEEDS
                if (fm, seed) in per_case
            ]
            if len(fail_arrays) != 4:
                continue
            mono_corrs.append(_mean_pair_corr(fail_arrays))
            mono_all_fail.append(float(np.mean(np.all(np.stack(fail_arrays), axis=0))))

        diverse_corrs: list[float] = []
        diverse_all_fail: list[float] = []
        n_seed_tuples = 0
        for fm_combo in itertools.combinations(fms, 4):
            choices = [
                [per_case[(fm, seed)] < threshold for seed in SEEDS if (fm, seed) in per_case]
                for fm in fm_combo
            ]
            if any(len(choice) == 0 for choice in choices):
                continue
            for seed_tuple in itertools.product(*choices):
                arrays = list(seed_tuple)
                diverse_corrs.append(_mean_pair_corr(arrays))
                diverse_all_fail.append(float(np.mean(np.all(np.stack(arrays), axis=0))))
                n_seed_tuples += 1

        rows.append(
            {
                "percentile": int(percentile),
                "threshold": float(threshold),
                "mono_fail_rbar": float(np.nanmean(mono_corrs)),
                "diverse_fail_rbar": float(np.nanmean(diverse_corrs)),
                "mono_all_fail_rate": float(np.nanmean(mono_all_fail)),
                "diverse_all_fail_rate": float(np.nanmean(diverse_all_fail)),
                "n_mono_pools": len(mono_all_fail),
                "n_diverse_seed_tuples": n_seed_tuples,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    per_case, source_files = _load(args.results_root)
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "task": TASK,
        "functional_floor": FUNCTIONAL_FLOOR,
        "threshold_definition": "empirical percentile over all model-case Dice values in the functional RIGA Cup primary pool",
        "fms": sorted({fm for fm, _seed in per_case}),
        "n_cases": int(len(next(iter(per_case.values())))),
        "n_members": int(len(per_case)),
        "source_files": source_files,
        "rows": _rows(per_case),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(out), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
