#!/usr/bin/env python3
"""Recompute the released RIGA cross-vendor distribution-shift diagnostic.

This script consumes only released per-case Dice JSON files. It intentionally
uses a fixed eight-bundle pool for both the ID and MESSIDOR OOD rows so that the
reported table is traceable to public bundle contents without hidden floor or
weighting conventions.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "fmpool_distshift_v1"
SEEDS = (42, 43, 44, 45)
FMS = (
    "biomedclip",
    "clip_vitl14",
    "convnext_tiny",
    "dinov2_vitb14",
    "dinov2_vits14",
    "efficientnet_b0",
    "resnet18",
    "resnet50",
)
FAMILY = {
    "biomedclip": "biomedclip",
    "clip_vitl14": "clip",
    "convnext_tiny": "convnext",
    "dinov2_vitb14": "dinov2",
    "dinov2_vits14": "dinov2",
    "efficientnet_b0": "efficientnet",
    "resnet18": "resnet",
    "resnet50": "resnet",
}
TASKS = {
    "id_reference": "riga_cup",
    "ood_messidor": "riga_cup_ood_messidor",
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
    return value


def _load_pool(results_root: Path, task: str) -> tuple[dict[tuple[str, int], np.ndarray], list[str], int]:
    per_case: dict[tuple[str, int], np.ndarray] = {}
    source_files: list[str] = []
    expected_ids: tuple[str, ...] | None = None
    task_root = results_root / "per_case_dice" / task
    for fm in FMS:
        for seed in SEEDS:
            path = task_root / fm / f"seed_{seed}.json"
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            ids = tuple(str(x) for x in payload["test_ids"])
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"test_ids mismatch for {path}")
            per_case[(fm, seed)] = np.asarray(payload["per_case_dice"], dtype=np.float64)
            source_files.append(str(path.relative_to(results_root)))
    if expected_ids is None:
        raise ValueError(f"no traces under {task_root}")
    return per_case, sorted(source_files), len(expected_ids)


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


def _point(per_case: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    keys = sorted(per_case)
    matrix = _corr_matrix(np.stack([per_case[k] for k in keys], axis=0))
    within: list[float] = []
    cross: list[float] = []
    for i, (fm_a, _seed_a) in enumerate(keys):
        for j in range(i + 1, len(keys)):
            fm_b, _seed_b = keys[j]
            value = float(matrix[i, j])
            if math.isnan(value):
                continue
            if fm_a == fm_b or FAMILY[fm_a] == FAMILY[fm_b]:
                within.append(value)
            else:
                cross.append(value)
    within_mean = float(np.mean(within))
    cross_mean = float(np.mean(cross))
    return {
        "within_rbar": within_mean,
        "cross_rbar": cross_mean,
        "gap": within_mean - cross_mean,
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
    }


def _case_bootstrap(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_case.values())))
    rows = {"within_rbar": [], "cross_rbar": [], "gap": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = {k: v[idx] for k, v in per_case.items()}
        point = _point(boot)
        for key in rows:
            rows[key].append(float(point[key]))
    return {
        key: [
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
        ]
        for key, vals in rows.items()
    }


def _summarize(results_root: Path, task: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    per_case, sources, n_cases = _load_pool(results_root, task)
    point = _point(per_case)
    point.update(
        {
            "task": task,
            "pool": "fixed 8-bundle x 4-seed",
            "n_cases": n_cases,
            "fms": list(FMS),
            "family_rule": "within = same FM or same broad family (DINOv2-B/S, ResNet-18/50); cross = otherwise",
            "floor": None,
            "case_bootstrap_95ci": _case_bootstrap(per_case, n_boot=n_boot, seed=seed),
            "source_files": sources,
        }
    )
    return point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = {
        name: _summarize(args.results_root, task, n_boot=args.n_boot, seed=args.seed)
        for name, task in TASKS.items()
    }
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "rows": rows,
        "gap_shift_ood_minus_id": rows["ood_messidor"]["gap"] - rows["id_reference"]["gap"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(out), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
