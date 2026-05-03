#!/usr/bin/env python3
"""Recompute the appendix tau-sweep from released per-case Dice traces."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "fmpool_tau_sweep_v1"
TASKS = ("riga_cup", "acdc_lv", "kvasir", "brats_wt")
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
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90)
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


def _load_task(results_root: Path, task: str) -> tuple[dict[tuple[str, int], np.ndarray], list[str]]:
    per_case: dict[tuple[str, int], np.ndarray] = {}
    source_files: list[str] = []
    expected_ids: tuple[str, ...] | None = None
    for fm in PRIMARY_FMS:
        for seed in SEEDS:
            path = results_root / "per_case_dice" / task / fm / f"seed_{seed}.json"
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


def _recovery(arrays: list[np.ndarray], tau: float) -> float:
    data = np.stack(arrays, axis=0)
    any_fail = np.any(data < tau, axis=0)
    # Dice-only proxy for ensemble recovery: the pool-average Dice crosses tau
    # on a case where at least one member is below tau.
    ensemble_pass = np.mean(data, axis=0) >= tau
    denom = int(any_fail.sum())
    if denom == 0:
        return float("nan")
    return float(np.logical_and(any_fail, ensemble_pass).sum() / denom)


def _task_rows(per_case: dict[tuple[str, int], np.ndarray]) -> list[dict[str, Any]]:
    fms = sorted({fm for fm, _seed in per_case})
    rows: list[dict[str, Any]] = []
    for tau in TAUS:
        mono_vals: list[float] = []
        for fm in fms:
            arrays = [per_case[(fm, seed)] for seed in SEEDS if (fm, seed) in per_case]
            if len(arrays) == 4:
                mono_vals.append(_recovery(arrays, tau))

        diverse_vals: list[float] = []
        n_fm_compositions = 0
        n_seed_tuples = 0
        for fm_combo in itertools.combinations(fms, 4):
            choices = [
                [per_case[(fm, seed)] for seed in SEEDS if (fm, seed) in per_case]
                for fm in fm_combo
            ]
            if any(len(choice) == 0 for choice in choices):
                continue
            n_fm_compositions += 1
            for seed_tuple in itertools.product(*choices):
                diverse_vals.append(_recovery(list(seed_tuple), tau))
                n_seed_tuples += 1

        mono = float(np.nanmean(mono_vals))
        diverse = float(np.nanmean(diverse_vals))
        rows.append(
            {
                "tau": float(tau),
                "mono_recovery": mono,
                "diverse_recovery": diverse,
                "delta": diverse - mono,
                "n_mono_pools": len(mono_vals),
                "n_diverse_fm_compositions": n_fm_compositions,
                "n_diverse_seed_tuples": n_seed_tuples,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tasks: dict[str, Any] = {}
    for task in TASKS:
        per_case, source_files = _load_task(args.results_root, task)
        tasks[task] = {
            "functional_floor": FUNCTIONAL_FLOOR,
            "fms": sorted({fm for fm, _seed in per_case}),
            "n_members": len(per_case),
            "source_files": source_files,
            "rows": _task_rows(per_case),
        }
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "ensemble_pass_definition": "mean per-case Dice of the 4-member pool is >= tau",
        "condition": "at least one pool member has per-case Dice < tau",
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(out), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
