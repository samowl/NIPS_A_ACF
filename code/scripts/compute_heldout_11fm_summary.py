#!/usr/bin/env python3
"""Summarize per-case held-out rerun traces.

The input is a per-case root produced by ``run_worker_v2.sh`` with
``FMPOOL_PER_CASE_ROOT`` set to an isolated held-out directory. The script does
not require raw data, feature caches, checkpoints, GPUs, or Torch.
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

from fmpool.estimators import compute_m_eff, functional_floor_filter, within_cross_rbar  # noqa: E402

import compute_all as ca  # noqa: E402


SCHEMA_VERSION = "fmpool_heldout_11fm_v1"
TASKS = ["msd_hippocampus", "isic2018", "msd_heart", "msd_prostate", "msd_spleen"]
EXPECTED_FMS = [
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "clip_vitl14",
    "clip_vitb16",
    "mae_vitb16",
    "deit_vitb16",
    "convnext_tiny",
    "efficientnet_b0",
    "resnet50",
    "resnet18",
]
EXPECTED_SEEDS = [42, 43, 44, 45]
FLOORS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PRIMARY_FLOOR = 0.30
TAU = 0.85


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


def _load_task(per_case_root: Path, task: str) -> list[ca.PerCaseJSON]:
    task_root = per_case_root / task
    if not task_root.is_dir():
        return []
    rows = [
        ca._load_per_case_json(path)
        for path in sorted(task_root.glob("*/seed_*.json"))
    ]
    rows = [row for row in rows if row.task == task]
    ca._validate_alignment(rows)
    return rows


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


def _floor_rows(per_case_all: dict[tuple[str, int], np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    for floor in FLOORS:
        if floor <= 0:
            per_case = dict(per_case_all)
        else:
            per_case = functional_floor_filter(per_case_all, min_mean_dice=floor, scope="per_fm")
        fms = sorted({fm for fm, _seed in per_case})
        row = {
            "floor": float(floor),
            "n_fms": int(len(fms)),
            "n_members": int(len(per_case)),
            "fms_kept": fms,
            "within_cross": _point(per_case) if len(per_case) >= 2 else None,
        }
        if len(per_case) >= 2:
            meff = compute_m_eff(per_case, tau=TAU)
            row["m_eff"] = {
                "tau": TAU,
                "rho_fail": float(meff["rho_fail"]),
                "M_eff": float(meff["M_eff"]),
                "n_fm_seed": int(meff["n_fm_seed"]),
            }
        rows.append(row)
    return rows


def _task_summary(per_case_root: Path, task: str) -> dict:
    jsons = _load_task(per_case_root, task)
    expected_keys = {(fm, seed) for fm in EXPECTED_FMS for seed in EXPECTED_SEEDS}
    if not jsons:
        return {
            "task": task,
            "status": "missing",
            "n_traces": 0,
            "expected_n_traces": int(len(expected_keys)),
            "missing_traces": [f"{fm}:seed_{seed}" for fm, seed in sorted(expected_keys)],
            "extra_traces": [],
        }
    per_case_all = ca._as_per_case_map(jsons)
    keys = set(per_case_all)
    missing = sorted(expected_keys - keys)
    extra = sorted(keys - expected_keys)
    status = "ok" if not missing and not extra else "partial"
    primary = functional_floor_filter(per_case_all, min_mean_dice=PRIMARY_FLOOR, scope="per_fm")
    return {
        "task": task,
        "status": status,
        "n_test": int(len(jsons[0].test_ids)),
        "n_traces": int(len(per_case_all)),
        "expected_n_traces": int(len(expected_keys)),
        "missing_traces": [f"{fm}:seed_{seed}" for fm, seed in missing],
        "extra_traces": [f"{fm}:seed_{seed}" for fm, seed in extra],
        "available_fms": sorted({fm for fm, _seed in per_case_all}),
        "source_files": [str(path.relative_to(per_case_root)) for path in sorted(j.path for j in jsons)],
        "floor_grid": _floor_rows(per_case_all),
        "primary_floor_0.30": {
            "fms_kept": sorted({fm for fm, _seed in primary}),
            "n_members": int(len(primary)),
            "within_cross": _point(primary) if len(primary) >= 2 else None,
            "m_eff": compute_m_eff(primary, tau=TAU) if len(primary) >= 2 else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = {task: _task_summary(args.per_case_root, task) for task in args.tasks}
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "per_case_root": str(args.per_case_root),
        "primary_floor": PRIMARY_FLOOR,
        "tau": TAU,
        "task_rows": rows,
    }
    _write_json(args.out, out)
    print(args.out)
    partial = {
        task: row for task, row in rows.items()
        if row.get("status") != "ok"
    }
    if partial:
        print(
            "HELDOUT_11FM_INCOMPLETE "
            + ", ".join(
                f"{task}:{row.get('status')}:{row.get('n_traces', 0)}/"
                f"{row.get('expected_n_traces', 0)}"
                for task, row in partial.items()
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
