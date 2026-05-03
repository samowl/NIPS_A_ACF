"""Summarize an M14 standard segmentation-recipe matrix for one task."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from fmpool.estimators import case_bootstrap, functional_floor_filter, within_cross_rbar

SCHEMA_VERSION = "m14_clinical_summary_v1"
DINOV2_FAMILY_PAIRS = frozenset(
    [frozenset({"dinov2_vitb14", "dinov2_vits14"})]
)
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
FUNCTIONAL_FLOOR = 0.30


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _ci_list(ci: tuple[float, float]) -> list[float]:
    return [float(ci[0]), float(ci[1])]


def _strict_json_value(value):
    if isinstance(value, dict):
        return {k: _strict_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summarize_m14(results_root: Path, task: str = "riga_cup") -> dict:
    root = results_root / "per_case_dice_m14" / task
    paths = sorted(root.glob("*/seed_*.json"))
    if not paths:
        raise RuntimeError(f"no M14 JSONs under {root}")

    payloads = [_load(path) for path in paths]
    test_ids = payloads[0]["test_ids"]
    per_case: dict[tuple[str, int], np.ndarray] = {}
    source_files: list[str] = []
    protocols: set[str] = set()

    for payload, path in zip(payloads, paths, strict=True):
        if payload["task"] != task:
            raise ValueError(f"{path}: unexpected task {payload['task']!r}")
        if payload["test_ids"] != test_ids:
            raise ValueError(f"{path}: test_ids mismatch")
        key = (str(payload["fm"]), int(payload["seed"]))
        if key in per_case:
            raise ValueError(f"duplicate M14 key {key} at {path}")
        per_case[key] = np.asarray(payload["per_case_dice"], dtype=float)
        source_files.append(_rel(path, results_root))
        protocols.add(str(payload.get("protocol", "m14_clinical_v1")))

    per_case = functional_floor_filter(
        per_case, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm"
    )
    point = within_cross_rbar(
        per_case, family_pairs_within=set(DINOV2_FAMILY_PAIRS), stat="pearson"
    )
    boot = case_bootstrap(
        per_case,
        family_pairs_within=set(DINOV2_FAMILY_PAIRS),
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )

    by_fm: dict[str, list[float]] = {}
    for (fm, _seed), values in per_case.items():
        by_fm.setdefault(fm, []).append(float(np.mean(values)))
    per_fm = [
        {
            "fm": fm,
            "n_seeds": len(vals),
            "mean_dice": float(np.mean(vals)),
            "min_dice": float(np.min(vals)),
            "max_dice": float(np.max(vals)),
        }
        for fm, vals in sorted(by_fm.items())
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "protocol": (
            "M14 standard segmentation-recipe sensitivity: frozen encoder, Dice+BCE loss, "
            "horizontal flip and +/-15 degree rotation augmentation, 224x224 input."
        ),
        "source_protocol_values": sorted(protocols),
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "functional_floor": FUNCTIONAL_FLOOR,
        "n_test": int(len(test_ids)),
        "pool": f"{len(by_fm)}-FM x 4-seed ({len(per_case)} members)",
        "fms_kept_after_floor": sorted(by_fm),
        "per_fm": per_fm,
        "within_cross": {
            "within": float(point["within"]),
            "cross": float(point["cross"]),
            "gap": float(point["gap"]),
            "n_within_pairs": int(point["n_within_pairs"]),
            "n_cross_pairs": int(point["n_cross_pairs"]),
            "gap_ci95_case": _ci_list(boot["ci95"]["gap"]),
        },
        "source_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/m14_clinical_summary.json"),
    )
    parser.add_argument("--task", default="riga_cup")
    args = parser.parse_args()

    summary = summarize_m14(args.results_root, task=args.task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(_strict_json_value(summary), f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
