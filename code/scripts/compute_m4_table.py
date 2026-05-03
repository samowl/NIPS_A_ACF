"""Regenerate the released M4 UNet-skip appendix table from per-case traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "m4_unet_summary_v1"
TASK_ORDER = ("kvasir", "acdc_lv", "riga_cup", "brats_wt")
FM_LABELS = {
    "dinov2_vitb14": "DINOv2-B",
    "dinov2_vits14": "DINOv2-S",
    "biomedclip": "BiomedCLIP",
    "convnext_tiny": "ConvNeXt-T",
    "resnet50": "ResNet-50",
}
SEEDS = (42, 43)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"array length mismatch: {a.shape} vs {b.shape}")
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_summary(results_root: Path) -> dict:
    source_root = results_root / "per_case_dice_m4"
    rows: list[dict] = []

    for task in TASK_ORDER:
        for fm, label in FM_LABELS.items():
            paths = [source_root / task / fm / f"seed_{seed}.json" for seed in SEEDS]
            if not all(path.exists() for path in paths):
                continue
            payloads = [_load(path) for path in paths]
            test_ids = payloads[0]["test_ids"]
            for payload, path, seed in zip(payloads, paths, SEEDS, strict=True):
                if payload.get("task") != task or payload.get("fm") != fm:
                    raise ValueError(f"{path}: task/fm mismatch")
                if int(payload.get("seed")) != seed:
                    raise ValueError(f"{path}: seed mismatch")
                if payload["test_ids"] != test_ids:
                    raise ValueError(f"{path}: test_ids mismatch within cell")

            traces = [
                np.asarray(payload["per_case_dice"], dtype=float) for payload in payloads
            ]
            row = {
                "task": task,
                "backbone": label,
                "fm": fm,
                "seeds": list(SEEDS),
                "n_test": int(len(test_ids)),
                "mean_dice": float(np.mean([float(p["mean_dice"]) for p in payloads])),
                "seed_pair_r": _pearson(traces[0], traces[1]),
                "source_files": [
                    str(path.relative_to(results_root)) for path in paths
                ],
            }
            rows.append(row)

    if not rows:
        raise RuntimeError(f"no M4 cells found under {source_root}")

    seed_pair_rs = [row["seed_pair_r"] for row in rows if not math.isnan(row["seed_pair_r"])]
    mean_dice = [row["mean_dice"] for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": "per_case_dice_m4",
        "seeds": list(SEEDS),
        "rows": rows,
        "n_cells": int(len(rows)),
        "seed_pair_r_range": [float(min(seed_pair_rs)), float(max(seed_pair_rs))],
        "mean_dice_range": [float(min(mean_dice)), float(max(mean_dice))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/_merged"),
        help="Root containing per_case_dice_m4/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/m4_unet_summary.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    summary = compute_summary(args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
