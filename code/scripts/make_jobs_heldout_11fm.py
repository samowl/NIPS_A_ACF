#!/usr/bin/env python3
"""Emit PASS-A/PASS-B jobs for the held-out 11-FM rerun.

The output is intended for an isolated per-case root, e.g.
``FMPOOL_PER_CASE_ROOT=$REPO/results/per_case_dice_heldout_11fm`` when running
``run_worker_v2.sh``.
"""
from __future__ import annotations

from pathlib import Path


TASKS = [
    "msd_hippocampus",
    "isic2018",
    "msd_heart",
    "msd_prostate",
    "msd_spleen",
]
FMS = [
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
SEEDS = [42, 43, 44, 45]


def main() -> None:
    lines: list[str] = []
    for task in TASKS:
        for fm in FMS:
            lines.append(f"EXTRACT {task} {fm}")
    for seed in SEEDS:
        for task in TASKS:
            for fm in FMS:
                lines.append(f"HEAD {task} {fm} {seed}")
    out = Path(__file__).resolve().parents[1] / "jobs_heldout_11fm.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_extract = len(TASKS) * len(FMS)
    n_head = n_extract * len(SEEDS)
    print(f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) to {out}")


if __name__ == "__main__":
    main()
