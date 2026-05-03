#!/usr/bin/env python3
"""Emit PASS-A/PASS-B jobs for the all-primary cross-checkpoint expansion.

This queue adds only new upstream checkpoints/family members to the current
primary task set. Existing 11-FM traces remain untouched and the cached worker
skips any output JSONs that already exist.
"""
from __future__ import annotations

from pathlib import Path


TASKS = ["kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc"]
FMS = [
    "dinov2_vitl14",
    "clip_vitb32",
    "resnet34",
    "resnet101",
    "convnext_small",
    "convnext_base",
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
    out = Path(__file__).resolve().parents[1] / "jobs_xcheckpoint.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_extract = len(TASKS) * len(FMS)
    n_head = n_extract * len(SEEDS)
    print(f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) to {out}")


if __name__ == "__main__":
    main()
