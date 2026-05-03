"""Emit two-pass job manifest for run_worker_v2.sh.

Format
------
First the EXTRACT (PASS-A) jobs, one per (task, fm) pair (55 lines):

    EXTRACT {task} {fm}

Then the HEAD (PASS-B) jobs, one per (task, fm, seed) triple (220 lines):

    HEAD {task} {fm} {seed}

The worker launcher must run any HEAD line only after the matching EXTRACT
line has produced ``results/feature_cache/{task}/{fm}/manifest.json``. The
ordering in this file places all EXTRACT lines first; workers are expected
to skip-if-cache-exists so re-running is idempotent.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc"]
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
    out = Path("jobs_v2.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_extract = len(TASKS) * len(FMS)
    n_head = n_extract * len(SEEDS)
    print(f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) to {out}")


if __name__ == "__main__":
    main()
