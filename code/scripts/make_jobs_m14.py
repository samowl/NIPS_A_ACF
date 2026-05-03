"""Emit jobs_m14.txt for the M14 frozen loss/augmentation matrix.

Format: one ``M14 {task} {fm} {seed}`` per line. The worker
``run_worker_m14.sh`` reads this file and dispatches each line to
``scripts/train_head_clinical.py``.

Matrix: 7 FMs x 1 task (riga_cup) x 4 seeds = 28 cells at encoder-native
224x224. ``biomedclip`` is excluded to match the released seven-FM RIGA M14
subset, not because 336x336 inputs are part of the released protocol.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["riga_cup"]
FMS = [
    "dinov2_vitb14",
    "dinov2_vits14",
    "clip_vitl14",
    "convnext_tiny",
    "efficientnet_b0",
    "resnet50",
    "resnet18",
]
SEEDS = [42, 43, 44, 45]


def main() -> None:
    lines: list[str] = []
    for seed in SEEDS:
        for fm in FMS:
            for task in TASKS:
                lines.append(f"M14 {task} {fm} {seed}")
    out = Path(__file__).resolve().parents[1] / "jobs_m14.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
