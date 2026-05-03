"""Emit jobs_m15.txt for the M15 fold-reslicing matrix.

Format: one ``M15 {task} {fm} {fold} {seed}`` per line. The worker
``run_worker_m15.sh`` reads this file and dispatches each line to
``scripts/train_head_5fold.py``.

Matrix: 8 FMs x 2 tasks (riga_cup, acdc_lv) x 5 folds x 1 seed = 80 cells.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["riga_cup", "acdc_lv"]
FMS = [
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "clip_vitl14",
    "convnext_tiny",
    "efficientnet_b0",
    "resnet50",
    "resnet18",
]
FOLDS = list(range(5))
SEED = 42


def main() -> None:
    lines: list[str] = []
    # Round-robin ordering: outer fold so all FMs share GPU pressure per fold.
    for fold in FOLDS:
        for fm in FMS:
            for task in TASKS:
                lines.append(f"M15 {task} {fm} {fold} {SEED}")
    out = Path(__file__).resolve().parents[1] / "jobs_m15.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
