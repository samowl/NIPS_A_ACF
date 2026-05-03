"""Emit M19 job manifest (BraTS foreground training-set-fraction sweep).

Format
------
One line per (task, fm, seed, train_frac) cell:

    HEAD_FRAC {task} {fm} {seed} {train_frac}

PASS-A feature cache (``results/feature_cache/brats_wt/{fm}/``) is reused
from the v2 pipeline; this matrix only varies PASS-B subset sampling.
``run_worker_m19.sh`` skips cells whose JSON output already exists.

Matrix: 5 FM x 1 task (brats_wt) x 1 seed x 4 fractions = 20 cells, matching
the completed feature-cache subset released with the paper.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["brats_wt"]
FMS = [
    "dinov2_vitb14",
    "biomedclip",
    "clip_vitl14",
    "efficientnet_b0",
    "resnet50",
]
SEEDS = [42]
FRACS = [0.10, 0.25, 0.50, 1.00]


def main() -> None:
    lines: list[str] = []
    for task in TASKS:
        for frac in FRACS:
            for fm in FMS:
                for seed in SEEDS:
                    lines.append(f"HEAD_FRAC {task} {fm} {seed} {frac}")
    out = Path(__file__).resolve().parents[1] / "jobs_m19.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
