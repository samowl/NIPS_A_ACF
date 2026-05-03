"""Emit M13 job manifest (RIGA leave-MESSIDOR-out OOD).

Format
------
First the EXTRACT (PASS-A) jobs, one per FM (8 lines):

    EXTRACT riga_cup_ood_messidor {fm}

Then the HEAD (PASS-B) jobs, one per (fm, seed) pair (32 lines):

    HEAD riga_cup_ood_messidor {fm} {seed}

The worker (``run_worker_m13.sh``) runs HEAD lines only after the matching
EXTRACT has produced
``results/feature_cache/riga_cup_ood_messidor/{fm}/manifest.json``. Both
verbs skip if their output already exists, so the worker is resume-safe
and idempotent.

Matrix: 8 FM x 1 task (riga_cup_ood_messidor) x 4 seeds = 32 cells.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["riga_cup_ood_messidor"]
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
    out = Path(__file__).resolve().parents[1] / "jobs_m13.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_extract = len(TASKS) * len(FMS)
    n_head = n_extract * len(SEEDS)
    print(
        f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) to {out}"
    )


if __name__ == "__main__":
    main()
