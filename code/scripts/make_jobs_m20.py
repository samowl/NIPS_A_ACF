"""Emit M20 job manifest (BraTS multimodal T1c/T2w/FLAIR as RGB).

Format
------
First the EXTRACT (PASS-A) jobs, one per FM (5 lines):

    EXTRACT brats_wt_multimodal {fm}

Then the HEAD (PASS-B) jobs, one per (fm, seed) pair (20 lines):

    HEAD brats_wt_multimodal {fm} {seed}

The worker (``run_worker_m20.sh``) runs HEAD lines only after the matching
EXTRACT has produced
``results/feature_cache/brats_wt_multimodal/{fm}/manifest.json``. Both
verbs skip if their output already exists.

FM subset (best 5 from the v2 matrix per the M20 spec): three ViT FMs
(dinov2_vitb14, dinov2_vits14, biomedclip), one CLIP-L, and one CNN
baseline (resnet50). The CNN families that performed worst on BraTS in
the v2 matrix (efficientnet_b0, resnet18, convnext_tiny) are dropped to
fit the 20-cell budget.

Matrix: 5 FM x 1 task (brats_wt_multimodal) x 4 seeds = 20 cells.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["brats_wt_multimodal"]
FMS = [
    "dinov2_vitb14",
    "biomedclip",
    "dinov2_vits14",
    "clip_vitl14",
    "resnet50",
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
    out = Path(__file__).resolve().parents[1] / "jobs_m20.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_extract = len(TASKS) * len(FMS)
    n_head = n_extract * len(SEEDS)
    print(
        f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) to {out}"
    )


if __name__ == "__main__":
    main()
