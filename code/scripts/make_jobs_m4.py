"""Emit M4 UNet-skip job manifest for run_worker_m4.sh.

Format
------
One job per line:

    M4 {task} {fm} {seed}

Grid (5 FMs x 4 tasks x 2 seeds = 40 cells):
- 5 FMs    : dinov2_vitb14, dinov2_vits14, biomedclip, convnext_tiny, resnet50
- 4 tasks  : kvasir, acdc_lv, riga_cup, brats_wt
- 2 seeds  : 42, 43

DEVIATION FROM SPEC §4: M4 swaps the SPEC §4 1x1-conv decoder for a
multi-scale UNet-skip decoder (see scripts/train_head_unet_skip.py).
"""
from __future__ import annotations

from pathlib import Path

# Best-5 FMs. Each must be a key of fmpool.multiscale.MULTISCALE_SPECS.
FMS: tuple[str, ...] = (
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "convnext_tiny",
    "resnet50",
)

TASKS: tuple[str, ...] = ("kvasir", "acdc_lv", "riga_cup", "brats_wt")

SEEDS: tuple[int, ...] = (42, 43)


def main() -> None:
    lines: list[str] = []
    for task in TASKS:
        for fm in FMS:
            for seed in SEEDS:
                lines.append(f"M4 {task} {fm} {seed}")
    expected = len(FMS) * len(TASKS) * len(SEEDS)
    if len(lines) != expected:
        raise SystemExit(
            f"job count mismatch: expected {expected} got {len(lines)}"
        )
    out = Path("jobs_m4.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} M4 jobs to {out} (5 FMs x 4 tasks x 2 seeds)")


if __name__ == "__main__":
    main()
