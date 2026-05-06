"""Emit M4 UNet-skip expansion job manifest.

Expansion grid: 9 FMs x 5 tasks x 2 seeds = 90 cells, with the
50 already-released cells (5 FMs x 4 tasks x 2 seeds + the originals)
skipped at runtime by ``run_worker_m4.sh`` (it checks ``out_json``).

The expansion adds:
  - 4 new backbones: resnet18, efficientnet_b0, mae_vitb16, deit_vitb16
  - 1 new task row (riga_disc) for all 9 backbones (the original 5 plus
    the 4 expansion backbones)
  - existing 5 backbones x 4 tasks already released

Net new cells: 4 new FMs x 5 tasks x 2 seeds + 5 existing FMs x
1 new task x 2 seeds = 40 + 10 = 50 cells.

DEVIATION FROM SPEC §4: M4 swaps the SPEC §4 1x1-conv decoder for a
multi-scale UNet-skip decoder; this module adds 4 new MULTISCALE_SPECS
wrappers (see fmpool.multiscale) plus the riga_disc row for the
existing 5 backbones, which were absent from jobs_m4.txt.
"""
from __future__ import annotations

from pathlib import Path

EXISTING_FMS: tuple[str, ...] = (
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "convnext_tiny",
    "resnet50",
)

NEW_FMS: tuple[str, ...] = (
    "resnet18",
    "efficientnet_b0",
    "mae_vitb16",
    "deit_vitb16",
)

ALL_FMS: tuple[str, ...] = EXISTING_FMS + NEW_FMS  # 9 backbones

EXISTING_TASKS: tuple[str, ...] = ("kvasir", "acdc_lv", "riga_cup", "brats_wt")
NEW_TASKS: tuple[str, ...] = ("riga_disc",)
ALL_TASKS: tuple[str, ...] = EXISTING_TASKS + NEW_TASKS  # 5 tasks

SEEDS: tuple[int, ...] = (42, 43)


def main() -> None:
    lines: list[str] = []
    # New backbones x all tasks
    for fm in NEW_FMS:
        for task in ALL_TASKS:
            for seed in SEEDS:
                lines.append(f"M4 {task} {fm} {seed}")
    # Existing backbones x new task only (riga_disc)
    for fm in EXISTING_FMS:
        for task in NEW_TASKS:
            for seed in SEEDS:
                lines.append(f"M4 {task} {fm} {seed}")

    expected = len(NEW_FMS) * len(ALL_TASKS) * len(SEEDS) + len(
        EXISTING_FMS
    ) * len(NEW_TASKS) * len(SEEDS)
    if len(lines) != expected:
        raise SystemExit(
            f"job count mismatch: expected {expected} got {len(lines)}"
        )

    out = Path("jobs_m4_expand.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {len(lines)} M4 expansion jobs to {out} "
        f"({len(NEW_FMS)} new FMs x {len(ALL_TASKS)} tasks x {len(SEEDS)} seeds + "
        f"{len(EXISTING_FMS)} existing FMs x {len(NEW_TASKS)} new task x {len(SEEDS)} seeds)"
    )


if __name__ == "__main__":
    main()
