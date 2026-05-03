"""Emit M9 (cross-architecture random-init) job manifest for run_worker_m9.sh.

This is the corrected M9 cell-set used for the decoder-sweep matrix
(line 180): "Random-init ConvNeXt-T + ResNet-50 on Cup/ACDC, 2 FM x 8 seeds
x 2 tasks = 32 cells". The legacy ``make_jobs_adaptation.py`` mislabels its
M9 entry (BiomedCLIP + CLIP-L on full_ft @ 30ep, which is a benchmark
replication row not a random-init row); this file emits the actual M9.

Format (matches scripts/train_adapted.py via run_worker_adapt.sh):

    ADAPT {task} {fm} {seed} {regime} {lora_rank|-} {lora_blocks|-} {epochs}

Grid (2 FMs x 2 tasks x 8 seeds = 32 cells):
- 2 FMs    : convnext_tiny, resnet50           (CNN architectures)
- 2 tasks  : acdc_lv, riga_cup
- 8 seeds  : 42..49
- regime   : rand_init                          (ALL 32 lines)
- epochs   : 30

The ``rand_init`` regime is already supported by train_adapted.py for both
ConvNeXt-T and ResNet50 via :func:`scripts.train_adapted.build_random_encoder`
(branches at lines 244-255). Unlike LoRA -- which raises NotImplementedError
on CNN trunks -- rand_init only resets weights, so the standard nn.Module
trunk works without modification.
"""
from __future__ import annotations

from pathlib import Path

# CNN backbones (NOT ViT; the LoRA path is irrelevant here).
FMS: tuple[str, ...] = ("convnext_tiny", "resnet50")

TASKS: tuple[str, ...] = ("acdc_lv", "riga_cup")

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49)

REGIME: str = "rand_init"
EPOCHS: int = 30


def main() -> None:
    lines: list[str] = []
    # Group by task -> fm -> seed for readability (matches user spec layout).
    for task in TASKS:
        for fm in FMS:
            for seed in SEEDS:
                lines.append(
                    f"ADAPT {task} {fm} {seed} {REGIME} - - {EPOCHS}"
                )
    expected = len(FMS) * len(TASKS) * len(SEEDS)
    if len(lines) != expected:
        raise SystemExit(
            f"job count mismatch: expected {expected} got {len(lines)}"
        )
    out = Path("jobs_m9.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {len(lines)} M9 jobs to {out} "
        f"(2 CNNs x 2 tasks x 8 seeds, regime={REGIME}, epochs={EPOCHS})"
    )


if __name__ == "__main__":
    main()
