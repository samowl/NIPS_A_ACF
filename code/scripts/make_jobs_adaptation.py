"""Emit jobs file for the §8 adaptation factorial (M6 .. M11).

Each line::

    ADAPT {task} {fm} {seed} {regime} {lora_rank|-} {lora_blocks|-} {epochs}

Workers should translate ADAPT lines into::

    python scripts/train_adapted.py --task TASK --fm FM --seed S \
        --regime REG [--lora-rank R] [--lora-blocks B] --epochs E \
        --batch-size 16 --data-root $FMPOOL_DATA_ROOT \
        --out results/per_case_dice

Cell counts (paper §8 / appendix tables):

    M6  : 6 LoRA configs   x 4 seeds  x 1 task              = 24
    M7  : 4 full-fine-tuning epochs x 4 seeds  x 1 (task, FM) = 16
    M8  : 1 FM x 8 seeds   x 3 tasks                        = 24
    M9  : 2 FMs x 8 seeds  x 2 tasks                        = 32
    M10 : 6 LoRA configs   x 4 seeds  x (RIGA-Cup, BiomedCLIP) = 24
    M11 : 3 regimes        x 4 seeds  x 2 tasks             = 24
    -------------------------------------------------------------------------
    TOTAL                                                  = 144
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (paper §8 / appendix tables)
# ---------------------------------------------------------------------------

LORA_RANKS: tuple[int, ...] = (8, 16, 64)
# "last-2 / last-4 / last-8 / all-12".
LORA_BLOCKS: tuple[int, ...] = (2, 4, 8, 12)
# 6 unique LoRA configs used in M6/M10 (paper choice spans rank/block diag).
LORA_CONFIGS: tuple[tuple[int, int], ...] = (
    (8, 2), (8, 4), (16, 4), (16, 8), (64, 8), (64, 12),
)

FT_EPOCHS: tuple[int, ...] = (30, 50, 100, 200)

SEEDS_4: tuple[int, ...] = (42, 43, 44, 45)
SEEDS_8: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49)

# Anchors (matches paper §8 narrative).
M6_TASK = "kvasir"
M6_FM = "dinov2_vitb14"

M7_TASK = "kvasir"
M7_FM = "dinov2_vitb14"

M8_FM = "dinov2_vitb14"
M8_TASKS: tuple[str, ...] = ("kvasir", "acdc_lv", "riga_cup")

M9_FMS: tuple[str, ...] = ("biomedclip", "clip_vitl14")
M9_TASKS: tuple[str, ...] = ("kvasir", "riga_cup")

M10_FM = "biomedclip"
M10_TASK = "riga_cup"

M11_TASKS: tuple[str, ...] = ("kvasir", "riga_cup")
M11_FM = "dinov2_vitb14"
M11_LORA: tuple[int, int] = (16, 4)
M11_FT_EPOCHS = 100
M11_RAND_EPOCHS = 100


# ---------------------------------------------------------------------------
# Job tuple
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    task: str
    fm: str
    seed: int
    regime: str  # "lora" | "full_ft" | "rand_init"
    lora_rank: int | None
    lora_blocks: int | None
    epochs: int

    def as_line(self) -> str:
        r = self.lora_rank if self.lora_rank is not None else "-"
        b = self.lora_blocks if self.lora_blocks is not None else "-"
        return (
            f"ADAPT {self.task} {self.fm} {self.seed} {self.regime} "
            f"{r} {b} {self.epochs}"
        )


def _lora_jobs(task: str, fm: str, seeds, configs, epochs: int = 30) -> list[Job]:
    return [
        Job(task=task, fm=fm, seed=s, regime="lora",
            lora_rank=r, lora_blocks=b, epochs=epochs)
        for s in seeds
        for (r, b) in configs
    ]


def _ft_jobs(task: str, fm: str, seeds, epochs_list) -> list[Job]:
    return [
        Job(task=task, fm=fm, seed=s, regime="full_ft",
            lora_rank=None, lora_blocks=None, epochs=e)
        for s in seeds
        for e in epochs_list
    ]


def _frozen_replication_jobs(task: str, fm: str, seeds) -> list[Job]:
    """M8/M9: extra-seed replication of the frozen probe (encoded as
    full_ft @ 30ep so the worker dispatches train_adapted.py uniformly).
    """
    return [
        Job(task=task, fm=fm, seed=s, regime="full_ft",
            lora_rank=None, lora_blocks=None, epochs=30)
        for s in seeds
    ]


def build_all() -> list[Job]:
    jobs: list[Job] = []

    # M6: 6 LoRA configs x 4 seeds on (kvasir, dinov2_vitb14) = 24
    jobs += _lora_jobs(M6_TASK, M6_FM, SEEDS_4, LORA_CONFIGS, epochs=30)

    # M7: full_ft epoch sweep, 4 epochs x 4 seeds = 16
    jobs += _ft_jobs(M7_TASK, M7_FM, SEEDS_4, FT_EPOCHS)

    # M8: 8 seeds x 3 tasks on dinov2_vitb14 = 24
    for task in M8_TASKS:
        jobs += _frozen_replication_jobs(task, M8_FM, SEEDS_8)

    # M9: 2 FMs x 2 tasks x 8 seeds = 32
    for fm in M9_FMS:
        for task in M9_TASKS:
            jobs += _frozen_replication_jobs(task, fm, SEEDS_8)

    # M10: 6 LoRA configs x 4 seeds on (riga_cup, biomedclip) = 24
    jobs += _lora_jobs(M10_TASK, M10_FM, SEEDS_4, LORA_CONFIGS, epochs=30)

    # M11: 3 regimes x 4 seeds x 2 tasks = 24
    for task in M11_TASKS:
        for s in SEEDS_4:
            r, b = M11_LORA
            jobs.append(Job(task=task, fm=M11_FM, seed=s, regime="lora",
                            lora_rank=r, lora_blocks=b, epochs=30))
            jobs.append(Job(task=task, fm=M11_FM, seed=s, regime="full_ft",
                            lora_rank=None, lora_blocks=None, epochs=M11_FT_EPOCHS))
            jobs.append(Job(task=task, fm=M11_FM, seed=s, regime="rand_init",
                            lora_rank=None, lora_blocks=None, epochs=M11_RAND_EPOCHS))
    return jobs


def main() -> None:
    jobs = build_all()
    out = Path("jobs_adaptation.txt")
    out.write_text("\n".join(j.as_line() for j in jobs) + "\n", encoding="utf-8")
    by_label = {"M6": 24, "M7": 16, "M8": 24, "M9": 32, "M10": 24, "M11": 24}
    expected = sum(by_label.values())
    print(f"wrote {len(jobs)} adaptation jobs to {out}")
    print(f"breakdown: {by_label}  total_expected={expected}  emitted={len(jobs)}")
    if len(jobs) != expected:
        raise SystemExit(f"job count mismatch: expected {expected} got {len(jobs)}")


if __name__ == "__main__":
    main()
