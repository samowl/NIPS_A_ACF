"""Emit the M1 BraTS-multiclass job manifest for run_worker_m1.sh.

M1 matrix: 8 FMs x 4 seeds = 32 HEAD jobs. The PASS-A feature cache is
shared with ``brats_wt`` (the FM features depend only on the image, not
the label schema), so EXTRACT lines target ``brats_wt`` and the HEAD
trainer rebuilds the multiclass masks at training time from the BraTS
NIfTI volumes.

Format
------
First the EXTRACT (PASS-A) jobs (8 lines, only emitted when ``--with-extract``
is set; otherwise we assume the brats_wt cache is already built by the
``jobs_v2.txt`` run):

    EXTRACT brats_wt {fm}

Then the HEAD (PASS-B) jobs (32 lines):

    HEAD brats_multiclass {fm} {seed}

The worker's HEAD verb dispatches to ``train_head_multiclass.py`` via
``run_worker_m1.sh``. The trainer reads from
``results/feature_cache/brats_wt/{fm}/`` (compatible cache, manifest
verified) and writes
``results/per_case_dice_m1/brats_multiclass/{fm}/seed_{seed}.json``.
Both EXTRACT and HEAD are skip-if-output-exists, so re-running is
idempotent and resume-safe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 8 foundation models matching jobs_v2.txt exactly so the brats_wt
# cache (already produced by the v2 worker) is reusable here.
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

# We always reuse the brats_wt cache for PASS-A. The HEAD task name is
# the new logical task, but the feature cache key stays brats_wt.
CACHE_TASK = "brats_wt"
HEAD_TASK = "brats_multiclass"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit M1 BraTS-multiclass job manifest"
    )
    p.add_argument(
        "--with-extract",
        action="store_true",
        help="Also emit EXTRACT brats_wt {fm} lines (default: omitted, "
             "assumes jobs_v2.txt already produced the brats_wt cache).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("jobs_m1.txt"),
        help="Output path (default: ./jobs_m1.txt at repo root)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lines: list[str] = []
    n_extract = 0
    if args.with_extract:
        for fm in FMS:
            lines.append(f"EXTRACT {CACHE_TASK} {fm}")
        n_extract = len(FMS)
    for seed in SEEDS:
        for fm in FMS:
            lines.append(f"HEAD {HEAD_TASK} {fm} {seed}")
    n_head = len(FMS) * len(SEEDS)
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) "
        f"to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
