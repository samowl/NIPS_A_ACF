#!/usr/bin/env python
"""Predict on the held-out test pool and emit per-case Dice JSON (SPEC §8).

Two modes:

  * RIGA Cup: per SPEC §2 the held-out test pool is *Magrabia*
    (`fmpool_test_ids.json`). This script runs `nnUNetv2_predict` on those
    PNGs with the seeded trainer of the requested fold and computes
    per-case Dice against `labelsTs` (falling back to `labelsTr` for legacy
    exports).
  * ACDC LV: per SPEC §2 the held-out fold's *val* set is the test pool.
    We reuse the per-fold validation predictions written by
    `nnUNetv2_train` (under `<results>/.../fold_<F>/validation/`).

Output schema (matches the FM JSONs at
`results/per_case_dice/{task}/{fm}/seed_{seed}.json`):
{
  "task": str,
  "fm":   "nnunet_2d_100ep" | "nnunet_2d_1000ep" | "nnunet_3d_fullres",
  "fold": int,
  "seed": int,
  "n_test": int,
  "test_ids": [str, ...],
  "per_case_dice": [float, ...],
  "mean_dice": float,
  "training_elapsed_s": float | null
}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("fmpool.predict_nnunet_holdout")


_FM_LABEL = {
    ("riga_cup", "2d", "nnUNetTrainer_100epochs"): "nnunet_2d_100ep",
    ("riga_cup", "2d", "nnUNetTrainer_1000epochs"): "nnunet_2d_1000ep",
    ("acdc_lv", "3d_fullres", "nnUNetTrainer_1000epochs"): "nnunet_3d_fullres",
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["riga_cup", "acdc_lv"])
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument("--config", required=True, choices=["2d", "3d_fullres"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--base-trainer",
        required=True,
        choices=["nnUNetTrainer_100epochs", "nnUNetTrainer_1000epochs"],
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--training-elapsed-s", type=float, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(bool)
    t = target.astype(bool)
    ps, ts = float(p.sum()), float(t.sum())
    if ps == 0.0 and ts == 0.0:
        return 1.0
    inter = float(np.logical_and(p, t).sum())
    return (2.0 * inter) / (ps + ts + 1e-6)


def _load_2d_label(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _release_case_id(case_id: str) -> str:
    """Stable non-identifying ID for released RIGA alignment traces."""
    return "riga_" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]


def _load_3d(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).dataobj)


def _model_folder(dataset_id, trainer_name, config) -> Path:
    from nnunetv2.utilities.file_path_utilities import get_output_folder  # type: ignore

    return Path(
        get_output_folder(
            dataset_name_or_id=dataset_id,
            trainer_name=trainer_name,
            plans_identifier="nnUNetPlans",
            configuration=config,
        )
    )


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _harvest_acdc(args, trainer_class_name) -> tuple[list[str], list[float]]:
    model = _model_folder(args.dataset_id, trainer_class_name, args.config)
    val_dir = model / f"fold_{args.fold}" / "validation"
    raw = Path(os.environ["nnUNet_raw"]) / f"Dataset{args.dataset_id:03d}_ACDCLV"
    gt = raw / "labelsTr"
    ids: list[str] = []
    scores: list[float] = []
    for pred_path in sorted(val_dir.glob("*.nii.gz")):
        gt_path = gt / pred_path.name
        if not gt_path.is_file():
            logger.warning("missing GT for %s", pred_path.name)
            continue
        pred = _load_3d(pred_path) > 0
        truth = _load_3d(gt_path) > 0
        ids.append(pred_path.name.replace(".nii.gz", ""))
        scores.append(_binary_dice(pred, truth))
    return ids, scores


def _harvest_riga(args, trainer_class_name) -> tuple[list[str], list[float]]:
    raw = Path(os.environ["nnUNet_raw"]) / f"Dataset{args.dataset_id:03d}_RIGACup"
    test_ids = json.loads((raw / "fmpool_test_ids.json").read_text())["test_ids"]
    images_ts = raw / "imagesTs"
    labels_ts = raw / "labelsTs"
    images_tr = raw / "imagesTr"
    labels_tr = raw / "labelsTr"

    work = Path(tempfile.mkdtemp(prefix="nnunet_predict_"))
    in_dir = work / "in"
    out_dir = work / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    try:
        for cid in test_ids:
            for ch in range(3):
                name = f"{cid}_{ch:04d}.png"
                src = images_ts / name
                if not src.is_file():
                    src = images_tr / name
                shutil.copy2(src, in_dir / src.name)
        cmd = [
            "nnUNetv2_predict",
            "-d", str(args.dataset_id),
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-c", args.config,
            "-f", str(args.fold),
            "-tr", trainer_class_name,
            "-p", "nnUNetPlans",
        ]
        logger.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        ids: list[str] = []
        scores: list[float] = []
        for cid in test_ids:
            pred_path = out_dir / f"{cid}.png"
            gt_path = labels_ts / f"{cid}.png"
            if not gt_path.is_file():
                gt_path = labels_tr / f"{cid}.png"
            if not pred_path.is_file() or not gt_path.is_file():
                logger.warning("missing pred/GT for %s", cid)
                continue
            pred = _load_2d_label(pred_path) > 0
            truth = _load_2d_label(gt_path) > 0
            ids.append(_release_case_id(cid))
            scores.append(_binary_dice(pred, truth))
        return ids, scores
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    fm_label = _FM_LABEL.get((args.task, args.config, args.base_trainer))
    if fm_label is None:
        print(
            f"[predict_nnunet_holdout] ERROR: no FM label for "
            f"task={args.task} config={args.config} trainer={args.base_trainer}",
            file=sys.stderr,
        )
        return 2

    epoch_suffix = args.base_trainer.removeprefix("nnUNetTrainer_")
    trainer_class_name = f"nnUNetTrainerSeed{args.seed}_{epoch_suffix}"
    try:
        from fmpool.nnunet_seeded import ensure_nnunet_discovery

        ensure_nnunet_discovery()
    except RuntimeError as exc:
        print(
            f"[predict_nnunet_holdout] ERROR: seeded trainer discovery failed: {exc}",
            file=sys.stderr,
        )
        return 4
    if args.task == "acdc_lv":
        ids, scores = _harvest_acdc(args, trainer_class_name)
    else:
        ids, scores = _harvest_riga(args, trainer_class_name)

    payload = {
        "task": args.task,
        "fm": fm_label,
        "dataset_id": int(args.dataset_id),
        "config": args.config,
        "base_trainer": args.base_trainer,
        "trainer_class_name": trainer_class_name,
        "plans_identifier": "nnUNetPlans",
        "checkpoint": "checkpoint_final.pth",
        # Workspace paths and host are intentionally omitted from the released
        # trace payload to avoid leaking site-specific identifiers. The audit
        # record (workspace SHA-256 of splits_final.json, dataset/plan IDs) is
        # documented in the parity audit report rather than in per-trace
        # payloads. Trainer class, base trainer, plans identifier, and final
        # checkpoint name above are sufficient to reproduce the run.
        "fold": int(args.fold),
        "seed": int(args.seed),
        "n_test": int(len(ids)),
        "test_ids": list(ids),
        "per_case_dice": [float(x) for x in scores],
        "mean_dice": float(np.mean(scores)) if scores else float("nan"),
        "training_elapsed_s": (
            float(args.training_elapsed_s) if args.training_elapsed_s else None
        ),
    }
    _atomic_write(args.out, payload)
    logger.info(
        "wrote %s (n=%d mean=%.4f)",
        args.out,
        payload["n_test"],
        payload["mean_dice"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
