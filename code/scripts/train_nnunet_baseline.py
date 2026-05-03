#!/usr/bin/env python
"""Run a seeded nnU-Net v2 baseline on one (task, fold, seed) cell (SPEC §5).

Example
-------
    python scripts/train_nnunet_baseline.py \
        --task acdc_lv --fold 0 --seed 42 \
        --dataset-id 027 --config 3d_fullres \
        --out results/per_case_dice/acdc_lv/nnunet_fold_0/

SPEC §5 contract
~~~~~~~~~~~~~~~~
* Calls the upstream ``nnunetv2.run.run_training.run_training`` Python API
  directly; does NOT shell out to ``nnUNetv2_train`` so seeds propagate via
  the :mod:`fmpool.nnunet_seeded` trainer subclass.
* Trainer class name is ``f"{base_trainer}_Seed{seed}"`` where
  ``base_trainer`` is a length preset from SPEC §5 (default
  ``nnUNetTrainer_100epochs``).
* Importing :mod:`fmpool.nnunet_seeded` registers the seeded variants into
  that module's namespace so nnU-Net's trainer lookup finds them.

Exit codes
~~~~~~~~~~
* ``0`` success.
* ``2`` nnunetv2 not installed.
* ``3`` dataset unavailable / prepare_nnunet_dataset raised.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from fmpool import datasets as fmpool_datasets
from fmpool import determinism
from fmpool import nnunet_seeded  # noqa: F401 — eager import registers variants.

logger = logging.getLogger("fmpool.train_nnunet_baseline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a seeded nnU-Net v2 baseline (SPEC §5)."
    )
    p.add_argument("--task", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="nnU-Net raw-dataset numeric ID (e.g. 27 for Dataset027).",
    )
    p.add_argument(
        "--config",
        default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
    )
    p.add_argument(
        "--base-trainer",
        default="nnUNetTrainer_100epochs",
        help="Base upstream trainer to seed. SPEC §5 default is 100 epochs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-case Dice JSON (SPEC §8).",
    )
    p.add_argument(
        "--nnunet-raw",
        type=str,
        default=None,
        help="Override nnUNet_raw root; otherwise uses env var.",
    )
    p.add_argument("--nnunet-preprocessed", type=str, default=None)
    p.add_argument("--nnunet-results", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# nnU-Net env wiring
# ---------------------------------------------------------------------------


def _set_nnunet_env(args: argparse.Namespace) -> None:
    """Propagate CLI overrides into the nnU-Net environment variables."""
    if args.nnunet_raw:
        os.environ["nnUNet_raw"] = str(args.nnunet_raw)
    if args.nnunet_preprocessed:
        os.environ["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    if args.nnunet_results:
        os.environ["nnUNet_results"] = str(args.nnunet_results)


def _locate_raw_dataset(dataset_id: int) -> Path | None:
    """Find ``nnUNet_raw/Dataset{ID:03d}_*`` on disk (if already staged)."""
    root_env = os.environ.get("nnUNet_raw")
    if not root_env:
        return None
    root = Path(root_env)
    if not root.is_dir():
        return None
    prefix = f"Dataset{dataset_id:03d}_"
    matches = sorted(root.glob(prefix + "*"))
    if not matches:
        return None
    return matches[0]


# ---------------------------------------------------------------------------
# Per-case Dice from predicted vs GT NIfTI
# ---------------------------------------------------------------------------


def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)
    p_sum = float(pred_b.sum())
    t_sum = float(tgt_b.sum())
    if p_sum == 0.0 and t_sum == 0.0:
        return 1.0
    inter = float(np.logical_and(pred_b, tgt_b).sum())
    return (2.0 * inter) / (p_sum + t_sum + 1e-6)


def _load_nifti_volume(path: Path) -> np.ndarray:
    """Read a NIfTI volume via SimpleITK, falling back to nibabel."""
    try:
        import SimpleITK as sitk  # type: ignore

        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img)
    except ImportError:
        import nibabel as nib  # type: ignore

        img = nib.load(str(path))
        return np.asarray(img.get_fdata())


def _collect_per_case_dice(
    pred_dir: Path,
    gt_dir: Path,
    positive_label: int | None = None,
) -> tuple[list[str], list[float]]:
    """Pair predicted vs GT NIfTI by filename and compute per-case Dice.

    ``positive_label`` selects a single foreground class (SPEC §2:
    ``acdc_lv`` uses ``label==3``). When ``None`` the Dice is over all
    non-zero voxels (SPEC §2 BraTS foreground derivative).
    """
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"no prediction files under {pred_dir}")
    ids: list[str] = []
    scores: list[float] = []
    for pred_path in pred_files:
        name = pred_path.name
        gt_path = gt_dir / name
        if not gt_path.is_file():
            logger.warning("missing GT for %s; skipping", name)
            continue
        pred = _load_nifti_volume(pred_path)
        gt = _load_nifti_volume(gt_path)
        if positive_label is not None:
            pred_bin = pred == positive_label
            gt_bin = gt == positive_label
        else:
            pred_bin = pred > 0
            gt_bin = gt > 0
        ids.append(name.replace(".nii.gz", ""))
        scores.append(_binary_dice(pred_bin, gt_bin))
    return ids, scores


# ---------------------------------------------------------------------------
# Held-out validation predictions
# ---------------------------------------------------------------------------


def _holdout_dir(
    dataset_id: int, config: str, fold: int, trainer_class_name: str
) -> Path:
    """Path to nnU-Net's per-fold validation output directory.

    nnU-Net writes validation predictions under
    ``{model_folder}/fold_{k}/validation/`` during ``run_training``.
    """
    from nnunetv2.utilities.file_path_utilities import get_output_folder  # type: ignore

    model_folder = Path(
        get_output_folder(
            dataset_name_or_id=dataset_id,
            trainer_name=trainer_class_name,
            plans_identifier="nnUNetPlans",
            configuration=config,
        )
    )
    val_dir = model_folder / f"fold_{fold}" / "validation"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"expected held-out validation predictions at {val_dir}"
        )
    return val_dir


# ---------------------------------------------------------------------------
# Atomic JSON
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Positive-label lookup per task (SPEC §2)
# ---------------------------------------------------------------------------

_POSITIVE_LABEL: dict[str, int | None] = {
    "kvasir": 1,
    "acdc_lv": 3,
    "brats_wt": None,  # union = any non-zero
    "riga_cup": 1,
    "riga_disc": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    # Step 1: seed everything.
    determinism.set_seed(args.seed)
    _set_nnunet_env(args)

    try:
        from nnunetv2.run.run_training import run_training  # type: ignore
    except ImportError as exc:
        print(
            f"[train_nnunet_baseline] ERROR: nnunetv2 not installed: {exc}",
            file=sys.stderr,
        )
        return 2

    # Step 2: ensure raw dataset layout exists.
    raw_dir = _locate_raw_dataset(args.dataset_id)
    if raw_dir is None:
        try:
            raw_dir = fmpool_datasets.prepare_nnunet_dataset(
                args.task, args.dataset_id
            )
        except (FileNotFoundError, NotImplementedError) as exc:
            logger.error("prepare_nnunet_dataset failed: %s", exc)
            print(
                f"[train_nnunet_baseline] ERROR: cannot materialise raw dataset "
                f"for task={args.task!r} dataset_id={args.dataset_id}: {exc}",
                file=sys.stderr,
            )
            return 3
    logger.info("nnU-Net raw dataset at %s", raw_dir)

    # Step 3: expose seeded trainers through nnU-Net's official discovery path,
    # then launch training via the Python API (do NOT shell out).
    try:
        nnunet_seeded.ensure_nnunet_discovery()
    except RuntimeError as exc:
        print(
            f"[train_nnunet_baseline] ERROR: seeded trainer discovery failed: {exc}",
            file=sys.stderr,
        )
        return 4
    trainer_class_name = f"{args.base_trainer}_Seed{args.seed}"
    if not hasattr(nnunet_seeded, trainer_class_name):
        logger.warning(
            "Seeded trainer %s not found on fmpool.nnunet_seeded; nnU-Net's "
            "lookup may fail. Known variants: %s",
            trainer_class_name,
            [n for n in dir(nnunet_seeded) if n.startswith("nnUNetTrainer_")],
        )

    t0 = time.time()
    run_training(
        dataset_name_or_id=args.dataset_id,
        configuration=args.config,
        fold=args.fold,
        trainer_class_name=trainer_class_name,
    )
    train_elapsed = time.time() - t0

    # Step 4: per-case Dice on held-out fold.
    val_dir = _holdout_dir(
        args.dataset_id, args.config, args.fold, trainer_class_name
    )
    gt_dir = raw_dir / "labelsTr"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"missing GT labels directory: {gt_dir}")
    positive = _POSITIVE_LABEL.get(args.task)
    ids, scores = _collect_per_case_dice(val_dir, gt_dir, positive_label=positive)

    # SHA-256 the final training checkpoint for provenance.
    try:
        from nnunetv2.utilities.file_path_utilities import get_output_folder  # type: ignore

        model_folder = Path(
            get_output_folder(
                dataset_name_or_id=args.dataset_id,
                trainer_name=trainer_class_name,
                plans_identifier="nnUNetPlans",
                configuration=args.config,
            )
        )
        ckpt_path = model_folder / f"fold_{args.fold}" / "checkpoint_final.pth"
        ckpt_sha = (
            determinism.sha256_file(ckpt_path) if ckpt_path.is_file() else ""
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not locate final checkpoint for hashing: %s", exc)
        ckpt_sha = ""

    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    payload = {
        "task": args.task,
        "fm": "nnunet",
        "fold": int(args.fold),
        "seed": int(args.seed),
        "config": args.config,
        "dataset_id": int(args.dataset_id),
        "base_trainer": args.base_trainer,
        "n_test": int(len(ids)),
        "test_ids": list(ids),
        "per_case_dice": [float(x) for x in scores],
        "mean_dice": float(np.mean(scores)) if scores else float("nan"),
        "checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s (n_test=%d mean_dice=%.4f)",
        out_path,
        payload["n_test"],
        payload["mean_dice"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
