#!/usr/bin/env python
"""Materialise RIGA Cup + ACDC LV into nnU-Net v2 raw layout (SPEC §5).

Usage
-----
    # In the raw-data environment
    export nnUNet_raw=nnunet_workspace/raw
    export nnUNet_preprocessed=nnunet_workspace/preprocessed
    export nnUNet_results=nnunet_workspace/results
    python scripts/prepare_nnunet_data.py --task riga_cup --dataset-id 501
    python scripts/prepare_nnunet_data.py --task acdc_lv  --dataset-id 502

Layout produced
---------------
    $nnUNet_raw/Dataset501_RIGACup/
        imagesTr/<case_id>_{0000,0001,0002}.png (RIGA Cup train RGB channels)
        labelsTr/<case_id>.png                 (train binary 0/1, 4-of-6 majority)
        imagesTs/<case_id>_{0000,0001,0002}.png (Magrabia held-out RGB channels)
        labelsTs/<case_id>.png                 (held-out labels for scoring only)
        dataset.json
        splits_final.json                      (5-fold over train pool)
        fmpool_test_ids.json                   (Magrabia held-out for SPEC §2)

    $nnUNet_raw/Dataset502_ACDCLV/
        imagesTr/<patient>_frame{ED,ES}_0000.nii.gz  (ACDC: 3D volume)
        labelsTr/<patient>_frame{ED,ES}.nii.gz       (binary {0,1}, LV only)
        dataset.json
        splits_final.json                      (5-fold patient-level)

The case_ids written here align with the loader-managed split JSONs under
``data/splits/`` so per-case Dice keys match between the FM models and the
nnU-Net baselines (SPEC §8).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Ensure local src/ wins over any installed copy.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from fmpool.datasets import resolve_task_root  # noqa: E402

logger = logging.getLogger("fmpool.prepare_nnunet_data")


# ---------------------------------------------------------------------------
# Dataset name conventions
# ---------------------------------------------------------------------------

_TASK_TO_DS_NAME = {
    "riga_cup": "RIGACup",
    "acdc_lv": "ACDCLV",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(_TASK_TO_DS_NAME))
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument(
        "--nnunet-raw",
        default=os.environ.get("nnUNet_raw"),
        help="Override nnUNet_raw root; defaults to env var.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-export even if dataset.json already exists.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _raw_dir(raw_root: Path, dataset_id: int, name: str) -> Path:
    return raw_root / f"Dataset{dataset_id:03d}_{name}"


def _safe_case_id(cid: str) -> str:
    """nnU-Net uses '_' as channel delimiter; '/' is illegal in filenames."""
    return cid.replace("/", "__").replace(" ", "_")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# RIGA Cup (2D)
# ---------------------------------------------------------------------------


def _export_riga_cup(raw_root: Path, dataset_id: int, force: bool) -> Path:
    """RIGA Cup: 2D fundus PNG + 4-of-6 majority cup binary mask."""
    from PIL import Image

    from fmpool.datasets.riga import RIGADataset

    out_dir = _raw_dir(raw_root, dataset_id, _TASK_TO_DS_NAME["riga_cup"])
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    test_images_dir = out_dir / "imagesTs"
    test_labels_dir = out_dir / "labelsTs"
    ds_json = out_dir / "dataset.json"
    if ds_json.is_file() and not force:
        logger.info("RIGA Cup already exported at %s; skip", out_dir)
        return out_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RIGADataset(target="cup", split="train")
    test_ds = RIGADataset(target="cup", split="test")

    written: list[str] = []
    nnunet_train_ids: list[str] = []
    nnunet_test_ids: list[str] = []

    def _dump(
        ds,
        bucket: list[str],
        image_out_dir: Path,
        label_out_dir: Path,
    ) -> None:
        for image_t, mask_t, case_id in ds:
            safe = _safe_case_id(case_id)
            img_np = image_t.permute(1, 2, 0).numpy()  # HxWx3 uint8
            mask_np = mask_t.squeeze(0).numpy().astype(np.uint8)  # 0/1
            for ch in range(3):
                Image.fromarray(img_np[:, :, ch], mode="L").save(
                    image_out_dir / f"{safe}_{ch:04d}.png"
                )
            Image.fromarray(mask_np, mode="L").save(
                label_out_dir / f"{safe}.png"
            )
            bucket.append(safe)
            written.append(safe)

    _dump(train_ds, nnunet_train_ids, images_dir, labels_dir)
    _dump(test_ds, nnunet_test_ids, test_images_dir, test_labels_dir)

    # 5-fold over the *training pool* (BinRushed + MESSIDOR). The held-out
    # Magrabia pool (SPEC §2 test) is exported under imagesTs/labelsTs, so
    # nnU-Net planning/preprocessing sees only training cases.
    rng = np.random.default_rng(42)
    perm = list(nnunet_train_ids)
    rng.shuffle(perm)
    n = len(perm)
    folds: list[dict] = []
    for k in range(5):
        lo = (n * k) // 5
        hi = (n * (k + 1)) // 5
        val = perm[lo:hi]
        train = [c for c in perm if c not in set(val)]
        folds.append({"train": train, "val": val})
    _write_json(out_dir / "splits_final.json", folds)
    _write_json(
        out_dir / "fmpool_test_ids.json",
        {"test_ids": nnunet_test_ids},
    )

    dataset_json = {
        "channel_names": {
            "0": "fundus_R",
            "1": "fundus_G",
            "2": "fundus_B",
        },
        "labels": {"background": 0, "cup": 1},
        "numTraining": len(nnunet_train_ids),
        "file_ending": ".png",
        "name": "RIGACup",
        "description": "RIGA+ optic cup, 4-of-6 rater majority. SPEC §2.",
    }
    _write_json(ds_json, dataset_json)
    logger.info(
        "RIGA Cup: wrote %d cases (train=%d test=%d) under %s",
        len(written),
        len(nnunet_train_ids),
        len(nnunet_test_ids),
        out_dir,
    )
    return out_dir


# ---------------------------------------------------------------------------
# ACDC LV (3D)
# ---------------------------------------------------------------------------


def _export_acdc_lv(raw_root: Path, dataset_id: int, force: bool) -> Path:
    """ACDC LV: 3D volumes, label binarised to {0, 1=LV}."""
    import nibabel as nib

    from fmpool.datasets.acdc import _FRAME_RE, _LV_LABEL  # type: ignore

    out_dir = _raw_dir(raw_root, dataset_id, _TASK_TO_DS_NAME["acdc_lv"])
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    ds_json = out_dir / "dataset.json"
    if ds_json.is_file() and not force:
        logger.info("ACDC LV already exported at %s; skip", out_dir)
        return out_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_task_root("acdc")
    images_src = root / "Images"
    masks_src = root / "Masks"

    frame_paths: list[tuple[str, str, Path, Path]] = []
    for entry in sorted(images_src.iterdir()):
        m = _FRAME_RE.match(entry.name)
        if m is None:
            continue
        patient = m.group(1)
        if int(patient[-3:]) > 100:
            continue
        frame = m.group(2)
        mask = masks_src / entry.name.replace(".nii.gz", "_gt.nii.gz")
        if not mask.is_file():
            mask = masks_src / entry.name
        if not mask.is_file():
            raise FileNotFoundError(f"missing GT for {entry.name}")
        frame_paths.append((patient, frame, entry, mask))

    written: list[str] = []
    patient_to_cases: dict[str, list[str]] = {}
    for patient, frame, img_path, mask_path in frame_paths:
        safe = _safe_case_id(f"{patient}_frame{frame}")
        img = nib.load(str(img_path))
        msk = nib.load(str(mask_path))
        img_data = np.asarray(img.dataobj)
        msk_data = np.asarray(msk.dataobj)
        msk_bin = (msk_data == _LV_LABEL).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(img_data.astype(np.float32), img.affine, img.header),
            images_dir / f"{safe}_0000.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(msk_bin, msk.affine, msk.header),
            labels_dir / f"{safe}.nii.gz",
        )
        written.append(safe)
        patient_to_cases.setdefault(patient, []).append(safe)

    # Patient-level 5-fold CV (matches fmpool.datasets.acdc seed=42 logic).
    patients = sorted(patient_to_cases)
    rng = np.random.default_rng(42)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    folds: list[dict] = []
    for k in range(5):
        lo = k * 20
        hi = lo + 20
        val_pat = set(shuffled[lo:hi])
        val_cases: list[str] = []
        train_cases: list[str] = []
        for p, cases in patient_to_cases.items():
            (val_cases if p in val_pat else train_cases).extend(cases)
        folds.append({"train": sorted(train_cases), "val": sorted(val_cases)})
    _write_json(out_dir / "splits_final.json", folds)

    dataset_json = {
        "channel_names": {"0": "cine"},
        "labels": {"background": 0, "LV": 1},
        "numTraining": len(written),
        "file_ending": ".nii.gz",
        "name": "ACDCLV",
        "description": "ACDC LV binary (paper binary task). SPEC §2.",
    }
    _write_json(ds_json, dataset_json)
    logger.info(
        "ACDC LV: wrote %d volumes from %d patients under %s",
        len(written),
        len(patients),
        out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if not args.nnunet_raw:
        print(
            "[prepare_nnunet_data] ERROR: --nnunet-raw or env nnUNet_raw required",
            file=sys.stderr,
        )
        return 2
    raw_root = Path(args.nnunet_raw)
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.task == "riga_cup":
        _export_riga_cup(raw_root, args.dataset_id, args.force)
    elif args.task == "acdc_lv":
        _export_acdc_lv(raw_root, args.dataset_id, args.force)
    else:
        raise ValueError(args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
