"""Shared helpers for Medical Segmentation Decathlon (MSD) loaders.

Used by ``msd_hippocampus``, ``msd_heart``, ``msd_prostate``, and
``msd_spleen``. Each MSD task ships a directory layout::

    <task_root>/
        imagesTr/<case>.nii.gz   (3D or 4D)
        labelsTr/<case>.nii.gz   (3D, integer labels)

We extract the *central axial slice* (``vol.shape[2] // 2``) of every
volume, normalise to ``uint8`` in ``[0, 255]``, replicate to ``[3,H,W]``,
and binarise the label as ``label > 0``. Cases are split 80/20 at the
volume (= patient) level via numpy seed 42.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fmpool.datasets import (
    load_committed_split,
    require_dir,
    resolve_task_root,
    save_committed_split,
)

_TARGET_HW = 224
_TEST_FRACTION = 0.2
_SPLIT_SEED = 42


def _list_niftis(directory: Path) -> list[str]:
    """Return sorted .nii.gz filenames, skipping ``._*`` macOS metadata."""
    out: list[str] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith("._"):
            continue
        if not name.endswith(".nii.gz"):
            continue
        out.append(name)
    return out


def _normalise_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2D float slice to uint8 [0,255]."""
    img = np.asarray(slice_2d, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def _resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F

    x = img.to(torch.float32).unsqueeze(0)
    x = F.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    return x.clamp(0, 255).squeeze(0).to(torch.uint8)


def _resize_bool(mask: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F

    x = mask.to(torch.float32).unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="nearest")
    return x.squeeze(0).to(torch.bool)


class MSDCenterSliceDataset(Dataset):
    """Base class for central-axial-slice MSD binary segmentation tasks.

    Subclasses configure ``task_key`` (used both for ``resolve_task_root``
    and for the committed split JSON) and optional ``modality_index``
    for 4D multi-modal volumes (e.g. Prostate T2 = 0). Output:
    ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)`` with
    ``case_id`` equal to the volume stem (e.g. ``hippocampus_001``).
    """

    task_key: str = ""  # set by subclass; passed to resolve_task_root
    split_task_name: str = ""  # name used in data/splits/<name>_split.json
    modality_index: Optional[int] = None  # None = 3D; int = take channel
    provenance_extra: str = ""

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"{self.split_task_name} split must be train/val/test, "
                f"got {split!r}"
            )
        if not self.task_key or not self.split_task_name:
            raise RuntimeError(
                "MSDCenterSliceDataset subclass must set task_key "
                "and split_task_name"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root(self.task_key)
        require_dir(self.root, dataset=self.split_task_name)
        self.images_dir = self.root / "imagesTr"
        self.labels_dir = self.root / "labelsTr"
        require_dir(self.images_dir, dataset=self.split_task_name)
        require_dir(self.labels_dir, dataset=self.split_task_name)

        manifest = self._load_or_generate_split()
        self.case_ids: list[str] = list(manifest[split])

    # ----- split handling -----------------------------------------------
    def _discover_cases(self) -> list[str]:
        image_files = _list_niftis(self.images_dir)
        label_files = set(_list_niftis(self.labels_dir))
        cases: list[str] = []
        for fname in image_files:
            if fname not in label_files:
                continue
            stem = fname[: -len(".nii.gz")]
            cases.append(stem)
        return cases

    def _load_or_generate_split(self) -> dict:
        committed = load_committed_split(self.split_task_name)
        if committed is not None:
            return {
                "train": list(committed.get("train", [])),
                "val": list(committed.get("val", [])),
                "test": list(committed.get("test", [])),
            }

        cases = self._discover_cases()
        if not cases:
            raise RuntimeError(
                f"[{self.split_task_name}] no paired (image, label) "
                f"NIfTIs under {self.root}"
            )
        rng = np.random.default_rng(_SPLIT_SEED)
        shuffled = list(cases)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * _TEST_FRACTION))
        if n_test == 0 and len(shuffled) > 1:
            n_test = 1
        test_cases = shuffled[:n_test]
        train_cases = shuffled[n_test:]
        # Sort for deterministic on-disk ordering of split JSON.
        train_cases = sorted(train_cases)
        test_cases = sorted(test_cases)

        provenance = (
            f"MSD task {self.task_key!r} central-axial-slice 80/20 "
            f"patient-level split via numpy.random.default_rng(42) over "
            f"the volume ID list. {self.provenance_extra}"
        ).strip()
        persisted = {
            "train": train_cases,
            "val": [],
            "test": test_cases,
        }
        save_committed_split(
            self.split_task_name, persisted, provenance=provenance
        )
        return {"train": train_cases, "val": [], "test": test_cases}

    # ----- I/O ----------------------------------------------------------
    def _load_volume(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        import nibabel as nib  # local: heavy optional dep

        img_path = self.images_dir / f"{case_id}.nii.gz"
        seg_path = self.labels_dir / f"{case_id}.nii.gz"
        vol = np.asarray(nib.load(str(img_path)).get_fdata())
        seg = np.asarray(nib.load(str(seg_path)).get_fdata())
        if self.modality_index is not None:
            if vol.ndim != 4:
                raise RuntimeError(
                    f"[{self.split_task_name}] expected 4D volume for "
                    f"modality slicing, got shape {vol.shape} for "
                    f"{case_id}"
                )
            vol = vol[..., self.modality_index]
        if vol.ndim != 3:
            raise RuntimeError(
                f"[{self.split_task_name}] expected 3D volume after "
                f"modality reduction, got shape {vol.shape} for {case_id}"
            )
        if seg.ndim != 3:
            raise RuntimeError(
                f"[{self.split_task_name}] expected 3D label volume, "
                f"got shape {seg.shape} for {case_id}"
            )
        return vol, seg

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        vol, seg = self._load_volume(case_id)
        # Use the third spatial axis (axial) as in BraTS / ACDC.
        z = vol.shape[2] // 2
        img_slice = vol[:, :, z]
        seg_slice = seg[:, :, z]

        img_u8 = _normalise_to_uint8(img_slice)
        img_t = torch.from_numpy(img_u8).unsqueeze(0).repeat(3, 1, 1)
        mask_bool = np.asarray(seg_slice) > 0
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        img_t = _resize_uint8(img_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t, case_id
