"""ACDC cardiac MRI LV loader (SPEC §2, row ``acdc`` / task ``acdc_lv``).

Per SPEC §2: public training set is 100 patients (001-100). Each
patient has two cine frames (ED, ES). We extract *all axial slices
with non-empty LV label* per frame, treat each slice as a case, and
group 5-fold CV at the patient level (20 patients/fold) with
``numpy`` seed 42.

Data in the configured ACDC root may additionally contain patients 101-150 with GT, even
though SPEC says GT is withheld for those. We exclude 101-150 and
record the decision in ``docs/DATA_NOTES.md``.
"""
from __future__ import annotations

import re
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

_NUM_FOLDS = 5
_PATIENTS_PER_FOLD = 20
_PUBLIC_PATIENT_MAX = 100
_LV_LABEL = 3
_TARGET_HW = 224
_FRAME_RE = re.compile(r"^(patient\d{3})_frame(\d{2})\.nii\.gz$")


class ACDCLVDataset(Dataset):
    """ACDC LV binary segmentation (SPEC §2, task ``acdc_lv``).

    Returns ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)``.
    ``case_id`` format: ``"{patient}_frame{frame}_slice{slice_idx:03d}"``.
    Split values accepted: ``fold_{k}_train`` / ``fold_{k}_val`` for
    ``k ∈ {0..4}``, plus plain ``train`` / ``test`` (= fold 0).
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("acdc")
        require_dir(self.root, dataset="acdc")
        self.images_dir = self.root / "Images"
        self.masks_dir = self.root / "Masks"
        require_dir(self.images_dir, dataset="acdc")
        require_dir(self.masks_dir, dataset="acdc")

        public_patients, all_patients = self._discover_patients()
        if len(public_patients) != _PUBLIC_PATIENT_MAX:
            raise RuntimeError(
                f"[acdc] expected {_PUBLIC_PATIENT_MAX} public patients "
                f"(001-100), found {len(public_patients)}. All patients "
                f"on disk: {len(all_patients)}"
            )

        manifest = self._load_or_generate_split(public_patients)
        self.case_ids: list[str] = self._resolve_split(split, manifest)

    def _discover_patients(self) -> tuple[list[str], list[str]]:
        """Return (public patients [001-100], all patients on disk)."""
        all_patients: set[str] = set()
        for entry in self.images_dir.iterdir():
            m = _FRAME_RE.match(entry.name)
            if m is None:
                continue
            all_patients.add(m.group(1))
        all_sorted = sorted(all_patients)
        public = [p for p in all_sorted if int(p[-3:]) <= _PUBLIC_PATIENT_MAX]
        return public, all_sorted

    def _frame_paths(
        self, patient: str
    ) -> list[tuple[str, Path, Path]]:
        frames: list[tuple[str, Path, Path]] = []
        for entry in sorted(self.images_dir.iterdir()):
            m = _FRAME_RE.match(entry.name)
            if m is None or m.group(1) != patient:
                continue
            frame_label = m.group(2)
            # ACDC masks use _gt suffix: patient001_frame01_gt.nii.gz
            mask_name = entry.name.replace(".nii.gz", "_gt.nii.gz")
            mask = self.masks_dir / mask_name
            if not mask.is_file():
                # Fallback: some layouts use the plain filename.
                fallback = self.masks_dir / entry.name
                if fallback.is_file():
                    mask = fallback
                else:
                    raise FileNotFoundError(
                        f"[acdc] mask missing for {entry.name}: expected {mask} or {fallback}"
                    )
            frames.append((frame_label, entry, mask))
        if not frames:
            raise RuntimeError(f"[acdc] no frames found for {patient}")
        return frames

    def _enumerate_cases(
        self, patients: list[str]
    ) -> tuple[list[str], dict[str, tuple[str, str, int]]]:
        import nibabel as nib  # local: heavy optional dep

        case_ids: list[str] = []
        index: dict[str, tuple[str, str, int]] = {}
        for patient in patients:
            for frame, _img_path, mask_path in self._frame_paths(patient):
                mask_arr = nib.load(str(mask_path)).get_fdata()
                for slice_idx in range(mask_arr.shape[2]):
                    slc = mask_arr[:, :, slice_idx]
                    if not np.any(slc == _LV_LABEL):
                        continue
                    case_id = (
                        f"{patient}_frame{frame}_slice{slice_idx:03d}"
                    )
                    case_ids.append(case_id)
                    index[case_id] = (patient, frame, slice_idx)
        self._case_index = index
        return case_ids, index

    def _load_or_generate_split(
        self, public_patients: list[str]
    ) -> dict:
        committed = load_committed_split("acdc_lv")
        if committed is not None:
            _, self._case_index = self._enumerate_cases(public_patients)
            return {
                "_all_cases": list(committed.get("_all_cases", [])),
                "folds": committed.get("folds", {}),
            }

        case_ids, _ = self._enumerate_cases(public_patients)

        rng = np.random.default_rng(42)
        shuffled = list(public_patients)
        rng.shuffle(shuffled)
        folds: dict[str, list[str]] = {}
        for k in range(_NUM_FOLDS):
            lo = k * _PATIENTS_PER_FOLD
            hi = lo + _PATIENTS_PER_FOLD
            val_patients = set(shuffled[lo:hi])
            train_patients = set(shuffled) - val_patients
            folds[f"fold_{k}_val"] = [
                cid
                for cid in case_ids
                if self._case_index[cid][0] in val_patients
            ]
            folds[f"fold_{k}_train"] = [
                cid
                for cid in case_ids
                if self._case_index[cid][0] in train_patients
            ]

        provenance = (
            "Patient-level 5-fold CV over ACDC patients 001-100 "
            "(SPEC §2). numpy.random.default_rng(seed=42).shuffle on "
            "patient list, 20 patients/fold. Patients 101-150 (with GT "
            "on disk but SPEC-withheld) are excluded. Slice cases "
            "retained where label==3 (LV) has any positive pixel."
        )
        persisted = {
            "train": folds["fold_0_train"],
            "val": [],
            "test": folds["fold_0_val"],
            "folds": folds,
            "_all_cases": case_ids,
        }
        save_committed_split("acdc_lv", persisted, provenance=provenance)
        return {"_all_cases": case_ids, "folds": folds}

    @staticmethod
    def _resolve_split(split: str, manifest: dict) -> list[str]:
        folds = manifest.get("folds", {})
        if split in folds:
            return list(folds[split])
        if split == "train":
            return list(folds.get("fold_0_train", []))
        if split == "val":
            return []
        if split == "test":
            return list(folds.get("fold_0_val", []))
        raise ValueError(
            f"ACDC split must be 'train', 'val', 'test', or "
            f"'fold_{{k}}_{{train,val}}'; got {split!r}"
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        patient, frame, slice_idx = self._case_index[case_id]
        img_path = self.images_dir / f"{patient}_frame{frame}.nii.gz"
        # ACDC masks use _gt suffix; fall back to plain name if missing.
        mask_path = self.masks_dir / f"{patient}_frame{frame}_gt.nii.gz"
        if not mask_path.is_file():
            fallback = self.masks_dir / f"{patient}_frame{frame}.nii.gz"
            if fallback.is_file():
                mask_path = fallback

        import nibabel as nib  # local: heavy optional dep

        vol = nib.load(str(img_path)).get_fdata()
        seg = nib.load(str(mask_path)).get_fdata()
        img_slice = np.asarray(vol[:, :, slice_idx], dtype=np.float32)
        seg_slice = np.asarray(seg[:, :, slice_idx])

        lo, hi = float(img_slice.min()), float(img_slice.max())
        if hi > lo:
            img_u8 = (
                (img_slice - lo) / (hi - lo) * 255.0
            ).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img_slice, dtype=np.uint8)

        img_t = torch.from_numpy(img_u8).unsqueeze(0).repeat(3, 1, 1)
        mask_bool = seg_slice == _LV_LABEL
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        img_t = _resize_uint8(img_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t, case_id


def _resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize a uint8 [C,H,W] tensor; return uint8 [C,size,size]."""
    import torch.nn.functional as F

    x = img.to(torch.float32).unsqueeze(0)
    x = F.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    return x.clamp(0, 255).squeeze(0).to(torch.uint8)


def _resize_bool(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest resize a bool [1,H,W] tensor; return bool [1,size,size]."""
    import torch.nn.functional as F

    x = mask.to(torch.float32).unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="nearest")
    return x.squeeze(0).to(torch.bool)
