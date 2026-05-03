"""BraTS 2024 GLI non-background foreground loader.

SPEC §2 specifies BraTS 2023 (1,251 subjects, labels {0,1,2,3}); the
data in the configured BraTS root is BraTS 2024 (1,350 + 271 additional subjects, labels
{0,1,2,3,4} with RC=4). See ``docs/DATA_NOTES.md``. The legacy task id is
``brats_wt``, but the binary target is ``seg > 0`` and therefore includes the
BraTS-GLI resection-cavity label; it is not the official 3D BraTS WT target.

Expected per-subject layout:
  <task_root>/training/training_data1_v2/<subject>/<subject>-{t1c,t1n,t2f,t2w,seg}.nii.gz
  <task_root>/additional/training_data_additional/<subject>/<subject>-{t1c,t1n,t2f,t2w,seg}.nii.gz

FLAIR-only 2D axial slice protocol (SPEC §2). For each subject, keep
up to the top-10 axial slices by tumour area, minimum 64 positive
pixels. Patient-level 80/20 split via numpy seed-42 over the subject
ID list.
"""
from __future__ import annotations

import logging
import os
import tempfile
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

log = logging.getLogger(__name__)

_MIN_TUMOUR_PIXELS = 64
_TOP_K_SLICES = 10
_TARGET_HW = 224
_TEST_FRACTION = 0.2
_SPLIT_SEED = 42


class BraTSWTDataset(Dataset):
    """BraTS 2D axial-slice non-background segmentation (``brats_wt`` id).

    Returns ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)``
    with ``case_id = f"{subject_id}_slice{axial_idx:03d}"``.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"BraTS split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("brats")
        require_dir(self.root, dataset="brats")

        self.subject_dirs = self._discover_subjects()
        if not self.subject_dirs:
            raise RuntimeError(
                f"[brats] no subjects under "
                f"{self.root}/(training|additional). Expected unzipped "
                f"BraTS2024-BraTS-GLI training archives."
            )

        manifest = self._load_or_generate_split()
        self._slice_index: dict[str, tuple[str, int]] = {
            cid: (sub, axi)
            for cid, sub, axi in manifest["_case_entries"]
        }
        self.case_ids: list[str] = list(manifest[split])

    def _discover_subjects(self) -> dict[str, Path]:
        subjects: dict[str, Path] = {}
        candidates = [
            self.root / "training" / "training_data1_v2",
            self.root / "additional" / "training_data_additional",
        ]
        for parent in candidates:
            if not parent.is_dir():
                continue
            for sub_dir in sorted(parent.iterdir()):
                if not sub_dir.is_dir():
                    continue
                sub_id = sub_dir.name
                seg = sub_dir / f"{sub_id}-seg.nii.gz"
                flair = sub_dir / f"{sub_id}-t2f.nii.gz"
                if seg.is_file() and flair.is_file():
                    subjects[sub_id] = sub_dir
        return subjects

    @staticmethod
    def _pick_slices(seg_vol: np.ndarray) -> list[int]:
        areas = (seg_vol > 0).reshape(-1, seg_vol.shape[-1]).sum(axis=0)
        keep = [
            int(i) for i, a in enumerate(areas) if a >= _MIN_TUMOUR_PIXELS
        ]
        keep.sort(key=lambda i: -int(areas[i]))
        return sorted(keep[:_TOP_K_SLICES])

    def _build_case_entries(self) -> list[tuple[str, str, int]]:
        import nibabel as nib  # local: heavy optional dep

        entries: list[tuple[str, str, int]] = []
        for sub_id, sub_dir in self.subject_dirs.items():
            seg_path = sub_dir / f"{sub_id}-seg.nii.gz"
            seg_vol = nib.load(str(seg_path)).get_fdata()
            for axi in self._pick_slices(seg_vol):
                case_id = f"{sub_id}_slice{axi:03d}"
                entries.append((case_id, sub_id, axi))
        return entries

    def _load_or_generate_split(self) -> dict:
        committed = load_committed_split("brats_wt")
        if committed is not None:
            case_entries = [
                (e[0], e[1], int(e[2]))
                for e in committed["_case_entries"]
            ]
            return {
                "train": list(committed.get("train", [])),
                "val": list(committed.get("val", [])),
                "test": list(committed.get("test", [])),
                "_case_entries": case_entries,
            }

        case_entries = self._build_case_entries()
        sub_ids = sorted({sub for _, sub, _ in case_entries})
        rng = np.random.default_rng(_SPLIT_SEED)
        shuffled = list(sub_ids)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * _TEST_FRACTION))
        test_subs = set(shuffled[:n_test])
        train_subs = set(shuffled) - test_subs

        train = [cid for cid, sub, _ in case_entries if sub in train_subs]
        test = [cid for cid, sub, _ in case_entries if sub in test_subs]

        provenance = (
            "BraTS 2024 GLI (on-disk) foreground = seg>0 "
            "(labels {1,2,3,4}; includes resection cavity label 4). "
            "Per-subject top-10 axial FLAIR slices with tumour area ≥ "
            "64 pixels. Patient-level 80/20 split via "
            "numpy.random.default_rng(42) over the subject ID list."
        )
        persisted = {
            "train": train,
            "val": [],
            "test": test,
            "_case_entries": [
                [cid, sub, axi] for cid, sub, axi in case_entries
            ],
        }
        save_committed_split("brats_wt", persisted, provenance=provenance)
        return {
            "train": train,
            "val": [],
            "test": test,
            "_case_entries": case_entries,
        }

    def __len__(self) -> int:
        return len(self.case_ids)

    def _subject_cache_path(self, sub_id: str) -> Path:
        return self.root / "_cache" / f"{sub_id}_slices.npz"

    def _load_subject_slices(
        self, sub_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return (img_u8 [K,H,W], seg_bool [K,H,W], slice_indices [K]).

        First call per subject reads both nibabel volumes once, normalises
        each kept slice to uint8 (per-slice min-max), and writes a compressed
        npz cache. Subsequent calls just decompress that npz (~5 MB, K<=10
        slices) instead of re-reading the full 240x240x155 volumes.

        Output is byte-identical to the previous per-call path for the same
            (subject, axi) pair: same per-slice normalisation, same seg>0 target.
        """
        cache = self._subject_cache_path(sub_id)
        if cache.is_file():
            try:
                arr = np.load(cache)
                return (
                    arr["img_u8"],
                    arr["seg_bool"].astype(bool),
                    arr["slice_indices"].tolist(),
                )
            except (OSError, KeyError, ValueError) as exc:
                log.warning(
                    "[brats] cache %s unreadable (%s); rebuilding", cache, exc
                )
                try:
                    cache.unlink()
                except FileNotFoundError:
                    pass

        import nibabel as nib  # local: heavy optional dep

        sub_dir = self.subject_dirs[sub_id]
        flair_vol = nib.load(str(sub_dir / f"{sub_id}-t2f.nii.gz")).get_fdata()
        seg_vol = nib.load(str(sub_dir / f"{sub_id}-seg.nii.gz")).get_fdata()
        kept = self._pick_slices(seg_vol)
        K = len(kept)
        H, W = int(seg_vol.shape[0]), int(seg_vol.shape[1])
        img_u8 = np.zeros((K, H, W), dtype=np.uint8)
        seg_bool = np.zeros((K, H, W), dtype=bool)
        for k, axi in enumerate(kept):
            img_slice = np.asarray(flair_vol[:, :, axi], dtype=np.float32)
            lo, hi = float(img_slice.min()), float(img_slice.max())
            if hi > lo:
                img_u8[k] = (
                    (img_slice - lo) / (hi - lo) * 255.0
                ).astype(np.uint8)
            else:
                img_u8[k] = 0
            seg_bool[k] = np.asarray(seg_vol[:, :, axi]) > 0

        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=cache.name + ".tmp.", dir=str(cache.parent)
        )
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                img_u8=img_u8,
                seg_bool=seg_bool,
                slice_indices=np.asarray(kept, dtype=np.int64),
            )
            # numpy adds .npz when missing; handle both cases atomically.
            tmp_npz = tmp + ".npz" if not tmp.endswith(".npz") else tmp
            if tmp_npz != tmp and Path(tmp_npz).is_file():
                os.replace(tmp_npz, cache)
                if Path(tmp).exists():
                    os.unlink(tmp)
            else:
                os.replace(tmp, cache)
        except BaseException:
            for cleanup in (tmp, tmp + ".npz"):
                try:
                    os.unlink(cleanup)
                except FileNotFoundError:
                    pass
            raise

        return img_u8, seg_bool, kept

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        sub_id, axi = self._slice_index[case_id]

        img_u8_stack, seg_bool_stack, kept = self._load_subject_slices(sub_id)
        try:
            k = kept.index(int(axi))
        except ValueError as exc:
            raise RuntimeError(
                f"[brats] axi={axi} not in cached slices for {sub_id}: {kept}"
            ) from exc
        img_u8 = img_u8_stack[k]
        mask_bool = seg_bool_stack[k]

        img_t = torch.from_numpy(np.ascontiguousarray(img_u8)).unsqueeze(0).repeat(3, 1, 1)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_bool)).unsqueeze(0)

        img_t = _resize_uint8(img_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t, case_id


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
