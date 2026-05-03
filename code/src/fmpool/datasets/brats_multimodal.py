"""BraTS 2024 GLI foreground multimodal RGB loader (M20).

Variant of :mod:`fmpool.datasets.brats` that stacks three MRI modalities
into the RGB channels of the input tensor:

    R = T1c (post-contrast T1)    -> ``<sub>-t1c.nii.gz``
    G = T2w (T2-weighted)          -> ``<sub>-t2w.nii.gz``
    B = FLAIR (T2-FLAIR)           -> ``<sub>-t2f.nii.gz``

The legacy task id is ``brats_wt_multimodal``, but the target is the loader's
binary ``seg > 0`` foreground derivative. For BraTS-GLI this includes the
resection-cavity label and is not the official 3D BraTS WT challenge target.

Slice selection (top-K axial slices by foreground area, >= 64 positive pixels)
and the patient-level 80/20 split are inherited from the unimodal loader,
but persisted under a separate task name so the cached split JSON does
not collide.

Per-slice min-max normalisation is applied independently per modality
(matches the unimodal loader's per-slice normalisation; intensity
distributions of t1c / t2w / t2f differ substantially, so a shared
scaling would distort relative contrast).

Output contract (matches all other 2D segmenters):
    image: ``uint8`` tensor ``[3, 224, 224]`` (R=t1c, G=t2w, B=flair)
    mask:  ``bool``  tensor ``[1, 224, 224]`` (seg > 0)
    case_id: ``f"{subject_id}_slice{axial_idx:03d}"``

Cache layout (kept disjoint from unimodal ``_cache/``):
    ``<task_root>/_cache_multimodal/<sub_id>_slices.npz``
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

_TASK_NAME = "brats_wt_multimodal"


class BraTSWTMultimodalDataset(Dataset):
    """BraTS foreground 2D multimodal segmentation (M20).

    Slice protocol matches :class:`BraTSWTDataset` but pulls T1c + T2w +
    FLAIR rather than FLAIR alone, and the per-modality normalised slices
    are packed into the RGB channels of a uint8 ``[3, H, W]`` tensor.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"BraTS multimodal split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("brats_wt_multimodal")
        require_dir(self.root, dataset="brats_wt_multimodal")

        self.subject_dirs = self._discover_subjects()
        if not self.subject_dirs:
            raise RuntimeError(
                f"[brats_wt_multimodal] no subjects under "
                f"{self.root}/(training|additional). Expected unzipped "
                f"BraTS2024-BraTS-GLI training archives with t1c/t2w/t2f/seg."
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
                t1c = sub_dir / f"{sub_id}-t1c.nii.gz"
                t2w = sub_dir / f"{sub_id}-t2w.nii.gz"
                flair = sub_dir / f"{sub_id}-t2f.nii.gz"
                if (
                    seg.is_file()
                    and t1c.is_file()
                    and t2w.is_file()
                    and flair.is_file()
                ):
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
        committed = load_committed_split(_TASK_NAME)
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
            "BraTS 2024 GLI foreground multimodal (M20). Modalities: R=t1c, "
            "G=t2w, B=t2f (FLAIR). Per-subject top-10 axial slices by "
            "foreground (seg>0) area >= 64 pixels, identical slice picks "
            "to brats_wt unimodal. Patient-level 80/20 split via "
            "numpy.random.default_rng(42) over the subject ID list. "
            "Per-slice per-modality min-max normalisation to uint8."
        )
        persisted = {
            "train": train,
            "val": [],
            "test": test,
            "_case_entries": [
                [cid, sub, axi] for cid, sub, axi in case_entries
            ],
        }
        save_committed_split(_TASK_NAME, persisted, provenance=provenance)
        return {
            "train": train,
            "val": [],
            "test": test,
            "_case_entries": case_entries,
        }

    def __len__(self) -> int:
        return len(self.case_ids)

    def _subject_cache_path(self, sub_id: str) -> Path:
        return self.root / "_cache_multimodal" / f"{sub_id}_slices.npz"

    @staticmethod
    def _normalise_slice_uint8(slice_2d: np.ndarray) -> np.ndarray:
        """Per-slice independent min-max -> uint8 (matches brats.py)."""
        s = np.asarray(slice_2d, dtype=np.float32)
        lo, hi = float(s.min()), float(s.max())
        if hi > lo:
            return ((s - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return np.zeros_like(s, dtype=np.uint8)

    def _load_subject_slices(
        self, sub_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return (img_u8 [K,3,H,W], seg_bool [K,H,W], slice_indices [K]).

        First call per subject reads three nibabel volumes (t1c, t2w, flair)
        once plus the seg, picks slices via the same protocol as the
        unimodal loader, normalises each kept slice independently per
        modality to uint8, and writes a compressed npz cache distinct
        from the unimodal ``_cache/`` (note ``_cache_multimodal``).
        """
        cache = self._subject_cache_path(sub_id)
        if cache.is_file():
            try:
                arr = np.load(cache)
                img_u8 = arr["img_u8"]
                if img_u8.ndim != 4 or img_u8.shape[1] != 3:
                    raise ValueError(
                        f"cache schema mismatch for {sub_id}: expected "
                        f"img_u8 [K,3,H,W], got {tuple(img_u8.shape)}. "
                        f"Likely a stale unimodal cache file at "
                        f"{cache}; delete it and re-extract."
                    )
                return (
                    img_u8,
                    arr["seg_bool"].astype(bool),
                    arr["slice_indices"].tolist(),
                )
            except (OSError, KeyError, ValueError) as exc:
                log.warning(
                    "[brats_wt_multimodal] cache %s unreadable (%s); "
                    "rebuilding",
                    cache,
                    exc,
                )
                try:
                    cache.unlink()
                except FileNotFoundError:
                    pass

        import nibabel as nib  # local: heavy optional dep

        sub_dir = self.subject_dirs[sub_id]
        t1c_vol = nib.load(str(sub_dir / f"{sub_id}-t1c.nii.gz")).get_fdata()
        t2w_vol = nib.load(str(sub_dir / f"{sub_id}-t2w.nii.gz")).get_fdata()
        flair_vol = nib.load(str(sub_dir / f"{sub_id}-t2f.nii.gz")).get_fdata()
        seg_vol = nib.load(str(sub_dir / f"{sub_id}-seg.nii.gz")).get_fdata()

        kept = self._pick_slices(seg_vol)
        K = len(kept)
        H, W = int(seg_vol.shape[0]), int(seg_vol.shape[1])
        img_u8 = np.zeros((K, 3, H, W), dtype=np.uint8)
        seg_bool = np.zeros((K, H, W), dtype=bool)
        for k, axi in enumerate(kept):
            img_u8[k, 0] = self._normalise_slice_uint8(t1c_vol[:, :, axi])
            img_u8[k, 1] = self._normalise_slice_uint8(t2w_vol[:, :, axi])
            img_u8[k, 2] = self._normalise_slice_uint8(flair_vol[:, :, axi])
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
                f"[brats_wt_multimodal] axi={axi} not in cached slices "
                f"for {sub_id}: {kept}"
            ) from exc
        img_u8 = img_u8_stack[k]      # [3, H, W]
        mask_bool = seg_bool_stack[k]  # [H, W]

        img_t = torch.from_numpy(np.ascontiguousarray(img_u8))
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
