"""BraTS 2024 GLI multiclass loader (M1, 4-way softmax / 3 foreground classes).

This is the M1 multiclass variant of ``brats.py``. The image protocol
(FLAIR only, top-10 axial slices per subject, ``_MIN_TUMOUR_PIXELS=64``,
patient-level 80/20 split via numpy seed-42) is identical to the legacy
``BraTSWTDataset`` foreground loader, so both datasets can share the committed split file
``data/splits/brats_wt_split.json`` and the same image cache npz.

The only difference is the label channel:

* ``BraTSWTDataset``      -> binary mask ``[1, H, W] bool``  (seg > 0)
* ``BraTSMulticlassDataset`` -> multi-channel one-hot mask
  ``[C, H, W] bool`` where C=3 foreground classes:

    channel 0 -> NCR/RC  (label == 1 OR label == 4)
    channel 1 -> ED      (label == 2)
    channel 2 -> ET      (label == 3)

Background (label == 0) is implicit and reconstructed at training time as
``1 - any(mask, dim=0)``. The trainer then forms a ``[C+1, H, W]`` one-hot
target for cross-entropy.

BraTS-2024 GLI has labels {0, 1, 2, 3, 4} where 4 = RC (resection cavity);
SPEC §2 nominally targets BraTS-2023 with labels {0, 1, 2, 3}. We merge
label 4 into the NCR channel so the channel layout matches BraTS-2023
semantics (NCR + ED + ET) and is portable across schemas. See
``docs/DATA_NOTES.md`` for the upstream label discussion.
"""
from __future__ import annotations

import logging
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

# Foreground channel order. Indices into the multi-channel bool mask.
NUM_FG_CLASSES: int = 3
CHANNEL_NCR: int = 0
CHANNEL_ED: int = 1
CHANNEL_ET: int = 2


class BraTSMulticlassDataset(Dataset):
    """BraTS 2D axial-slice multiclass segmentation (M1).

    Returns ``(image_uint8 [3,224,224], mask_bool [3,224,224], case_id)``
    where the 3 mask channels are the per-class foreground bitmaps
    (NCR/RC, ED, ET). ``case_id = f"{subject_id}_slice{axial_idx:03d}"``.

    Shares the ``brats_wt`` committed split file and the per-subject image
    cache used by :class:`BraTSWTDataset`. The split is *not* re-derived
    here — if ``data/splits/brats_wt_split.json`` is missing, the user
    must instantiate ``BraTSWTDataset`` once first (or this class will
    fall back to generating the same split from scratch with the same
    seed-42 RNG).
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
        require_dir(self.root, dataset="brats_multiclass")

        self.subject_dirs = self._discover_subjects()
        if not self.subject_dirs:
            raise RuntimeError(
                f"[brats_multiclass] no subjects under "
                f"{self.root}/(training|additional). Expected unzipped "
                f"BraTS2024-BraTS-GLI training archives."
            )

        manifest = self._load_or_generate_split()
        self._slice_index: dict[str, tuple[str, int]] = {
            cid: (sub, axi)
            for cid, sub, axi in manifest["_case_entries"]
        }
        self.case_ids: list[str] = list(manifest[split])

    # ------------------------------------------------------------------
    # Subject discovery (mirror of brats.py)
    # ------------------------------------------------------------------
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
    # Slice selection is on the foreground mask (seg > 0) to keep
        # the slice index aligned across binary and multiclass datasets.
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
        # Reuse the committed brats_wt split: same patient-level 80/20 over
        # the same subject ID list with the same numpy seed-42 RNG.
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

        # Fallback: regenerate exactly the same split that brats.py would
        # produce, and persist it under the brats_wt key so both datasets
        # stay in lockstep.
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
            "BraTS 2024 GLI (on-disk) foreground-aligned slice index. Same "
            "selection protocol as brats_wt: per-subject top-10 axial "
            "FLAIR slices with tumour area >= 64 pixels. Patient-level "
            "80/20 split via numpy.random.default_rng(42) over the "
            "subject ID list. Persisted under brats_wt so both binary "
            "and multiclass datasets share the same split."
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

    # ------------------------------------------------------------------
    # Per-subject loading and caching.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.case_ids)

    def _subject_cache_path(self, sub_id: str) -> Path:
        # Separate npz from brats.py's binary cache: a different label
        # schema lives in a sibling file so the two datasets never collide.
        return self.root / "_cache_multiclass" / f"{sub_id}_slices.npz"

    def _load_subject_slices(
        self, sub_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return (img_u8 [K,H,W], seg_int [K,H,W] uint8, slice_indices [K]).

        The image normalisation is byte-identical to ``BraTSWTDataset``
        (per-slice min-max). The label volume is kept as raw integer
        labels {0,1,2,3,4} so the multiclass dataset can decode any
        channel layout without re-reading nibabel volumes.
        """
        cache = self._subject_cache_path(sub_id)
        if cache.is_file():
            try:
                arr = np.load(cache)
                return (
                    arr["img_u8"],
                    arr["seg_int"].astype(np.uint8),
                    arr["slice_indices"].tolist(),
                )
            except (OSError, KeyError, ValueError) as exc:
                log.warning(
                    "[brats_multiclass] cache %s unreadable (%s); rebuilding",
                    cache, exc,
                )
                try:
                    cache.unlink()
                except FileNotFoundError:
                    pass

        import nibabel as nib  # local: heavy optional dep

        sub_dir = self.subject_dirs[sub_id]
        flair_vol = nib.load(
            str(sub_dir / f"{sub_id}-t2f.nii.gz")
        ).get_fdata()
        seg_vol = nib.load(str(sub_dir / f"{sub_id}-seg.nii.gz")).get_fdata()
        kept = self._pick_slices(seg_vol)
        K = len(kept)
        H, W = int(seg_vol.shape[0]), int(seg_vol.shape[1])
        img_u8 = np.zeros((K, H, W), dtype=np.uint8)
        seg_int = np.zeros((K, H, W), dtype=np.uint8)
        for k, axi in enumerate(kept):
            img_slice = np.asarray(
                flair_vol[:, :, axi], dtype=np.float32
            )
            lo, hi = float(img_slice.min()), float(img_slice.max())
            if hi > lo:
                img_u8[k] = (
                    (img_slice - lo) / (hi - lo) * 255.0
                ).astype(np.uint8)
            else:
                img_u8[k] = 0
            seg_int[k] = np.asarray(
                seg_vol[:, :, axi], dtype=np.float32
            ).round().clip(0, 255).astype(np.uint8)

        cache.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via numpy's compressed savez. Use a unique tmp
        # filename so concurrent workers don't trample each other.
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(
            prefix=cache.name + ".tmp.", dir=str(cache.parent)
        )
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                img_u8=img_u8,
                seg_int=seg_int,
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

        return img_u8, seg_int, kept

    # ------------------------------------------------------------------
    # Sample assembly.
    # ------------------------------------------------------------------
    @staticmethod
    def _seg_to_multichannel(seg_int: np.ndarray) -> np.ndarray:
        """Decode raw label volume {0,1,2,3,4} to ``[C, H, W] bool``.

        Channel 0 = NCR/RC (label==1 or label==4)
        Channel 1 = ED      (label==2)
        Channel 2 = ET      (label==3)
        """
        ncr = (seg_int == 1) | (seg_int == 4)
        ed = seg_int == 2
        et = seg_int == 3
        return np.stack([ncr, ed, et], axis=0)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        sub_id, axi = self._slice_index[case_id]

        img_u8_stack, seg_int_stack, kept = self._load_subject_slices(sub_id)
        try:
            k = kept.index(int(axi))
        except ValueError as exc:
            raise RuntimeError(
                f"[brats_multiclass] axi={axi} not in cached slices for "
                f"{sub_id}: {kept}"
            ) from exc
        img_u8 = img_u8_stack[k]
        seg_int = seg_int_stack[k]
        mask_mc = self._seg_to_multichannel(seg_int)  # [C, H, W] bool

        img_t = (
            torch.from_numpy(np.ascontiguousarray(img_u8))
            .unsqueeze(0)
            .repeat(3, 1, 1)
        )
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_mc))

        img_t = _resize_uint8(img_t, _TARGET_HW)
        mask_t = _resize_bool_mc(mask_t, _TARGET_HW)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t, case_id


# ---------------------------------------------------------------------------
# Resize helpers (mirror brats.py but multi-channel for the mask).
# ---------------------------------------------------------------------------
def _resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F

    x = img.to(torch.float32).unsqueeze(0)
    x = F.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    return x.clamp(0, 255).squeeze(0).to(torch.uint8)


def _resize_bool_mc(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize for ``[C, H, W] bool`` masks."""
    import torch.nn.functional as F

    if mask.ndim != 3:
        raise ValueError(
            f"multichannel mask expected [C,H,W], got {tuple(mask.shape)}"
        )
    x = mask.to(torch.float32).unsqueeze(0)  # [1, C, H, W]
    x = F.interpolate(x, size=(size, size), mode="nearest")
    return x.squeeze(0).to(torch.bool)
