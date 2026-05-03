"""ISIC 2018 Task 1 binary skin-lesion segmentation loader.

Expected on-disk layout::

    <task_root>/
        ISIC2018_Task1-2_Training_Input/<ISIC_id>.jpg          (400 images)
        ISIC2018_Task1_Training_GroundTruth/<ISIC_id>_segmentation.png

Per the held-out appendix, we use the staged 400-image subset.
Mask is single-channel uint8 in {0, 255}; binarise via ``> 127``.
80/20 patient-level split (each ISIC image is one patient/lesion case)
via ``numpy.random.default_rng(42)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fmpool.datasets import (
    load_committed_split,
    require_dir,
    resolve_task_root,
    save_committed_split,
)

_TEST_FRACTION = 0.2
_SPLIT_SEED = 42
_MASK_BINARISE_THRESHOLD = 127
_EXPECTED_TOTAL = 400

_INPUT_SUBDIR = "ISIC2018_Task1-2_Training_Input"
_GT_SUBDIR = "ISIC2018_Task1_Training_GroundTruth"


class ISIC2018Dataset(Dataset):
    """ISIC 2018 Task 1 skin-lesion binary segmentation.

    Returns ``(image_uint8 [3,H,W], mask_bool [1,H,W], case_id)`` where
    ``case_id`` is the ISIC stem (e.g. ``ISIC_0000000``). Native image
    resolution is preserved (downstream resize is the training script's
    job, matching kvasir.py behaviour).
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"isic2018 split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("isic2018")
        require_dir(self.root, dataset="isic2018")

        self.images_dir = self.root / _INPUT_SUBDIR
        self.masks_dir = self.root / _GT_SUBDIR
        require_dir(self.images_dir, dataset="isic2018")
        require_dir(self.masks_dir, dataset="isic2018")

        manifest = self._load_or_generate_split()
        self.case_ids: list[str] = list(manifest[split])

    @staticmethod
    def _list_jpgs(directory: Path) -> list[str]:
        out: list[str] = []
        for p in sorted(directory.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("._"):
                continue
            if p.suffix.lower() != ".jpg":
                continue
            out.append(name)
        return out

    def _discover_cases(self) -> list[str]:
        image_files = self._list_jpgs(self.images_dir)
        cases: list[str] = []
        for fname in image_files:
            stem = Path(fname).stem  # ISIC_0000000
            mask_path = self.masks_dir / f"{stem}_segmentation.png"
            if not mask_path.is_file():
                continue
            cases.append(stem)
        return cases

    def _load_or_generate_split(self) -> dict:
        committed = load_committed_split("isic2018")
        if committed is not None:
            return {
                "train": list(committed.get("train", [])),
                "val": list(committed.get("val", [])),
                "test": list(committed.get("test", [])),
            }

        cases = self._discover_cases()
        if not cases:
            raise RuntimeError(
                f"[isic2018] no paired (image, mask) cases under "
                f"{self.root}"
            )
        if len(cases) != _EXPECTED_TOTAL:
            count_note = (
                f" Found {len(cases)} cases (expected {_EXPECTED_TOTAL})."
            )
        else:
            count_note = f" Found {len(cases)} cases."

        rng = np.random.default_rng(_SPLIT_SEED)
        shuffled = list(cases)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * _TEST_FRACTION))
        if n_test == 0 and len(shuffled) > 1:
            n_test = 1
        test_cases = sorted(shuffled[:n_test])
        train_cases = sorted(shuffled[n_test:])

        provenance = (
            "ISIC 2018 Task 1 binary skin-lesion segmentation, "
            "400-image subset per the held-out appendix. "
            "Patient-level (= per-lesion) 80/20 split via "
            "numpy.random.default_rng(42) over the ISIC ID list. "
            "Binary mask = (uint8 > 127)." + count_note
        )
        persisted = {
            "train": train_cases,
            "val": [],
            "test": test_cases,
        }
        save_committed_split("isic2018", persisted, provenance=provenance)
        return {"train": train_cases, "val": [], "test": test_cases}

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        img_path = self.images_dir / f"{case_id}.jpg"
        mask_path = self.masks_dir / f"{case_id}_segmentation.png"

        image = np.asarray(
            Image.open(img_path).convert("RGB"), dtype=np.uint8
        )
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        mask_raw = np.asarray(
            Image.open(mask_path).convert("L"), dtype=np.uint8
        )
        mask_bool = mask_raw > _MASK_BINARISE_THRESHOLD
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)
        return image_t, mask_t, case_id
