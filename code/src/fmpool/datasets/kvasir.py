"""Kvasir-SEG polyp segmentation loader (SPEC §2, row ``kvasir``).

Per SPEC §2: 1000 cases total, binary mask via ``mask > 127`` (JPEG
lossy), 880/120 train/test split. If the upstream repo ships
``train.txt`` / ``val.txt``, we honour it; otherwise we fall back to
an alphabetical 880/120 split and commit provenance to
``data/splits/kvasir_split.json`` (SPEC §9).
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

_EXPECTED_TOTAL = 1000
_TRAIN_COUNT = 880
_TEST_COUNT = 120
_MASK_BINARISE_THRESHOLD = 127


class KvasirSEG(Dataset):
    """Kvasir-SEG polyp segmentation (SPEC §2).

    Returns ``(image_uint8 [3,H,W], mask_bool [1,H,W], case_id)``.
    No augmentation or normalisation here — SPEC §4 training script
    applies the FM-specific normalisation.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Kvasir split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("kvasir")
        require_dir(self.root, dataset="kvasir")

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        require_dir(self.images_dir, dataset="kvasir")
        require_dir(self.masks_dir, dataset="kvasir")

        image_files = sorted(
            p.name for p in self.images_dir.iterdir()
            if p.suffix.lower() == ".jpg"
        )
        if len(image_files) != _EXPECTED_TOTAL:
            raise RuntimeError(
                f"[kvasir] expected {_EXPECTED_TOTAL} images in "
                f"{self.images_dir}, found {len(image_files)}"
            )

        self.case_ids: list[str] = self._load_or_generate_split(image_files)[
            split
        ]

    @staticmethod
    def _find_official_split(root: Path) -> Optional[dict[str, list[str]]]:
        """Locate upstream train.txt/val.txt if present.

        The DebeshJha/Kvasir-SEG GitHub repo may ship a split file.
        When no split file is present in the configured data root,
        return ``None`` to trigger the deterministic fallback split.
        """
        train_txt = root / "train.txt"
        test_txt_candidates = [root / "val.txt", root / "test.txt"]
        if not train_txt.is_file():
            return None
        test_txt = next(
            (p for p in test_txt_candidates if p.is_file()), None
        )
        if test_txt is None:
            return None

        def _read(p: Path) -> list[str]:
            return [
                line.strip()
                for line in p.read_text().splitlines()
                if line.strip()
            ]

        return {
            "train": _read(train_txt),
            "val": [],
            "test": _read(test_txt),
        }

    def _load_or_generate_split(
        self, image_files: list[str]
    ) -> dict[str, list[str]]:
        committed = load_committed_split("kvasir")
        if committed is not None:
            return {
                "train": list(committed.get("train", [])),
                "val": list(committed.get("val", [])),
                "test": list(committed.get("test", [])),
            }

        official = self._find_official_split(self.root)
        if official is not None:
            split = official
            provenance = (
                "Official DebeshJha/Kvasir-SEG train.txt / val.txt on disk."
            )
        else:
            stems = [Path(f).stem for f in sorted(image_files)]
            if len(stems) != _EXPECTED_TOTAL:
                raise RuntimeError(
                    f"[kvasir] fallback split impossible: have "
                    f"{len(stems)} stems, expected {_EXPECTED_TOTAL}"
                )
            split = {
                "train": stems[:_TRAIN_COUNT],
                "val": [],
                "test": stems[_TRAIN_COUNT:],
            }
            provenance = (
                "Fallback alphabetical split: first 880 stems train, "
                "last 120 stems test. No upstream train.txt/val.txt was "
                "present on disk at init time."
            )
        if (
            len(split["train"]) != _TRAIN_COUNT
            or len(split["test"]) != _TEST_COUNT
        ):
            raise RuntimeError(
                f"[kvasir] split count mismatch: "
                f"train={len(split['train'])} (expect {_TRAIN_COUNT}) "
                f"test={len(split['test'])} (expect {_TEST_COUNT})"
            )
        save_committed_split("kvasir", split, provenance=provenance)
        return split

    def __len__(self) -> int:
        return len(self.case_ids)

    def _resolve(self, case_id: str, where: Path) -> Path:
        candidate = where / case_id
        if candidate.is_file():
            return candidate
        candidate_jpg = where / f"{case_id}.jpg"
        if candidate_jpg.is_file():
            return candidate_jpg
        raise FileNotFoundError(
            f"[kvasir] file not found for case_id={case_id!r} in {where}"
        )

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        img_path = self._resolve(case_id, self.images_dir)
        mask_path = self._resolve(case_id, self.masks_dir)

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

        return image_t, mask_t, Path(case_id).stem
