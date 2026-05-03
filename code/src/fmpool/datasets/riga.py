"""RIGA+ fundus disc/cup loader (SPEC §2, row ``riga``; tasks
``riga_cup`` and ``riga_disc``).

Expected on-disk layout:
  <task_root>/BinRushed_{train,test}.csv
  <task_root>/Magrabia_{train,test}.csv
  <task_root>/MESSIDOR_Base{1,2,3}_{train,test}.csv
  <task_root>/RIGA/<source>/<subdir>/imageN.tif
  <task_root>/RIGA-mask/<source>/<subdir>/imageN-{1..6}.tif

Each rater TIFF is single-channel uint8 with values {0, 128, 255}:
0 = background, 128 = disc rim, 255 = cup (inner contour). Disc
binary = rater >= 128. Majority vote across 6 raters per SPEC §2
(``sum(rater_i) >= 4``).

SPEC §2 defines: train = BinRushed + MESSIDOR, test = Magrabia. We
pool the *train* and *test* CSVs within each source so the full
BinRushed / MESSIDOR / Magrabia complements are used (the per-file
sub-splits inside each source are unused here — see DATA_NOTES.md).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Optional

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

_NUM_RATERS = 6
_MAJORITY_THRESHOLD = 4
_TARGET_HW = 224
_CUP_PIXEL = 255
_DISC_MIN_PIXEL = 128

_TRAIN_SOURCES = (
    "BinRushed",
    "MESSIDOR_Base1",
    "MESSIDOR_Base2",
    "MESSIDOR_Base3",
)
_TEST_SOURCES = ("Magrabia",)


class RIGADataset(Dataset):
    """RIGA+ fundus cup/disc (SPEC §2, tasks ``riga_cup`` / ``riga_disc``).

    Returns ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)``
    with ``case_id`` equal to the CSV image path minus its leading
    ``RIGA/`` prefix (unique across the train+test pool).
    """

    def __init__(
        self,
        target: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if target not in {"cup", "disc"}:
            raise ValueError(
                f"RIGA target must be 'cup' or 'disc', got {target!r}"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"RIGA split must be train/val/test, got {split!r}"
            )
        self.target = target
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("riga")
        require_dir(self.root, dataset="riga")

        task_name = f"riga_{target}"
        manifest = self._load_or_generate_split(task_name)
        self._image_paths: dict[str, str] = manifest["_image_paths"]
        self.case_ids: list[str] = list(manifest[split])

    def _read_csv_rows(
        self, sources: Iterable[str]
    ) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for src in sources:
            for sub in ("train", "test"):
                csv_path = self.root / f"{src}_{sub}.csv"
                if not csv_path.is_file():
                    continue
                with csv_path.open("r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        img_rel = row["image"].strip()
                        mask_rel = row["mask"].strip()
                        if img_rel and mask_rel:
                            rows.append((img_rel, mask_rel))
        return rows

    def _load_or_generate_split(self, task_name: str) -> dict:
        committed = load_committed_split(task_name)
        if committed is not None:
            return {
                "train": list(committed.get("train", [])),
                "val": list(committed.get("val", [])),
                "test": list(committed.get("test", [])),
                "_image_paths": dict(committed.get("_image_paths", {})),
            }

        train_rows = self._read_csv_rows(_TRAIN_SOURCES)
        test_rows = self._read_csv_rows(_TEST_SOURCES)
        if not train_rows or not test_rows:
            raise RuntimeError(
                f"[riga] empty CSV pool: train={len(train_rows)}, "
                f"test={len(test_rows)}. Expected BinRushed + MESSIDOR "
                f"and Magrabia CSVs under {self.root}"
            )

        image_paths: dict[str, str] = {}
        train_ids: list[str] = []
        test_ids: list[str] = []
        for img_rel, _mask_rel in train_rows:
            cid = self._case_id_for(img_rel)
            image_paths[cid] = img_rel
            train_ids.append(cid)
        for img_rel, _mask_rel in test_rows:
            cid = self._case_id_for(img_rel)
            image_paths[cid] = img_rel
            test_ids.append(cid)

        provenance = (
            "RIGA+ official CSVs: train = BinRushed + MESSIDOR_Base1-3 "
            "(pooled both _train and _test CSVs per source); test = "
            "Magrabia (pooled). Majority vote across 6 rater TIFFs per "
            "SPEC §2: cup = sum(rater==255) ≥ 4, disc = "
            "sum(rater>=128) ≥ 4."
        )
        persisted = {
            "train": train_ids,
            "val": [],
            "test": test_ids,
            "_image_paths": image_paths,
        }
        save_committed_split(task_name, persisted, provenance=provenance)
        return {
            "train": train_ids,
            "val": [],
            "test": test_ids,
            "_image_paths": image_paths,
        }

    @staticmethod
    def _case_id_for(image_rel: str) -> str:
        if image_rel.startswith("RIGA/"):
            return image_rel[len("RIGA/"):]
        return image_rel

    def _rater_mask_paths(self, image_rel: str) -> list[Path]:
        img_path = Path(image_rel)
        stem = img_path.stem  # "image1"
        rel_parent = img_path.parent
        # Strip leading "RIGA" component so mask dir mirrors under RIGA-mask/
        parts = rel_parent.parts
        if parts and parts[0] == "RIGA":
            parts = parts[1:]
        mask_dir = self.root / "RIGA-mask" / Path(*parts)
        return [
            mask_dir / f"{stem}-{i}.tif"
            for i in range(1, _NUM_RATERS + 1)
        ]

    def _load_majority_mask(self, image_rel: str) -> np.ndarray:
        rater_paths = self._rater_mask_paths(image_rel)
        votes: Optional[np.ndarray] = None
        n_found = 0
        for p in rater_paths:
            if not p.is_file():
                continue
            rater = np.asarray(
                Image.open(p).convert("L"), dtype=np.uint8
            )
            if self.target == "cup":
                hit = rater == _CUP_PIXEL
            else:
                hit = rater >= _DISC_MIN_PIXEL
            votes = (
                hit.astype(np.uint8)
                if votes is None
                else votes + hit.astype(np.uint8)
            )
            n_found += 1
        if votes is None or n_found == 0:
            raise FileNotFoundError(
                f"[riga] no rater masks found for {image_rel} under "
                f"{self.root / 'RIGA-mask'}"
            )
        if n_found != _NUM_RATERS:
            # Silent <6-rater majority would inflate GT area (SPEC §2: 4/6).
            raise FileNotFoundError(
                f"[riga] expected {_NUM_RATERS} rater masks for {image_rel}, "
                f"found {n_found}. Cannot compute honest 4-of-6 majority vote."
            )
        return votes >= _MAJORITY_THRESHOLD

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        image_rel = self._image_paths[case_id]
        img_path = self.root / image_rel
        image = np.asarray(
            Image.open(img_path).convert("RGB"), dtype=np.uint8
        )
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        mask_bool = self._load_majority_mask(image_rel)
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        image_t = _resize_uint8(image_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)
        return image_t, mask_t, case_id


class RIGAOODMessidorDataset(RIGADataset):
    """RIGA+ leave-MESSIDOR-out OOD split (M13).

    Train pool = BinRushed + MAGRABIA (both ``_train`` and ``_test`` CSVs
    pooled per source). Test pool = MESSIDOR_Base{1,2,3} (both pooled).
    No examples leak across pools because the three sub-corpora are
    disjoint by acquisition site. ``val`` is empty (matches existing
    RIGA convention used by ``train_head_cached.py``).

    Persisted under task name ``riga_cup_ood_messidor`` so the split
    JSON is independent of the standard ``riga_cup`` split.
    """

    _OOD_TRAIN_SOURCES = ("BinRushed", "Magrabia")
    _OOD_TEST_SOURCES = (
        "MESSIDOR_Base1",
        "MESSIDOR_Base2",
        "MESSIDOR_Base3",
    )

    def __init__(
        self,
        target: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if target != "cup":
            raise ValueError(
                f"RIGAOODMessidorDataset only supports target='cup', "
                f"got {target!r}"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"RIGA OOD split must be train/val/test, got {split!r}"
            )
        # Bypass RIGADataset.__init__ to load the OOD-specific manifest.
        Dataset.__init__(self)
        self.target = target
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("riga")
        require_dir(self.root, dataset="riga")

        manifest = self._load_or_generate_ood_split(
            "riga_cup_ood_messidor"
        )
        self._image_paths: dict[str, str] = manifest["_image_paths"]
        self.case_ids: list[str] = list(manifest[split])

    def _load_or_generate_ood_split(self, task_name: str) -> dict:
        committed = load_committed_split(task_name)
        if committed is not None:
            train_ids = list(committed.get("train", []))
            test_ids = list(committed.get("test", []))
            # Re-run the disjointness check on every load: a stale or
            # manually edited JSON could otherwise leak MESSIDOR cases
            # into train without being detected.
            leak = sorted(set(train_ids) & set(test_ids))
            if leak:
                raise RuntimeError(
                    f"[riga_ood] committed split {task_name} has "
                    f"{len(leak)} case_ids in BOTH train and test pools; "
                    f"first 5: {leak[:5]}"
                )
            return {
                "train": train_ids,
                "val": list(committed.get("val", [])),
                "test": test_ids,
                "_image_paths": dict(committed.get("_image_paths", {})),
            }

        train_rows = self._read_csv_rows(self._OOD_TRAIN_SOURCES)
        test_rows = self._read_csv_rows(self._OOD_TEST_SOURCES)
        if not train_rows or not test_rows:
            raise RuntimeError(
                f"[riga_ood] empty CSV pool: "
                f"train(BinRushed+Magrabia)={len(train_rows)}, "
                f"test(MESSIDOR)={len(test_rows)}. Expected per-source "
                f"CSVs under {self.root}"
            )

        image_paths: dict[str, str] = {}
        train_ids: list[str] = []
        test_ids: list[str] = []
        for img_rel, _mask_rel in train_rows:
            cid = self._case_id_for(img_rel)
            image_paths[cid] = img_rel
            train_ids.append(cid)
        for img_rel, _mask_rel in test_rows:
            cid = self._case_id_for(img_rel)
            image_paths[cid] = img_rel
            test_ids.append(cid)

        # Defensive disjointness check: MESSIDOR vs BinRushed+Magrabia
        # share no case_ids by construction (different parent dirs), but
        # an accidental CSV duplication would silently leak. Fail loud.
        leak = sorted(set(train_ids) & set(test_ids))
        if leak:
            raise RuntimeError(
                f"[riga_ood] {len(leak)} case_ids appear in BOTH "
                f"train(BinRushed+Magrabia) and test(MESSIDOR) pools; "
                f"first 5: {leak[:5]}"
            )

        provenance = (
            "RIGA+ leave-MESSIDOR-out OOD (M13). train = BinRushed + "
            "Magrabia (pooled both _train and _test CSVs); test = "
            "MESSIDOR_Base1-3 (pooled). val = []. Cup target only; "
            "majority vote 4-of-6 raters identical to standard "
            "riga_cup loader."
        )
        persisted = {
            "train": train_ids,
            "val": [],
            "test": test_ids,
            "_image_paths": image_paths,
        }
        save_committed_split(task_name, persisted, provenance=provenance)
        return {
            "train": train_ids,
            "val": [],
            "test": test_ids,
            "_image_paths": image_paths,
        }


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
