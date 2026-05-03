"""Medical segmentation dataset loaders (SPEC §2).

Public API:
    build_dataset(task, split, transform=None) -> torch.utils.data.Dataset
    list_splits(task) -> dict[str, list[str]]

All loaders return ``(image, mask, case_id)`` where ``image`` is a
``uint8`` tensor in ``[0, 255]`` of shape ``[3, H, W]`` (normalisation
is the training script's job per SPEC §4) and ``mask`` is a ``bool``
tensor of shape ``[1, H, W]``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PATHS_YAML = _REPO_ROOT / "configs" / "paths.yaml"
_SPLITS_DIR = _REPO_ROOT / "data" / "splits"

_DEFAULT_DATA_ROOT = str(_REPO_ROOT / "data" / "raw")

_VALID_TASKS: tuple[str, ...] = (
    "kvasir",
    "acdc_lv",
    "brats_wt",
    "brats_wt_multimodal",
    "brats_multiclass",
    "riga_cup",
    "riga_disc",
    "riga_cup_ood_messidor",
    "msd_hippocampus",
    "isic2018",
    "msd_heart",
    "msd_prostate",
    "msd_spleen",
)


Transform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SplitManifest:
    """Schema for ``data/splits/{task}_split.json`` (SPEC §2)."""

    task: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _load_paths_config() -> dict:
    """Load configs/paths.yaml if present; return {} otherwise."""
    if not _PATHS_YAML.is_file():
        return {}
    with _PATHS_YAML.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def resolve_data_root() -> Path:
    """Resolve the raw-data root.

    Priority: ``FMPOOL_DATA_ROOT`` env var > ``configs/paths.yaml``
    > repository-local default path. Does NOT require the directory to
    exist — callers are responsible for emitting a descriptive
    ``FileNotFoundError`` via :func:`require_dir`.
    """
    env = os.environ.get("FMPOOL_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    cfg = _load_paths_config()
    root = Path(cfg.get("data_root", _DEFAULT_DATA_ROOT))
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root


def resolve_task_root(task: str) -> Path:
    """Return the dataset directory for ``task`` under the data root."""
    cfg = _load_paths_config()
    subs = (cfg.get("datasets", {}) if cfg else {}) or {}
    default_sub = {
        "kvasir": "Kvasir-SEG",
        "acdc": "ACDC",
        "brats": "brats2024_gli",
        "brats_multiclass": "brats2024_gli",
        "brats_wt_multimodal": "brats2024_gli",
        "riga": "RIGA/RIGAPlus",
        "msd_hippocampus": "Task04_Hippocampus",
        "msd_heart": "Task02_Heart",
        "msd_prostate": "Task05_Prostate",
        "msd_spleen": "Task09_Spleen",
        "isic2018": "ISIC2018_400",
    }[task]
    return resolve_data_root() / subs.get(task, default_sub)


def require_dir(path: Path, *, dataset: str) -> None:
    """Raise a descriptive ``FileNotFoundError`` if ``path`` is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"[{dataset}] expected data directory not found: {path}\n"
            f"  Set FMPOOL_DATA_ROOT to a directory containing the dataset, "
            f"or edit {_PATHS_YAML}.\n"
            f"  Set FMPOOL_DATA_ROOT or edit configs/paths.yaml (default: {_DEFAULT_DATA_ROOT}/...)."
        )


# ---------------------------------------------------------------------------
# Split manifest persistence (SPEC §9)
# ---------------------------------------------------------------------------


def splits_path(task: str) -> Path:
    """Path to the committed split JSON for ``task``."""
    return _SPLITS_DIR / f"{task}_split.json"


def load_committed_split(task: str) -> Optional[dict]:
    """Return the committed split JSON dict, or ``None`` if absent."""
    p = splits_path(task)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_committed_split(task: str, split: dict, *, provenance: str) -> None:
    """Persist the split JSON with a provenance note (SPEC §9)."""
    _SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "provenance": provenance, **split}
    with splits_path(task).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


def load_split(task: str) -> SplitManifest:
    """Legacy typed wrapper around :func:`load_committed_split`."""
    if task not in _VALID_TASKS:
        raise KeyError(
            f"Unknown task {task!r}; expected one of {sorted(_VALID_TASKS)}"
        )
    raw = load_committed_split(task)
    if raw is None:
        raise FileNotFoundError(
            f"split manifest missing for task={task!r}: {splits_path(task)}. "
            "SPEC §2 requires committed split files before training."
        )
    for key in ("train", "val", "test"):
        if key not in raw:
            raise ValueError(
                f"split file {splits_path(task)} missing required key {key!r}"
            )
    return SplitManifest(
        task=task,
        train=tuple(raw["train"]),
        val=tuple(raw["val"]),
        test=tuple(raw["test"]),
    )


# ---------------------------------------------------------------------------
# Public dataset builders
# ---------------------------------------------------------------------------


def build_dataset(
    task: str,
    split: str,
    transform: Optional[Transform] = None,
) -> Dataset:
    """Instantiate a dataset for ``(task, split)``.

    Supported tasks: ``kvasir``, ``acdc_lv``, ``brats_wt``,
    ``riga_cup``, ``riga_disc``. Supported splits: ``train`` / ``val``
    / ``test`` for Kvasir/BraTS/RIGA; ACDC additionally accepts
    ``fold_{k}_train`` / ``fold_{k}_val`` for 5-fold CV.
    """
    if task not in _VALID_TASKS:
        raise KeyError(
            f"Unknown task {task!r}; expected one of {sorted(_VALID_TASKS)}"
        )
    # Import only the module actually needed — acdc/brats pull in
    # nibabel, which is not installed on dev machines without the
    # medical-imaging extras.
    if task == "kvasir":
        from fmpool.datasets.kvasir import KvasirSEG

        return KvasirSEG(split=split, transform=transform)
    if task == "acdc_lv":
        from fmpool.datasets.acdc import ACDCLVDataset

        return ACDCLVDataset(split=split, transform=transform)
    if task == "brats_wt":
        from fmpool.datasets.brats import BraTSWTDataset

        return BraTSWTDataset(split=split, transform=transform)
    if task == "brats_wt_multimodal":
        from fmpool.datasets.brats_multimodal import BraTSWTMultimodalDataset

        return BraTSWTMultimodalDataset(split=split, transform=transform)
    if task == "brats_multiclass":
        from fmpool.datasets.brats_multiclass import BraTSMulticlassDataset

        return BraTSMulticlassDataset(split=split, transform=transform)
    if task in {"riga_cup", "riga_disc"}:
        from fmpool.datasets.riga import RIGADataset

        target = "cup" if task == "riga_cup" else "disc"
        return RIGADataset(target=target, split=split, transform=transform)
    if task == "riga_cup_ood_messidor":
        from fmpool.datasets.riga import RIGAOODMessidorDataset

        return RIGAOODMessidorDataset(target="cup", split=split, transform=transform)
    if task == "msd_hippocampus":
        from fmpool.datasets.msd_hippocampus import MSDHippocampusDataset

        return MSDHippocampusDataset(split=split, transform=transform)
    if task == "msd_heart":
        from fmpool.datasets.msd_heart import MSDHeartDataset

        return MSDHeartDataset(split=split, transform=transform)
    if task == "msd_prostate":
        from fmpool.datasets.msd_prostate import MSDProstateDataset

        return MSDProstateDataset(split=split, transform=transform)
    if task == "msd_spleen":
        from fmpool.datasets.msd_spleen import MSDSpleenDataset

        return MSDSpleenDataset(split=split, transform=transform)
    if task == "isic2018":
        from fmpool.datasets.isic2018 import ISIC2018Dataset

        return ISIC2018Dataset(split=split, transform=transform)
    raise ValueError(f"unknown task: {task!r}")


def list_splits(task: str) -> dict[str, list[str]]:
    """Return the committed split dict for ``task``.

    Does not trigger split generation. Split files are produced by
    the first successful ``build_dataset`` call against real data.
    """
    committed = load_committed_split(task)
    if committed is None:
        raise FileNotFoundError(
            f"split JSON for task {task!r} has not been committed yet; "
            f"instantiate the dataset once to generate {splits_path(task)}"
        )
    return {k: committed[k] for k in ("train", "val", "test") if k in committed}


def prepare_nnunet_dataset(
    task: str,
    dataset_id: int,
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
) -> Path:
    """SPEC §5 nnU-Net raw-dataset materialiser (stub; wired separately)."""
    if task not in _VALID_TASKS:
        raise KeyError(
            f"Unknown task {task!r}; expected one of {sorted(_VALID_TASKS)}"
        )
    raise NotImplementedError(
        f"prepare_nnunet_dataset({task!r}, dataset_id={dataset_id}): "
        "concrete converter not yet wired."
    )


__all__ = [
    "SplitManifest",
    "Transform",
    "build_dataset",
    "list_splits",
    "load_split",
    "load_committed_split",
    "prepare_nnunet_dataset",
    "require_dir",
    "resolve_data_root",
    "resolve_task_root",
    "save_committed_split",
    "splits_path",
]
