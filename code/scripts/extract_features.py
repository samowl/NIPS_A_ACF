#!/usr/bin/env python
"""PASS-A: extract+cache encoder features once per (task, fm).

Reads the committed train/test split for ``task``, runs the frozen FM
encoder under ``torch.autocast(bf16)`` once over both splits, and writes
``[N, D, h, w]`` fp16 feature tensors plus 224x224 uint8 mask tensors and
the case_ids into ``results/feature_cache/{task}/{fm}/{split}.pt``.

Subsequent PASS-B head trainers only need ``torch.load`` of these tensors
and a 1x1 conv decoder, eliminating per-seed/per-epoch encoder forwards.

Cache invariants are recorded in ``manifest.json`` so PASS-B can validate
provenance. Features are deterministic given the same (encoder weights,
data, normalisation).

Exit codes
----------
0 success, 2 FM unloadable, 3 dataset unavailable, 4 cache validation fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fmpool import datasets as fmpool_datasets
from fmpool import determinism, encoders

logger = logging.getLogger("fmpool.extract_features")

INPUT_HW: tuple[int, int] = (224, 224)
SCHEMA_VERSION: str = "feat_cache_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PASS-A feature extractor")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable bf16 autocast (fall back to fp32)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


def _make_normaliser(spec: encoders.EncoderSpec) -> Callable:
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec.norm_std, dtype=torch.float32).view(3, 1, 1)

    def _fn(image: torch.Tensor, mask: torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"image expected [3,H,W], got {tuple(image.shape)}")
        img = image.float()
        if img.max() > 1.5:
            img = img / 255.0
        img = F.interpolate(
            img.unsqueeze(0), size=INPUT_HW, mode="bilinear", align_corners=False
        ).squeeze(0)
        img = (img - mean) / std
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != INPUT_HW:
            m = F.interpolate(
                m.unsqueeze(0).float(), size=INPUT_HW, mode="nearest"
            ).squeeze(0)
        return img, m.to(torch.bool if mask.dtype == torch.bool else mask.dtype)

    return _fn


def _collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack(
        [item[1].squeeze(0) if item[1].ndim == 3 else item[1] for item in batch],
        dim=0,
    )  # [B, 224, 224]
    ids = [item[2] for item in batch]
    return imgs, masks, ids


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _hash_case_ids(ids: list[str]) -> str:
    h = hashlib.sha256()
    for cid in ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@torch.no_grad()
def _extract_split(
    encoder, fm_id: str, loader: DataLoader, device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int, int]:
    feats_chunks: list[torch.Tensor] = []
    masks_chunks: list[torch.Tensor] = []
    ids_all: list[str] = []
    n_total = 0
    feat_d = -1
    feat_h = -1
    feat_w = -1
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast_ctx:
            feats = encoders.extract_features(encoder, fm_id, imgs)
        feats_fp16 = feats.to(torch.float16).contiguous().cpu()
        feats_chunks.append(feats_fp16)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        masks_u8 = masks.to(torch.bool).to(torch.uint8).cpu()
        masks_chunks.append(masks_u8)
        ids_all.extend(ids)
        n_total += feats_fp16.shape[0]
        if feat_d < 0:
            feat_d = feats_fp16.shape[1]
            feat_h = feats_fp16.shape[2]
            feat_w = feats_fp16.shape[3]
    feats_all = torch.cat(feats_chunks, dim=0)
    masks_all = torch.cat(masks_chunks, dim=0)
    if feats_all.shape[0] != len(ids_all) or masks_all.shape[0] != len(ids_all):
        raise RuntimeError(
            f"extract: shape mismatch feats {feats_all.shape} masks "
            f"{masks_all.shape} ids {len(ids_all)}"
        )
    return feats_all, masks_all, ids_all, feat_d, feat_h


def _cache_dir(cache_root: Path, task: str, fm: str) -> Path:
    return Path(cache_root) / task / fm


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    determinism.set_seed(0)

    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    cache_root = Path(
        args.cache_root
        or Path(__file__).resolve().parents[1] / "results" / "feature_cache"
    )
    out_dir = _cache_dir(cache_root, args.task, args.fm)
    manifest_path = out_dir / "manifest.json"
    train_path = out_dir / "train.pt"
    test_path = out_dir / "test.pt"

    if manifest_path.is_file() and train_path.is_file() and test_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if (manifest.get("schema_version") == SCHEMA_VERSION
                    and manifest.get("fm") == args.fm
                    and manifest.get("task") == args.task):
                logger.info("cache already present at %s; skipping", out_dir)
                return 0
            logger.warning(
                "cache present but manifest mismatch (schema=%s); regenerating",
                manifest.get("schema_version"),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("manifest unreadable (%s); regenerating", exc)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        encoder, spec = encoders.build_encoder(args.fm)
    except Exception as exc:
        logger.error("FM %s unloadable: %s", args.fm, exc)
        return 2
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    normalise_fn = _make_normaliser(spec)

    try:
        train_ds = fmpool_datasets.build_dataset(args.task, "train", transform=normalise_fn)
        test_ds = fmpool_datasets.build_dataset(args.task, "test", transform=normalise_fn)
    except (FileNotFoundError, NotImplementedError) as exc:
        logger.error("dataset unavailable: %s", exc)
        return 3

    def _make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=_collate,
            drop_last=False,
        )

    train_loader = _make_loader(train_ds)
    test_loader = _make_loader(test_ds)

    use_amp = (not args.no_amp) and (device.type == "cuda")

    t0 = time.time()
    train_feats, train_masks, train_ids, fd, fh = _extract_split(
        encoder, args.fm, train_loader, device, use_amp
    )
    train_elapsed = time.time() - t0
    logger.info(
        "train: N=%d D=%d HxW=%dx%d %.1fs",
        train_feats.shape[0], fd, fh, train_feats.shape[3], train_elapsed,
    )

    t1 = time.time()
    test_feats, test_masks, test_ids, _, _ = _extract_split(
        encoder, args.fm, test_loader, device, use_amp
    )
    test_elapsed = time.time() - t1
    logger.info("test: N=%d %.1fs", test_feats.shape[0], test_elapsed)

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {"feats": train_feats, "masks": train_masks, "case_ids": train_ids},
        train_path,
    )
    _atomic_torch_save(
        {"feats": test_feats, "masks": test_masks, "case_ids": test_ids},
        test_path,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "fm": args.fm,
        "feature_dim": int(spec.feature_dim),
        "feature_hw": [int(train_feats.shape[2]), int(train_feats.shape[3])],
        "input_hw": list(INPUT_HW),
        "use_amp_bf16": bool(use_amp),
        "n_train": int(train_feats.shape[0]),
        "n_test": int(test_feats.shape[0]),
        "train_case_ids_sha256": _hash_case_ids(train_ids),
        "test_case_ids_sha256": _hash_case_ids(test_ids),
        "norm_mean": list(spec.norm_mean),
        "norm_std": list(spec.norm_std),
        "extract_seconds": float(train_elapsed + test_elapsed),
    }
    _atomic_write_bytes(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8"),
    )
    logger.info("wrote cache to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
