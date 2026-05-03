#!/usr/bin/env python
"""Train a linear probe over a frozen FM encoder (SPEC §4).

Example
-------
    python scripts/train_frozen_fm.py \
        --task kvasir --fm dinov2_vitb14 --seed 42 \
        --data-root data/raw \
        --epochs 30 --batch-size 16 \
        --out results/per_case_dice/kvasir/dinov2_vitb14/

SPEC §4 is enforced strictly: the encoder is frozen in ``eval()`` mode with
``requires_grad=False``; only the 1x1 decoder is trained with Adam
(lr=1e-3) using per-case BCE-with-logits at the canonical 224 x 224
resolution. Per-case Dice on the test split is written to a JSON file
matching SPEC §8. The decoder checkpoint is SHA-256 hashed and logged in
the same JSON for provenance.

Exit codes
~~~~~~~~~~
* ``0`` - success.
* ``2`` - FM checkpoint unloadable (network, bad weights). SPEC §4 forbids
  falling back to random features; we hard-fail.
* ``3`` - dataset unavailable (missing on disk, missing split manifest).
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fmpool import datasets as fmpool_datasets
from fmpool import determinism, encoders
from fmpool.decoder import LinearSegHead

logger = logging.getLogger("fmpool.train_frozen_fm")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a frozen-FM linear seg probe (SPEC §4)."
    )
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory with raw datasets (SPEC §2). Exported as FMPOOL_DATA_ROOT.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default 0 for determinism; higher values "
             "require worker_init_fn seeding per SPEC §9).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-case Dice JSON (SPEC §8).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override default results/checkpoints/{task}/{fm}/ path.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a device ('cuda', 'cpu'). Auto-detect if None.",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_normaliser(
    spec: encoders.EncoderSpec,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """SPEC §4 normalise_fn: resize image+mask to 224x224 then apply per-FM mean/std.

    Dataset loaders pass ``(image_t, mask_t)`` pairs; this returns the
    normalised image (float32 in the FM's normalised distribution) and a
    nearest-neighbour-resized mask at 224x224 so downstream BCE loss can
    compare logits to GT pixels directly.
    """
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec.norm_std, dtype=torch.float32).view(3, 1, 1)

    def _fn(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"normalise_fn expects image [3, H, W], got {tuple(image.shape)}"
            )
        img = image.float()
        if img.max() > 1.5:
            img = img / 255.0
        img = F.interpolate(
            img.unsqueeze(0),
            size=INPUT_HW,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img = (img - mean) / std
        # Resize mask to 224x224 via nearest neighbour to preserve binary values.
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != INPUT_HW:
            m = F.interpolate(
                m.unsqueeze(0).float(), size=INPUT_HW, mode="nearest"
            ).squeeze(0)
        return img, m.to(torch.bool if mask.dtype == torch.bool else mask.dtype)

    return _fn


def _resize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Resize ``[1, H, W]`` binary mask to 224x224 using nearest-neighbour."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"expected [1, H, W] mask, got {tuple(mask.shape)}")
    resized = F.interpolate(
        mask.float().unsqueeze(0), size=INPUT_HW, mode="nearest"
    ).squeeze(0)
    return (resized > 0.5).float()


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([_resize_mask(item[1]) for item in batch], dim=0)
    ids = [item[2] for item in batch]
    return imgs, masks, ids


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _train_decoder(
    encoder: nn.Module,
    decoder: LinearSegHead,
    fm_id: str,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    decoder.train()
    opt = torch.optim.Adam(decoder.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        n_batches = 0
        running = 0.0
        for imgs, masks, _ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad():
                feats = encoders.extract_features(encoder, fm_id, imgs)
            logits = decoder(feats)
            loss = loss_fn(logits, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        avg = running / max(n_batches, 1)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate(
    encoder: nn.Module,
    decoder: LinearSegHead,
    fm_id: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    decoder.eval()
    from fmpool.estimators import dice_per_case

    all_ids: list[str] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks_np = (masks.numpy() > 0.5).astype(bool)
        feats = encoders.extract_features(encoder, fm_id, imgs)
        logits = decoder(feats)
        preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(bool)
        all_ids.extend(ids)
        all_preds.append(preds.reshape(preds.shape[0], -1))
        all_targets.append(masks_np.reshape(masks_np.shape[0], -1))
    if not all_preds:
        return [], np.zeros((0,), dtype=np.float64)
    pred_all = np.concatenate(all_preds, axis=0)
    tgt_all = np.concatenate(all_targets, axis=0)
    per_case = dice_per_case(pred_all, tgt_all)
    return all_ids, per_case


# ---------------------------------------------------------------------------
# Atomic JSON + checkpoint (SPEC §4 edge case: survive ENOSPC / crash).
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_checkpoint(decoder: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(decoder.state_dict(), tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    # Step 1: seed everything before anything stochastic.
    determinism.set_seed(args.seed)

    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Step 2: build frozen encoder. SPEC §4 forbids random-feature fallback.
    try:
        encoder, spec = encoders.build_encoder(args.fm)
    except Exception as exc:  # noqa: BLE001
        logger.error("FM %s unloadable: %s", args.fm, exc)
        print(
            f"[train_frozen_fm] ERROR: cannot load FM {args.fm!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    normalise_fn = make_normaliser(spec)

    # Step 3: build datasets (train + test).
    try:
        train_ds = fmpool_datasets.build_dataset(
            args.task, split="train", transform=normalise_fn
        )
        test_ds = fmpool_datasets.build_dataset(
            args.task, split="test", transform=normalise_fn
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        logger.error("dataset unavailable: %s", exc)
        print(
            f"[train_frozen_fm] ERROR: dataset {args.task!r} unavailable: {exc}",
            file=sys.stderr,
        )
        return 3

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=False,
    )

    # Step 4: build decoder on target device.
    decoder = LinearSegHead(
        in_dim=spec.feature_dim, num_classes=1, out_size=INPUT_HW
    ).to(device)

    # Step 5: train only the decoder.
    t0 = time.time()
    _train_decoder(
        encoder=encoder,
        decoder=decoder,
        fm_id=args.fm,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
    )
    train_elapsed = time.time() - t0

    # Step 6: evaluate on held-out test set.
    test_ids, per_case = _evaluate(
        encoder=encoder,
        decoder=decoder,
        fm_id=args.fm,
        loader=test_loader,
        device=device,
    )

    # Step 7: atomic checkpoint save + SHA-256.
    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path("results/checkpoints") / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_save_checkpoint(decoder, ckpt_path)
    ckpt_sha = determinism.sha256_file(ckpt_path)

    # Step 8: atomic per-case Dice JSON write (SPEC §8 schema).
    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    payload = {
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "n_test": int(len(per_case)),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(np.mean(per_case)) if per_case.size else float("nan"),
        "checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s (n_test=%d mean_dice=%.4f)",
        out_path,
        payload["n_test"],
        payload["mean_dice"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
