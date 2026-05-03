#!/usr/bin/env python
"""M14: frozen loss/augmentation sensitivity -- Dice+BCE + flip/rotate augment.

Single-pass trainer that re-runs the encoder forward each epoch (no PASS-A
cache reuse) because augmentation must be applied to the image+mask pair
before the encoder sees them. The encoder is still frozen (eval,
requires_grad=False); only the 1x1 conv head is trained.

Resolution note (CRITICAL): earlier planning notes considered 336x336 input.
However the released frozen encoder zoo in ``fmpool.encoders`` hard-asserts 224x224
input (e.g. ``_DINOv2Wrapper.forward`` raises if the patch grid is not 16x16
at 224). Supporting non-native grids would require re-engineering positional
embeddings for every backbone, which is outside the M14 scope. We therefore
retain INPUT_HW=224 and record this in the JSON output. The sensitivity
signal (Dice+BCE loss, paired flip/rotate augmentation) is preserved.

Departures from SPEC §4 (the rest matches train_frozen_fm.py):
  - INPUT_HW = 224x224 (encoder native; see resolution note)
  - Loss = Dice + BCE-with-logits (default BCE only)
  - Augment = random horizontal flip + rotation +-15 deg (image+mask paired)
  - Otherwise: Adam lr=1e-3, 30 epochs, bf16 autocast on encoder + head

JSON output schema mirrors ``train_head_cached.py`` with extra keys:
  - "input_hw": [224, 224]
  - "loss_type": "dice_plus_bce"
  - "augment": "hflip+rot15"
  - "resolution_note": "M14 released at encoder-native 224; see paper caveat"

Exit codes
----------
0 success, 2 FM unloadable, 3 dataset unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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

logger = logging.getLogger("fmpool.train_head_clinical")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
ROT_DEG: float = 15.0
SCHEMA_VERSION: str = "m14_clinical_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M14 frozen standard-recipe trainer")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable bf16 autocast")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_normaliser(spec: encoders.EncoderSpec) -> Callable:
    """Resize image+mask to encoder-native 224x224.

    Mean/std normalisation is applied *after* augmentation inside the train
    step so rotation does not leak black-corner artefacts into normalised
    statistics. Returned image is float in [0, 1].
    """
    def _fn(image: torch.Tensor, mask: torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"image expected [3,H,W], got {tuple(image.shape)}")
        img = image.float()
        if img.max() > 1.5:
            img = img / 255.0
        img = F.interpolate(
            img.unsqueeze(0), size=INPUT_HW, mode="bilinear", align_corners=False
        ).squeeze(0)
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != INPUT_HW:
            m = F.interpolate(
                m.unsqueeze(0).float(), size=INPUT_HW, mode="nearest"
            ).squeeze(0)
        return img, m.to(torch.bool if mask.dtype == torch.bool else mask.dtype)

    return _fn


def _augment_pair(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal flip + rotation +-15 deg, applied per-sample to (img, mask).

    imgs:  [B, 3, H, W] float
    masks: [B, 1, H, W] float (binary 0/1)
    """
    B = imgs.shape[0]
    flip = (torch.rand(B, generator=gen) > 0.5).to(device)
    angles_deg = (torch.rand(B, generator=gen) * 2.0 - 1.0) * ROT_DEG
    angles = (angles_deg * math.pi / 180.0).to(device, dtype=torch.float32)

    if flip.any():
        flipped_imgs = torch.flip(imgs, dims=[-1])
        flipped_masks = torch.flip(masks, dims=[-1])
        sel = flip.view(B, 1, 1, 1)
        imgs = torch.where(sel, flipped_imgs, imgs)
        masks = torch.where(sel, flipped_masks, masks)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zeros = torch.zeros_like(cos)
    theta = torch.stack(
        [
            torch.stack([cos, sin, zeros], dim=-1),
            torch.stack([-sin, cos, zeros], dim=-1),
        ],
        dim=-2,
    )  # [B, 2, 3]
    grid = F.affine_grid(theta, imgs.shape, align_corners=False)
    # padding_mode="reflection" avoids black-corner artefacts at +-15 deg
    # rotation; mask uses zeros so introduced regions are background-not-
    # foreground (consistent with binary segmentation convention).
    imgs_rot = F.grid_sample(
        imgs, grid, mode="bilinear",
        padding_mode="reflection", align_corners=False,
    )
    masks_rot = F.grid_sample(
        masks, grid, mode="nearest",
        padding_mode="zeros", align_corners=False,
    )
    return imgs_rot, (masks_rot > 0.5).float()


def _normalise_for_encoder(
    imgs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (imgs - mean) / std


def _dice_plus_bce(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """Soft Dice + BCE-with-logits, equally weighted. logits/targets [B,1,H,W]."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).flatten(1).sum(dim=1)
    denom = probs.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return bce + dice.mean()


def _per_case_dice(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """preds, masks bool [B, H, W] -> float32 [B] (both empty -> 1.0)."""
    p = preds.flatten(1).to(torch.float32)
    m = masks.flatten(1).to(torch.float32)
    inter = (p * m).sum(dim=1)
    denom = p.sum(dim=1) + m.sum(dim=1)
    return torch.where(denom > 0, 2.0 * inter / denom, torch.ones_like(inter))


def _collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks_list = []
    for item in batch:
        m = item[1]
        if m.ndim == 2:
            m = m.unsqueeze(0)
        masks_list.append(m)
    masks = torch.stack(masks_list, dim=0).to(torch.float32)  # [B,1,H,W]
    ids = [item[2] for item in batch]
    return imgs, masks, ids


def _train(
    encoder: nn.Module,
    head: LinearSegHead,
    fm_id: str,
    loader: DataLoader,
    epochs: int,
    seed: int,
    use_amp: bool,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    autocast_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (use_amp and device.type == "cuda")
        else (lambda: torch.autocast(
            device_type="cpu", dtype=torch.bfloat16, enabled=False
        ))
    )
    for epoch in range(epochs):
        running = torch.zeros((), device=device, dtype=torch.float32)
        n_batches = 0
        for imgs, masks, _ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            imgs_aug, masks_aug = _augment_pair(imgs, masks, gen, device)
            imgs_norm = _normalise_for_encoder(imgs_aug, mean, std)
            opt.zero_grad(set_to_none=True)
            with autocast_factory():
                with torch.no_grad():
                    feats = encoders.extract_features(encoder, fm_id, imgs_norm)
                logits = head(feats)
                if logits.shape[-2:] != imgs_norm.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=imgs_norm.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                loss = _dice_plus_bce(logits.float(), masks_aug)
            loss.backward()
            opt.step()
            running += loss.detach().to(torch.float32)
            n_batches += 1
        avg = float(running.item()) / max(n_batches, 1)
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate(
    encoder: nn.Module,
    head: LinearSegHead,
    fm_id: str,
    loader: DataLoader,
    use_amp: bool,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    head.eval()
    all_ids: list[str] = []
    case_dices: list[torch.Tensor] = []
    autocast_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (use_amp and device.type == "cuda")
        else (lambda: torch.autocast(
            device_type="cpu", dtype=torch.bfloat16, enabled=False
        ))
    )
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        imgs_norm = _normalise_for_encoder(imgs, mean, std)
        with autocast_factory():
            feats = encoders.extract_features(encoder, fm_id, imgs_norm)
            logits = head(feats)
            if logits.shape[-2:] != imgs_norm.shape[-2:]:
                logits = F.interpolate(
                    logits, size=imgs_norm.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
        preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1)
        masks_bool = (masks > 0.5).squeeze(1).to(torch.bool)
        case_dices.append(_per_case_dice(preds, masks_bool).cpu())
        all_ids.extend(ids)
    if not case_dices:
        return [], torch.zeros((0,), dtype=torch.float32)
    return all_ids, torch.cat(case_dices, dim=0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    determinism.set_seed(args.seed)

    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        encoder, spec = encoders.build_encoder(args.fm)
    except Exception as exc:  # noqa: BLE001
        logger.error("FM %s unloadable: %s", args.fm, exc)
        return 2
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    mean = torch.tensor(spec.norm_mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor(spec.norm_std, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    normalise_fn = _make_normaliser(spec)

    try:
        train_ds = fmpool_datasets.build_dataset(
            args.task, split="train", transform=normalise_fn
        )
        test_ds = fmpool_datasets.build_dataset(
            args.task, split="test", transform=normalise_fn
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        logger.error("dataset unavailable: %s", exc)
        return 3

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate,
        pin_memory=pin, persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate,
        pin_memory=pin, persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    head = LinearSegHead(
        in_dim=spec.feature_dim, num_classes=1, out_size=INPUT_HW
    ).to(device)

    use_amp = (not args.no_amp) and (device.type == "cuda")

    t0 = time.time()
    _train(
        encoder=encoder, head=head, fm_id=args.fm,
        loader=train_loader, epochs=args.epochs, seed=args.seed,
        use_amp=use_amp, device=device, mean=mean, std=std,
    )
    train_elapsed = time.time() - t0

    test_ids, per_case = _evaluate(
        encoder=encoder, head=head, fm_id=args.fm,
        loader=test_loader, use_amp=use_amp, device=device,
        mean=mean, std=std,
    )

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(__file__).resolve().parents[1]
        / "results" / "checkpoints_m14" / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = _sha256_file(ckpt_path)

    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    payload = {
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "n_test": int(per_case.numel()),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(per_case.mean().item()) if per_case.numel() else float("nan"),
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": bool(use_amp),
        "input_hw": list(INPUT_HW),
        "loss_type": "dice_plus_bce",
        "augment": "hflip+rot15_reflection",
        "resolution_note": (
            "M14 released at encoder-native 224 because frozen encoders in "
            "fmpool.encoders require native grids (DINOv2/CLIP/CNN positional "
            "encodings hard-asserted to a 16x16 or 7x7 patch grid)."
        ),
        "schema_version": SCHEMA_VERSION,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s n_test=%d mean_dice=%.4f train_s=%.1f",
        out_path, payload["n_test"], payload["mean_dice"], train_elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
