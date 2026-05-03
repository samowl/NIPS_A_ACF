#!/usr/bin/env python
"""M1 PASS-B: train a 1x1-conv multiclass decoder on cached features.

Variant of ``train_head_cached.py`` for the M1 multiclass matrix.

Differences vs ``train_head_cached.py``:
  * head out_channels = NUM_CLASSES = 4 (background + 3 foreground:
    NCR/RC, ED, ET) trained with cross-entropy softmax instead of BCE.
  * masks are NOT taken from the brats_wt feature cache (those are
    binary). Instead the cached ``case_ids`` are used to look up
    per-case multi-channel masks via :class:`BraTSMulticlassDataset`,
    keeping the feature side identical to ``brats_wt``.
  * per-case Dice is the mean over the 3 foreground classes; the
    output JSON additionally records per-class Dice (length C-1) and
    a per-case-per-class matrix.

The (task, fm) feature cache is the same artifact produced by
``extract_features.py`` for ``brats_wt``. We verify the manifest's
``task == "brats_wt"`` (or ``brats_multiclass`` if you ran a fresh
extract for the new task — both are accepted) before training.

Exit codes
----------
0 success, 3 cache missing/invalid, 4 manifest mismatch.
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
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmpool import determinism
from fmpool.decoder import LinearSegHead

logger = logging.getLogger("fmpool.train_head_multiclass")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
SCHEMA_VERSION: str = "feat_cache_v1_multiclass"
# Background + NCR/RC + ED + ET = 4-way softmax.
NUM_CLASSES: int = 4
CLASS_NAMES: tuple[str, ...] = ("background", "ncr_rc", "ed", "et")
# Indices of the 3 foreground classes in the 4-way label space.
FG_CLASS_INDICES: tuple[int, ...] = (1, 2, 3)

# (task, fm) cache compatibility. We accept caches built for either
# brats_wt (preferred — same image+normalisation, only the masks differ
# and we rebuild them from disk) or brats_multiclass (in case a parallel
# extract was run with the new task name).
COMPATIBLE_CACHE_TASKS: frozenset[str] = frozenset(
    {"brats_wt", "brats_multiclass"}
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M1 PASS-B head trainer (multiclass softmax)"
    )
    p.add_argument("--task", required=True,
                   help="logical task (must be 'brats_multiclass')")
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cache-task", default="brats_wt",
                   help="task name under cache-root (default: brats_wt — "
                        "we reuse the binary cache because the FM features "
                        "are mask-independent)")
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable bf16 autocast on head training")
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


# ---------------------------------------------------------------------------
# Cache loading.
# ---------------------------------------------------------------------------
def _load_cache(cache_dir: Path) -> tuple[dict, dict, dict]:
    """Load the brats_wt-compatible feature cache.

    The masks tensor is left untouched (it is binary uint8 per
    ``feat_cache_v1``); the multiclass trainer ignores it and rebuilds
    masks from :class:`BraTSMulticlassDataset`.
    """
    manifest_path = cache_dir / "manifest.json"
    train_path = cache_dir / "train.pt"
    test_path = cache_dir / "test.pt"
    if (not manifest_path.is_file()
            or not train_path.is_file()
            or not test_path.is_file()):
        raise FileNotFoundError(
            f"feature cache incomplete in {cache_dir}: "
            f"manifest={manifest_path.is_file()} "
            f"train={train_path.is_file()} test={test_path.is_file()}"
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != "feat_cache_v1":
        raise ValueError(
            f"manifest schema_version="
            f"{manifest.get('schema_version')!r} != expected 'feat_cache_v1'"
        )
    train_blob = torch.load(train_path, map_location="cpu", weights_only=False)
    test_blob = torch.load(test_path, map_location="cpu", weights_only=False)
    return manifest, train_blob, test_blob


# ---------------------------------------------------------------------------
# Mask reconstruction. Since the brats_wt cache stores binary WT masks but
# we need 3-channel multiclass masks, we re-load them from the dataset
# class. The returned tensor is ordered exactly as the cache's case_ids
# list, so it lines up with the cached features index-for-index.
# ---------------------------------------------------------------------------
def _build_multiclass_mask_tensor(
    case_ids: Sequence[str], split: str
) -> torch.Tensor:
    """Return ``[N, C_fg, 224, 224] uint8`` masks aligned with ``case_ids``.

    C_fg = 3 foreground channels (NCR/RC, ED, ET). Per-subject NPZ
    caches are decompressed at most once per call, even if many slices
    of the same subject are present.
    """
    from fmpool.datasets.brats_multiclass import (
        BraTSMulticlassDataset,
        NUM_FG_CLASSES,
        _resize_bool_mc,
    )

    ds = BraTSMulticlassDataset(split=split, transform=None)
    cid_to_idx = {cid: i for i, cid in enumerate(ds.case_ids)}
    missing = [cid for cid in case_ids if cid not in cid_to_idx]
    if missing:
        raise RuntimeError(
            f"[m1] {len(missing)} case_ids in feature cache are not in "
            f"BraTSMulticlassDataset.{split}: first 5 = {missing[:5]}"
        )
    out = torch.zeros(
        (len(case_ids), NUM_FG_CLASSES, INPUT_HW[0], INPUT_HW[1]),
        dtype=torch.uint8,
    )
    # Group output positions by subject so each subject's NPZ is loaded
    # exactly once per call.
    by_subject: dict[str, list[tuple[int, int]]] = {}
    for i, cid in enumerate(case_ids):
        sub_id, axi = ds._slice_index[cid]
        by_subject.setdefault(sub_id, []).append((i, int(axi)))

    for sub_id, slot_list in by_subject.items():
        img_u8_stack, seg_int_stack, kept = ds._load_subject_slices(sub_id)
        for out_i, axi in slot_list:
            try:
                k = kept.index(int(axi))
            except ValueError as exc:
                raise RuntimeError(
                    f"[m1] axi={axi} not in cached slices for {sub_id}: "
                    f"{kept}"
                ) from exc
            seg_int = seg_int_stack[k]
            mask_mc_np = ds._seg_to_multichannel(seg_int)
            mask_t = torch.from_numpy(mask_mc_np)  # [C, H, W] bool
            mask_t = _resize_bool_mc(mask_t, INPUT_HW[0])
            if mask_t.shape != (NUM_FG_CLASSES, INPUT_HW[0], INPUT_HW[1]):
                raise RuntimeError(
                    f"[m1] mask shape {tuple(mask_t.shape)} for sub_id="
                    f"{sub_id!r} axi={axi} != expected "
                    f"({NUM_FG_CLASSES}, {INPUT_HW[0]}, {INPUT_HW[1]})"
                )
            out[out_i] = mask_t.to(torch.uint8)
    return out


def _to_class_index(mask_fg: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, C_fg, H, W] uint8`` -> ``[B, H, W] int64`` class labels.

    Class indexing matches CLASS_NAMES: 0=bg, 1=NCR/RC, 2=ED, 3=ET.
    Pixel-level overlap is impossible by construction (BraTS raw labels
    are mutually exclusive integers per voxel and nearest-neighbour
    resize on per-channel bool masks does not introduce overlap when
    fed mutually-exclusive sources at the same H,W). We assert that
    invariant explicitly so future transform or cache changes that
    accidentally introduce multi-hot pixels fail loudly.
    """
    sum_per_pixel = mask_fg.to(torch.int64).sum(dim=1)  # [B, H, W]
    if torch.any(sum_per_pixel > 1):
        n_overlap = int((sum_per_pixel > 1).sum().item())
        raise RuntimeError(
            f"[m1] multiclass mask has {n_overlap} multi-hot pixels; "
            "BraTS labels must be mutually exclusive per voxel"
        )
    bg = (sum_per_pixel == 0).to(torch.int64)
    cls_fg = mask_fg.to(torch.int64).argmax(dim=1) + 1  # [B, H, W]
    return torch.where(bg.bool(), torch.zeros_like(cls_fg), cls_fg)


# ---------------------------------------------------------------------------
# Per-case multiclass Dice.
# ---------------------------------------------------------------------------
def _per_case_per_class_dice(
    pred_cls: torch.Tensor, gt_cls: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute per-case per-class Dice over foreground classes only.

    pred_cls, gt_cls: int64 ``[B, H, W]`` with values in ``[0, num_classes)``.
    Returns ``[B, num_classes - 1]`` float32 (background channel dropped).

    Convention (matches ``train_head_cached.py``): if both prediction
    and ground truth are empty for a class, Dice = 1.0. If only one is
    non-empty, Dice = 0.0 by the standard 2|A∩B|/(|A|+|B|) formula.
    """
    B = pred_cls.shape[0]
    out = torch.zeros((B, num_classes - 1), dtype=torch.float32,
                      device=pred_cls.device)
    for c in range(1, num_classes):
        p = (pred_cls == c).flatten(1).to(torch.float32)
        g = (gt_cls == c).flatten(1).to(torch.float32)
        inter = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        dice = torch.where(
            denom > 0, 2.0 * inter / denom, torch.ones_like(inter)
        )
        out[:, c - 1] = dice
    return out


# ---------------------------------------------------------------------------
# Train / eval loops.
# ---------------------------------------------------------------------------
def _train_head(
    head: LinearSegHead,
    feats: torch.Tensor,
    masks_fg_u8: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    use_amp: bool,
    device: torch.device,
) -> None:
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    n = feats.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    autocast_enabled = use_amp and (device.type == "cuda")
    for epoch in range(epochs):
        idx = torch.randperm(n, generator=gen)
        running = torch.zeros((), device=device, dtype=torch.float32)
        n_batches = 0
        for start in range(0, n, batch_size):
            end = start + batch_size
            chunk = idx[start:end]
            f_b = feats[chunk].to(device, non_blocking=True)
            m_fg_b = masks_fg_u8[chunk].to(device, non_blocking=True)
            target_cls = _to_class_index(m_fg_b)  # [B, H, W] int64
            opt.zero_grad(set_to_none=True)
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = head(f_b)  # [B, NUM_CLASSES, H, W]
                    loss = ce(logits, target_cls)
            else:
                logits = head(f_b.to(torch.float32))
                loss = ce(logits, target_cls)
            loss.backward()
            opt.step()
            running += loss.detach().to(torch.float32)
            n_batches += 1
        avg = float(running.item()) / max(n_batches, 1)
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info("epoch %d/%d ce_loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate_head(
    head: LinearSegHead,
    feats: torch.Tensor,
    masks_fg_u8: torch.Tensor,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return ``[N_test, NUM_CLASSES - 1]`` per-case per-class Dice."""
    head.eval()
    n = feats.shape[0]
    chunks: list[torch.Tensor] = []
    autocast_enabled = use_amp and (device.type == "cuda")
    for start in range(0, n, batch_size):
        end = start + batch_size
        f_b = feats[start:end].to(device, non_blocking=True)
        m_fg_b = masks_fg_u8[start:end].to(device, non_blocking=True)
        gt_cls = _to_class_index(m_fg_b)
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(f_b)
        else:
            logits = head(f_b.to(torch.float32))
        pred_cls = logits.float().argmax(dim=1)  # [B, H, W]
        chunks.append(
            _per_case_per_class_dice(pred_cls, gt_cls, NUM_CLASSES).cpu()
        )
    if not chunks:
        return torch.zeros((0, NUM_CLASSES - 1), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.task != "brats_multiclass":
        logger.error(
            "train_head_multiclass.py only supports task=brats_multiclass; "
            "got %r", args.task,
        )
        return 4

    # Trainer-level resume skip: the worker also skips, but a direct CLI
    # re-run of the trainer would otherwise overwrite an existing JSON after
    # re-doing all the expensive feature loading.
    out_path_early = Path(args.out) / f"seed_{args.seed}.json"
    if out_path_early.is_file():
        logger.info(
            "skip: %s already exists; trainer-level resume short-circuit",
            out_path_early,
        )
        return 0

    determinism.set_seed(args.seed)

    cache_root = Path(
        args.cache_root
        or Path(__file__).resolve().parents[1] / "results" / "feature_cache"
    )
    cache_dir = cache_root / args.cache_task / args.fm
    try:
        manifest, train_blob, test_blob = _load_cache(cache_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("cache invalid: %s", exc)
        return 3

    cache_task = manifest.get("task")
    if cache_task not in COMPATIBLE_CACHE_TASKS or manifest.get("fm") != args.fm:
        logger.error(
            "manifest mismatch: file says (%s,%s); "
            "expected task in %s and fm=%s",
            cache_task, manifest.get("fm"),
            sorted(COMPATIBLE_CACHE_TASKS), args.fm,
        )
        return 4

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    feature_dim = int(manifest["feature_dim"])
    feat_h, feat_w = manifest["feature_hw"]

    pin = (device.type == "cuda")
    train_feats = train_blob["feats"].pin_memory() if pin else train_blob["feats"]
    test_feats = test_blob["feats"].pin_memory() if pin else test_blob["feats"]
    train_ids: list[str] = list(train_blob["case_ids"])
    test_ids: list[str] = list(test_blob["case_ids"])

    # Verify both train and test tensors against the manifest's (D, h, w),
    # check feats/case_ids length parity, and reject duplicate case_ids.
    expected_dhw = (feature_dim, int(feat_h), int(feat_w))
    for split_name, feats_t, ids_list in (
        ("train", train_feats, train_ids),
        ("test", test_feats, test_ids),
    ):
        got_dhw = (
            int(feats_t.shape[1]),
            int(feats_t.shape[2]),
            int(feats_t.shape[3]),
        )
        if got_dhw != expected_dhw:
            logger.error(
                "%s feature shape mismatch with manifest: tensor DHW=%s "
                "vs manifest DHW=%s",
                split_name, got_dhw, expected_dhw,
            )
            return 4
        if int(feats_t.shape[0]) != len(ids_list):
            logger.error(
                "%s feats[0]=%d != len(case_ids)=%d",
                split_name, int(feats_t.shape[0]), len(ids_list),
            )
            return 4
        if len(set(ids_list)) != len(ids_list):
            logger.error(
                "%s case_ids contains duplicates (n=%d unique=%d)",
                split_name, len(ids_list), len(set(ids_list)),
            )
            return 4

    # Rebuild multiclass masks from disk (the cached masks are binary).
    logger.info(
        "rebuilding multiclass masks for %d train / %d test cases",
        len(train_ids), len(test_ids),
    )
    train_masks_fg = _build_multiclass_mask_tensor(train_ids, split="train")
    test_masks_fg = _build_multiclass_mask_tensor(test_ids, split="test")
    if pin:
        train_masks_fg = train_masks_fg.pin_memory()
        test_masks_fg = test_masks_fg.pin_memory()

    head = LinearSegHead(
        in_dim=feature_dim, num_classes=NUM_CLASSES, out_size=INPUT_HW
    ).to(device)

    use_amp = (not args.no_amp) and (device.type == "cuda")

    t0 = time.time()
    _train_head(
        head=head,
        feats=train_feats,
        masks_fg_u8=train_masks_fg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        use_amp=use_amp,
        device=device,
    )
    train_elapsed = time.time() - t0

    per_case_per_class = _evaluate_head(
        head=head,
        feats=test_feats,
        masks_fg_u8=test_masks_fg,
        batch_size=args.batch_size,
        use_amp=use_amp,
        device=device,
    )

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(__file__).resolve().parents[1]
        / "results" / "checkpoints_m1" / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = _sha256_file(ckpt_path)

    manifest_path = cache_dir / "manifest.json"
    cache_manifest_sha = _sha256_file(manifest_path)

    # Aggregations.
    if per_case_per_class.numel():
        per_case_dice = per_case_per_class.mean(dim=1)  # [N_test]
        per_class_dice = per_case_per_class.mean(dim=0)  # [C_fg]
        mean_dice = float(per_case_dice.mean().item())
    else:
        per_case_dice = torch.zeros((0,), dtype=torch.float32)
        per_class_dice = torch.zeros(
            (NUM_CLASSES - 1,), dtype=torch.float32
        )
        mean_dice = float("nan")

    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    payload = {
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "n_test": int(per_case_per_class.shape[0]),
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "fg_class_names": [CLASS_NAMES[i] for i in FG_CLASS_INDICES],
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case_dice.tolist()],
        "per_case_per_class_dice": [
            [float(x) for x in row]
            for row in per_case_per_class.tolist()
        ],
        "per_class_dice": [float(x) for x in per_class_dice.tolist()],
        "mean_dice": mean_dice,
        "cache_task": cache_task,
        "cache_manifest_sha256": cache_manifest_sha,
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": bool(use_amp),
        "schema_version": SCHEMA_VERSION,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s n_test=%d mean_dice=%.4f per_class=%s train_s=%.1f",
        out_path, payload["n_test"], payload["mean_dice"],
        payload["per_class_dice"], train_elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
