#!/usr/bin/env python
"""M15: fold-reslicing head trainer.

Reuses the existing PASS-A feature cache produced by ``extract_features.py``
(``results/feature_cache/{task}/{fm}/{train,test}.pt``). The cache pools all
case_ids in the dataset (train + test = full patient pool); this trainer
re-slices the pool by 5-fold CV membership at PASS-B, so no re-extraction is
needed.

Fold membership comes from:
  - ACDC: ``data/splits/acdc_lv_split.json`` -> ``folds["fold_{k}_{train,val}"]``
  - RIGA cup: ``data/splits/riga_cup_5fold.json`` -> ``folds["fold_{k}_{train,val}"]``

Optimisations and config match ``train_head_cached.py`` (Adam lr=1e-3, BCE,
30 epochs, bf16 autocast, 1x1 ``LinearSegHead``, on-device per-case Dice).

JSON output: ``results/per_case_dice_m15/{task}/{fm}/fold_{k}_seed_{seed}.json``

Exit codes
----------
0 success, 3 cache or split missing, 4 manifest/case-id mismatch.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmpool import determinism
from fmpool.decoder import LinearSegHead

logger = logging.getLogger("fmpool.train_head_5fold")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
SCHEMA_VERSION: str = "m15_5fold_v1"
CACHE_SCHEMA_VERSION: str = "feat_cache_v1"
EXPECTED_INPUT_HW: tuple[int, int] = (224, 224)
N_FOLDS: int = 5

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M15 5-fold head trainer (cached features)")
    p.add_argument("--task", required=True, choices=["acdc_lv", "riga_cup"])
    p.add_argument("--fm", required=True)
    p.add_argument("--fold", type=int, required=True, choices=list(range(N_FOLDS)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--no-amp", action="store_true")
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


def _load_fold_split(task: str, fold: int) -> tuple[list[str], list[str]]:
    """Return (train_ids, val_ids) for the requested task and fold."""
    if task == "acdc_lv":
        path = SPLITS_DIR / "acdc_lv_split.json"
    elif task == "riga_cup":
        path = SPLITS_DIR / "riga_cup_5fold.json"
    else:
        raise ValueError(f"unsupported task for 5-fold: {task!r}")
    if not path.is_file():
        raise FileNotFoundError(
            f"5-fold split missing for task={task!r}: {path}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds = data.get("folds", {})
    train_key = f"fold_{fold}_train"
    val_key = f"fold_{fold}_val"
    if train_key not in folds or val_key not in folds:
        raise KeyError(
            f"split file {path} missing {train_key!r} or {val_key!r}; "
            f"available={sorted(folds.keys())}"
        )
    return list(folds[train_key]), list(folds[val_key])


def _load_unified_cache(
    cache_dir: Path, task: str, fm: str
) -> tuple[dict, torch.Tensor, torch.Tensor, list[str]]:
    """Load PASS-A train+test caches and concatenate into a unified pool.

    Returns (manifest, feats[N,D,h,w] fp16, masks[N,H,W] uint8, case_ids[N]).
    Duplicates raise ValueError because feature cache schema guarantees
    disjoint train/test ids.
    """
    manifest_path = cache_dir / "manifest.json"
    train_path = cache_dir / "train.pt"
    test_path = cache_dir / "test.pt"
    for p in (manifest_path, train_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(f"cache file missing: {p}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("task") != task or manifest.get("fm") != fm:
        raise ValueError(
            f"manifest mismatch in {manifest_path}: task/fm "
            f"({manifest.get('task')},{manifest.get('fm')}) != requested ({task},{fm})"
        )
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"cache schema_version={manifest.get('schema_version')!r} != "
            f"expected {CACHE_SCHEMA_VERSION!r} (regenerate via extract_features.py)"
        )
    cache_input_hw = tuple(manifest.get("input_hw", []))
    if cache_input_hw != EXPECTED_INPUT_HW:
        raise ValueError(
            f"cache input_hw={cache_input_hw} != expected {EXPECTED_INPUT_HW}; "
            "M15 requires 224x224 PASS-A cache"
        )
    train_blob = torch.load(train_path, map_location="cpu", weights_only=False)
    test_blob = torch.load(test_path, map_location="cpu", weights_only=False)
    feats = torch.cat([train_blob["feats"], test_blob["feats"]], dim=0)
    masks = torch.cat([train_blob["masks"], test_blob["masks"]], dim=0)
    case_ids: list[str] = list(train_blob["case_ids"]) + list(test_blob["case_ids"])
    if len(case_ids) != feats.shape[0]:
        raise ValueError(
            f"case_id count {len(case_ids)} != feats batch {feats.shape[0]}"
        )
    if len(set(case_ids)) != len(case_ids):
        seen: dict[str, int] = {}
        dups: list[str] = []
        for cid in case_ids:
            seen[cid] = seen.get(cid, 0) + 1
        for cid, c in seen.items():
            if c > 1:
                dups.append(cid)
                if len(dups) >= 5:
                    break
        raise ValueError(
            f"feature cache has duplicate case_ids (sample: {dups}); "
            "5-fold slicing requires disjoint train/test ids."
        )
    return manifest, feats, masks, case_ids


def _resolve_indices(
    case_ids: list[str], wanted: list[str], *, fold: int, role: str
) -> torch.Tensor:
    pos: dict[str, int] = {cid: i for i, cid in enumerate(case_ids)}
    missing = [w for w in wanted if w not in pos]
    if missing:
        raise KeyError(
            f"{len(missing)} {role} ids for fold={fold} not in feature cache "
            f"(first missing: {missing[:5]})"
        )
    return torch.tensor([pos[w] for w in wanted], dtype=torch.long)


def _per_case_dice(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    p = preds.flatten(1).to(torch.float32)
    m = masks.flatten(1).to(torch.float32)
    inter = (p * m).sum(dim=1)
    denom = p.sum(dim=1) + m.sum(dim=1)
    return torch.where(denom > 0, 2.0 * inter / denom, torch.ones_like(inter))


def _train_head(
    head: LinearSegHead,
    feats: torch.Tensor,
    masks: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    use_amp: bool,
    device: torch.device,
) -> None:
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
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
            m_b = masks[chunk].to(device, non_blocking=True).to(torch.float32).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = head(f_b)
                    loss = bce(logits, m_b)
            else:
                logits = head(f_b.to(torch.float32))
                loss = bce(logits, m_b)
            loss.backward()
            opt.step()
            running += loss.detach().to(torch.float32)
            n_batches += 1
        avg = float(running.item()) / max(n_batches, 1)
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate_head(
    head: LinearSegHead,
    feats: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    head.eval()
    n = feats.shape[0]
    case_dices: list[torch.Tensor] = []
    autocast_enabled = use_amp and (device.type == "cuda")
    for start in range(0, n, batch_size):
        end = start + batch_size
        f_b = feats[start:end].to(device, non_blocking=True)
        m_b = masks[start:end].to(device, non_blocking=True).to(torch.bool)
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(f_b)
        else:
            logits = head(f_b.to(torch.float32))
        preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1)
        case_dices.append(_per_case_dice(preds, m_b).cpu())
    if not case_dices:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(case_dices, dim=0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    determinism.set_seed(args.seed)

    cache_root = Path(
        args.cache_root
        or REPO_ROOT / "results" / "feature_cache"
    )
    cache_dir = cache_root / args.task / args.fm

    try:
        manifest, feats, masks, case_ids = _load_unified_cache(
            cache_dir, args.task, args.fm
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("cache invalid: %s", exc)
        return 3

    try:
        train_ids, val_ids = _load_fold_split(args.task, args.fold)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("fold split invalid: %s", exc)
        return 3

    overlap = set(train_ids) & set(val_ids)
    if overlap:
        logger.error(
            "fold=%d has %d ids in both train and val (leak); sample=%s",
            args.fold, len(overlap), list(overlap)[:5],
        )
        return 4

    try:
        train_idx = _resolve_indices(case_ids, train_ids, fold=args.fold, role="train")
        val_idx = _resolve_indices(case_ids, val_ids, fold=args.fold, role="val")
    except KeyError as exc:
        logger.error("fold case-ids vs cache mismatch: %s", exc)
        return 4

    feature_dim = int(manifest["feature_dim"])
    feat_h, feat_w = manifest["feature_hw"]
    if (feats.shape[1] != feature_dim
            or feats.shape[2] != feat_h
            or feats.shape[3] != feat_w):
        logger.error(
            "feature shape mismatch: tensor %s vs manifest D=%d hw=%dx%d",
            tuple(feats.shape), feature_dim, feat_h, feat_w,
        )
        return 4

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pin = (device.type == "cuda")

    train_feats = feats.index_select(0, train_idx).contiguous()
    train_masks = masks.index_select(0, train_idx).contiguous()
    val_feats = feats.index_select(0, val_idx).contiguous()
    val_masks = masks.index_select(0, val_idx).contiguous()
    if pin:
        train_feats = train_feats.pin_memory()
        train_masks = train_masks.pin_memory()
        val_feats = val_feats.pin_memory()
        val_masks = val_masks.pin_memory()
    val_case_ids = [case_ids[i] for i in val_idx.tolist()]

    head = LinearSegHead(
        in_dim=feature_dim, num_classes=1, out_size=INPUT_HW
    ).to(device)
    use_amp = (not args.no_amp) and (device.type == "cuda")

    t0 = time.time()
    _train_head(
        head=head, feats=train_feats, masks=train_masks,
        epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, use_amp=use_amp, device=device,
    )
    train_elapsed = time.time() - t0

    per_case = _evaluate_head(
        head=head, feats=val_feats, masks=val_masks,
        batch_size=args.batch_size, use_amp=use_amp, device=device,
    )

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else REPO_ROOT / "results" / "checkpoints_m15" / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"fold_{args.fold}_seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = _sha256_file(ckpt_path)

    cache_manifest_sha = _sha256_file(cache_dir / "manifest.json")

    out_dir = Path(args.out)
    out_path = out_dir / f"fold_{args.fold}_seed_{args.seed}.json"
    payload = {
        "task": args.task,
        "fm": args.fm,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "n_train": int(train_idx.numel()),
        "n_val": int(per_case.numel()),
        "val_ids": list(val_case_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(per_case.mean().item()) if per_case.numel() else float("nan"),
        "cache_manifest_sha256": cache_manifest_sha,
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": bool(use_amp),
        "schema_version": SCHEMA_VERSION,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s n_train=%d n_val=%d mean_dice=%.4f train_s=%.1f",
        out_path, payload["n_train"], payload["n_val"],
        payload["mean_dice"], train_elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
