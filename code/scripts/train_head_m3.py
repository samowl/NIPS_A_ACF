#!/usr/bin/env python
"""M3: train a configurable decoder head on cached features (one seed/call).

This is a parallel of ``train_head_cached.py`` that replaces the hardcoded
``LinearSegHead`` with the head built by ``fmpool.heads.build_head``. The
feature cache layout is identical (schema_version ``feat_cache_v1``) so the
PASS-A artifacts are reused without re-extraction.

The output schema extends ``train_head_cached.py`` with two extra fields:

    head_design: str             # one of fmpool.heads.HEAD_DESIGNS
    n_trainable_params: int      # parameters in the head only

Output path is::

    results/per_case_dice_m3/{head_design}/{task}/{fm}/seed_{seed}.json

Determinism: ``fmpool.determinism.set_seed`` before head init, deterministic
shuffle via per-seed generator. Training schedule matches SPEC §4
(Adam lr=1e-3, BCE-with-logits, 30 epochs, bf16 autocast on CUDA).

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
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from fmpool import determinism
from fmpool.heads import HEAD_DESIGNS, build_head, count_trainable_params

logger = logging.getLogger("fmpool.train_head_m3")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
SCHEMA_VERSION: str = "feat_cache_v1"
M3_SCHEMA_VERSION: str = "m3_head_sweep_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 head trainer (cached features)")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--head-design",
        required=True,
        choices=list(HEAD_DESIGNS),
        help="Decoder head design.",
    )
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory; seed_{seed}.json is written here.")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable bf16 autocast on head training.")
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


def _load_cache(cache_dir: Path) -> tuple[dict, dict, dict]:
    manifest_path = cache_dir / "manifest.json"
    train_path = cache_dir / "train.pt"
    test_path = cache_dir / "test.pt"
    if not manifest_path.is_file() or not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"feature cache incomplete in {cache_dir}: "
            f"manifest={manifest_path.is_file()} "
            f"train={train_path.is_file()} test={test_path.is_file()}"
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version={manifest.get('schema_version')!r} "
            f"!= expected {SCHEMA_VERSION!r}"
        )
    train_blob = torch.load(train_path, map_location="cpu", weights_only=False)
    test_blob = torch.load(test_path, map_location="cpu", weights_only=False)
    return manifest, train_blob, test_blob


def _per_case_dice_on_device(
    preds: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    p = preds.flatten(1).to(torch.float32)
    m = masks.flatten(1).to(torch.float32)
    inter = (p * m).sum(dim=1)
    denom = p.sum(dim=1) + m.sum(dim=1)
    return torch.where(denom > 0, 2.0 * inter / denom, torch.ones_like(inter))


def _train_head(
    head: nn.Module,
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
    head: nn.Module,
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
        case_dices.append(_per_case_dice_on_device(preds, m_b).cpu())
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
        or Path(__file__).resolve().parents[1] / "results" / "feature_cache"
    )
    cache_dir = cache_root / args.task / args.fm
    try:
        manifest, train_blob, test_blob = _load_cache(cache_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("cache invalid: %s", exc)
        return 3

    if manifest.get("task") != args.task or manifest.get("fm") != args.fm:
        logger.error(
            "manifest mismatch: file says (%s,%s) but called with (%s,%s)",
            manifest.get("task"), manifest.get("fm"), args.task, args.fm,
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
    train_masks = train_blob["masks"].pin_memory() if pin else train_blob["masks"]
    test_feats = test_blob["feats"].pin_memory() if pin else test_blob["feats"]
    test_masks = test_blob["masks"].pin_memory() if pin else test_blob["masks"]
    test_ids = list(test_blob["case_ids"])

    if train_feats.shape[1] != feature_dim or train_feats.shape[2] != feat_h:
        logger.error(
            "feature shape mismatch with manifest: tensor %s vs manifest D=%d hw=%dx%d",
            tuple(train_feats.shape), feature_dim, feat_h, feat_w,
        )
        return 4

    head = build_head(
        args.head_design,
        in_dim=feature_dim,
        out_channels=1,
        out_size=INPUT_HW,
    ).to(device)
    n_params = count_trainable_params(head)
    logger.info(
        "head=%s in_dim=%d n_trainable_params=%d",
        args.head_design, feature_dim, n_params,
    )

    use_amp = (not args.no_amp) and (device.type == "cuda")

    t0 = time.time()
    _train_head(
        head=head,
        feats=train_feats,
        masks=train_masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        use_amp=use_amp,
        device=device,
    )
    train_elapsed = time.time() - t0

    per_case = _evaluate_head(
        head=head,
        feats=test_feats,
        masks=test_masks,
        batch_size=args.batch_size,
        use_amp=use_amp,
        device=device,
    )

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(__file__).resolve().parents[1]
        / "results" / "checkpoints_m3" / args.head_design / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = _sha256_file(ckpt_path)

    manifest_path = cache_dir / "manifest.json"
    cache_manifest_sha = _sha256_file(manifest_path)

    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    payload = {
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "head_design": str(args.head_design),
        "n_trainable_params": int(n_params),
        "n_test": int(per_case.numel()),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(per_case.mean().item()) if per_case.numel() else float("nan"),
        "cache_manifest_sha256": cache_manifest_sha,
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": bool(use_amp),
        "schema_version": M3_SCHEMA_VERSION,
        "feat_cache_schema_version": SCHEMA_VERSION,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s head=%s n_test=%d mean_dice=%.4f train_s=%.1f",
        out_path, args.head_design, payload["n_test"],
        payload["mean_dice"], train_elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
