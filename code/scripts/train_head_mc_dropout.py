#!/usr/bin/env python
"""M22: PASS-B head trainer with MC dropout for predictive uncertainty.

Mirrors :mod:`scripts.train_head_cached` but inserts ``Dropout2d`` before the
1x1 projection of :class:`fmpool.decoder.LinearSegHead`. After training, the
head is evaluated 10 times in stochastic mode (``model.train()`` to keep
dropout active) on the full test split. Per-case Dice mean and std across
MC samples are recorded alongside the standard schema.

Designed for M22: 1 FM (dinov2_vitb14) x 1 task (riga_cup) x 4 seeds x
3 dropout rates {0.1, 0.2, 0.4} = 12 cells.

Output schema additions over ``train_head_cached``:
    - ``mc_n_samples`` (int): number of stochastic forward passes
    - ``dropout_rate`` (float): the active rate
    - ``mc_per_case_dice_mean`` (list[float], length n_test)
    - ``mc_per_case_dice_std``  (list[float], length n_test)
    - ``mc_mean_dice`` (float): mean over cases of per-case mean Dice
    - ``mc_std_dice``  (float): mean over cases of per-case std Dice
    - ``mc_predictive_entropy`` (float): mean of pixel-wise H(mean_prob)
    - ``deterministic_mean_dice`` (float): single eval-mode pass for parity

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
from fmpool.decoder import LinearSegHead

logger = logging.getLogger("fmpool.train_head_mc_dropout")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
MC_SAMPLES_DEFAULT: int = 10
SCHEMA_VERSION: str = "feat_cache_v1"
M22_SCHEMA_TAG: str = "mc_dropout_v1"
EPS: float = 1e-12


class LinearSegHeadDropout(nn.Module):
    """LinearSegHead with Dropout2d before the 1x1 projection.

    Composition (rather than subclassing) keeps the original head intact.
    Forward signature matches ``LinearSegHead.forward``.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        out_size: tuple[int, int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )
        self.dropout = nn.Dropout2d(p=float(dropout_rate))
        self.head = LinearSegHead(
            in_dim=in_dim, num_classes=num_classes, out_size=out_size
        )
        self.dropout_rate = float(dropout_rate)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(spatial))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M22 MC-dropout PASS-B head trainer (cached features)"
    )
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--dropout-rate",
        type=float,
        required=True,
        help="Dropout2d rate (e.g. 0.1, 0.2, 0.4)",
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=MC_SAMPLES_DEFAULT,
        help="Number of stochastic forward passes at inference",
    )
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable bf16 autocast on head training",
    )
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
    """Per-case Dice on-device. preds, masks bool [B, H, W]."""
    p = preds.flatten(1).to(torch.float32)
    m = masks.flatten(1).to(torch.float32)
    inter = (p * m).sum(dim=1)
    denom = p.sum(dim=1) + m.sum(dim=1)
    return torch.where(
        denom > 0, 2.0 * inter / denom, torch.ones_like(inter)
    )


def _train_head(
    head: LinearSegHeadDropout,
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
            m_b = masks[chunk].to(device, non_blocking=True).to(
                torch.float32
            ).unsqueeze(1)
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
def _evaluate_deterministic(
    head: LinearSegHeadDropout,
    feats: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    """Eval-mode (dropout disabled) per-case Dice for parity with non-MC runs."""
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


@torch.no_grad()
def _evaluate_mc(
    head: LinearSegHeadDropout,
    feats: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    mc_samples: int,
    seed: int,
    use_amp: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MC dropout evaluation (head.train() keeps Dropout2d active, no_grad).

    Returns
    -------
    per_case_mean : [n_test] mean Dice across MC samples
    per_case_std  : [n_test] std  Dice across MC samples (population)
    mean_pred_entropy : scalar mean pixel-wise binary entropy of mean prob
    """
    head.train()  # keep Dropout2d active
    n = feats.shape[0]
    if n == 0 or mc_samples <= 0:
        return (
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((), dtype=torch.float32),
        )
    autocast_enabled = use_amp and (device.type == "cuda")
    dice_samples = torch.empty((mc_samples, n), dtype=torch.float32)
    base_seed = int(seed) * 1_000_003

    # Pass 1: per-MC-sample per-case Dice.
    for s in range(mc_samples):
        if device.type == "cuda":
            torch.cuda.manual_seed_all(base_seed + s)
        else:
            torch.manual_seed(base_seed + s)
        case_dices: list[torch.Tensor] = []
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
        dice_samples[s] = torch.cat(case_dices, dim=0)

    # Pass 2: pixel-wise entropy of mean prob over MC samples.
    entropy_sum = torch.zeros((), device=device, dtype=torch.float64)
    pixel_count = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, n, batch_size):
        end = start + batch_size
        f_b = feats[start:end].to(device, non_blocking=True)
        acc = torch.zeros(
            (f_b.shape[0], INPUT_HW[0], INPUT_HW[1]),
            device=device,
            dtype=torch.float32,
        )
        for s in range(mc_samples):
            if device.type == "cuda":
                torch.cuda.manual_seed_all(base_seed + s + 7)
            else:
                torch.manual_seed(base_seed + s + 7)
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = head(f_b)
            else:
                logits = head(f_b.to(torch.float32))
            acc += torch.sigmoid(logits.float()).squeeze(1)
        mean_prob = acc / float(mc_samples)
        p = mean_prob.clamp(EPS, 1.0 - EPS)
        h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        entropy_sum += h.sum().to(torch.float64)
        pixel_count += float(h.numel())

    per_case_mean = dice_samples.mean(dim=0)
    per_case_std = dice_samples.std(dim=0, unbiased=False)
    mean_entropy = (
        (entropy_sum / pixel_count).to(torch.float32).cpu()
        if float(pixel_count.item()) > 0
        else torch.zeros((), dtype=torch.float32)
    )
    return per_case_mean, per_case_std, mean_entropy


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

    head = LinearSegHeadDropout(
        in_dim=feature_dim,
        num_classes=1,
        out_size=INPUT_HW,
        dropout_rate=args.dropout_rate,
    ).to(device)
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

    det_per_case = _evaluate_deterministic(
        head=head,
        feats=test_feats,
        masks=test_masks,
        batch_size=args.batch_size,
        use_amp=use_amp,
        device=device,
    )
    mc_mean_per_case, mc_std_per_case, mc_pred_entropy = _evaluate_mc(
        head=head,
        feats=test_feats,
        masks=test_masks,
        batch_size=args.batch_size,
        mc_samples=int(args.mc_samples),
        seed=int(args.seed),
        use_amp=use_amp,
        device=device,
    )

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(__file__).resolve().parents[1]
        / "results" / "checkpoints_m22" / f"p{args.dropout_rate}"
        / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = _sha256_file(ckpt_path)

    manifest_path = cache_dir / "manifest.json"
    cache_manifest_sha = _sha256_file(manifest_path)

    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    n_test = int(det_per_case.numel())
    # Schema parity guard for downstream aggregators.
    if not (
        len(test_ids) == n_test
        == int(mc_mean_per_case.numel())
        == int(mc_std_per_case.numel())
    ):
        logger.error(
            "schema parity mismatch: ids=%d det=%d mc_mean=%d mc_std=%d",
            len(test_ids),
            n_test,
            int(mc_mean_per_case.numel()),
            int(mc_std_per_case.numel()),
        )
        return 4
    payload = {
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "dropout_rate": float(args.dropout_rate),
        "mc_n_samples": int(args.mc_samples),
        "n_test": n_test,
        "test_ids": list(test_ids),
        # Downstream aggregators require mean_dice == mean(per_case_dice).
        # For M22 the canonical per-case Dice is the MC mean across samples;
        # the deterministic eval-mode pass is preserved separately.
        "per_case_dice": [float(x) for x in mc_mean_per_case.tolist()],
        "mean_dice": float(mc_mean_per_case.mean().item())
        if mc_mean_per_case.numel()
        else float("nan"),
        "deterministic_per_case_dice": [float(x) for x in det_per_case.tolist()],
        "deterministic_mean_dice": float(det_per_case.mean().item())
        if det_per_case.numel()
        else float("nan"),
        "mc_per_case_dice_mean": [float(x) for x in mc_mean_per_case.tolist()],
        "mc_per_case_dice_std": [float(x) for x in mc_std_per_case.tolist()],
        "mc_mean_dice": float(mc_mean_per_case.mean().item())
        if mc_mean_per_case.numel()
        else float("nan"),
        "mc_std_dice": float(mc_std_per_case.mean().item())
        if mc_std_per_case.numel()
        else float("nan"),
        "mc_predictive_entropy": float(mc_pred_entropy.item()),
        "cache_manifest_sha256": cache_manifest_sha,
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": bool(use_amp),
        "schema_version": SCHEMA_VERSION,
        "m22_schema_tag": M22_SCHEMA_TAG,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s n_test=%d det_mean=%.4f mc_mean=%.4f mc_std=%.4f H=%.4f",
        out_path,
        n_test,
        payload["deterministic_mean_dice"],
        payload["mc_mean_dice"],
        payload["mc_std_dice"],
        payload["mc_predictive_entropy"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
