#!/usr/bin/env python
"""Train an adapted FM encoder + 1x1 conv head (paper §8 adaptation factorial).

Three regimes:

* ``lora``      — Encoder frozen; LoRA modules injected on Q/K/V (and MLP
                  optionally) for the trailing ``--lora-blocks`` transformer
                  blocks at rank ``--lora-rank``. The 1x1 conv head is trained.
* ``full_ft``   — All encoder parameters unfrozen; same Adam BCE loop. 1x1
                  conv head trained from scratch.
* ``rand_init`` — Encoder rebuilt from scratch (no pretrained weights);
                  decoder + encoder trained jointly.

Per-case Dice JSON schema is a superset of :mod:`scripts.train_frozen_fm`:
extra fields ``regime``, ``lora_rank``, ``lora_blocks``, ``epochs``,
``n_trainable_params``.

Output path::

    {out_root}/{regime}/{config}/{task}/{fm}/seed_{seed}.json

with ``{config}`` ∈ ``lora_r{R}_b{B}`` / ``ft_e{E}`` / ``rand_init``.

Verified LoRA target modules per FM (Hu et al. 2021, peft convention)::

    DINOv2 / BiomedCLIP : ``blocks.{i}.attn.qkv`` (fused Linear, 3D out)
    CLIP ViT-L/14       : ``transformer.resblocks.{i}.attn.in_proj_weight``
                          (nn.MultiheadAttention; in_proj_weight is [3D, D])
    Torchvision CNNs    : LoRA not applicable (no ViT blocks).

Determinism: ``set_seed`` runs before any random init. bf16 autocast wraps
encoder, head, and BCE; bf16 needs no GradScaler.
"""
from __future__ import annotations

import argparse
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

logger = logging.getLogger("fmpool.train_adapted")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
VIT_FMS: tuple[str, ...] = (
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "clip_vitl14",
)


# ---------------------------------------------------------------------------
# LoRA modules
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear`` with a low-rank trainable delta.

    ``y = base(x) + (alpha/r) * (x A^T) B^T`` (Hu et al. 2021).
    Base linear is frozen; A is kaiming-uniform, B is zero-init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int | None = None) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out + delta


class _MHALoRAHook(nn.Module):
    """Trainable LoRA delta applied to ``MultiheadAttention.in_proj_weight``.

    open_clip CLIP ViT-L/14 uses ``nn.MultiheadAttention`` whose Q,K,V
    projection is fused as ``in_proj_weight`` of shape ``[3*E, E]``. We
    monkey-patch the module's ``forward`` to add a low-rank delta to the
    QKV linear and route through ``scaled_dot_product_attention``.
    """

    def __init__(self, mha: nn.MultiheadAttention, rank: int, alpha: int | None = None):
        super().__init__()
        if mha.in_proj_weight is None:
            raise NotImplementedError(
                "LoRA on MHA assumes a fused in_proj_weight; got separated q/k/v."
            )
        self.mha = mha
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        embed = mha.embed_dim
        self.lora_A = nn.Parameter(torch.empty(self.rank, embed))
        self.lora_B = nn.Parameter(torch.zeros(3 * embed, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta_in_proj(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def _wrap_mha_with_lora(mha: nn.MultiheadAttention, rank: int) -> _MHALoRAHook:
    """Inject a LoRA hook into ``mha`` and freeze its original parameters."""
    hook = _MHALoRAHook(mha, rank)
    for p in mha.parameters():
        p.requires_grad = False
    orig_forward = mha.forward

    def _patched_forward(query, key, value, *args, **kwargs):  # noqa: ANN001
        # Self-attention only (CLIP ViT). Otherwise fall back to original.
        if not (query is key and key is value):
            return orig_forward(query, key, value, *args, **kwargs)
        x = query
        x_seq = x if mha.batch_first else x.transpose(0, 1)  # [B, N, E]
        E = mha.embed_dim
        qkv = F.linear(x_seq, mha.in_proj_weight, mha.in_proj_bias) + hook.delta_in_proj(x_seq)
        q, k, v = qkv.split(E, dim=-1)
        B_, N_, _ = q.shape
        h = mha.num_heads
        d = E // h
        q = q.reshape(B_, N_, h, d).transpose(1, 2)
        k = k.reshape(B_, N_, h, d).transpose(1, 2)
        v = v.reshape(B_, N_, h, d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B_, N_, E)
        out = mha.out_proj(out)
        if not mha.batch_first:
            out = out.transpose(0, 1)
        return out, None

    mha.forward = _patched_forward  # type: ignore[assignment]
    return hook


def _last_block_indices(num_blocks: int, n_last: int) -> set[int]:
    if n_last <= 0 or num_blocks <= 0:
        return set()
    n = min(n_last, num_blocks)
    return set(range(num_blocks - n, num_blocks))


def _vit_blocks(encoder: nn.Module, fm: str) -> nn.Module:
    if fm.startswith("dinov2_"):
        return encoder.model.blocks
    if fm == "biomedclip":
        return encoder.trunk.blocks
    if fm == "clip_vitl14":
        return encoder.visual.transformer.resblocks
    raise NotImplementedError(f"LoRA not supported for FM {fm!r}")


def apply_lora(
    encoder: nn.Module, fm: str, rank: int, n_last_blocks: int
) -> list[nn.Parameter]:
    """Inject LoRA adapters into the trailing blocks of ``encoder``.

    Encoder is frozen first; only injected LoRA params keep ``requires_grad``.
    Returns the list of trainable LoRA parameters.
    """
    if fm not in VIT_FMS:
        raise NotImplementedError(
            f"LoRA only supported on ViT FMs {VIT_FMS}; got {fm!r}"
        )
    for p in encoder.parameters():
        p.requires_grad = False

    blocks = _vit_blocks(encoder, fm)
    target_idx = _last_block_indices(len(blocks), n_last_blocks)
    trainable: list[nn.Parameter] = []
    for i, block in enumerate(blocks):
        if i not in target_idx:
            continue
        if fm == "clip_vitl14":
            hook = _wrap_mha_with_lora(block.attn, rank)
            block.add_module("lora_qkv", hook)
            trainable.extend(p for p in hook.parameters() if p.requires_grad)
        else:
            attn = block.attn
            wrapped = LoRALinear(attn.qkv, rank=rank)
            attn.qkv = wrapped
            trainable.extend(p for p in wrapped.parameters() if p.requires_grad)
    return trainable


# ---------------------------------------------------------------------------
# Random-init encoders (regime ``rand_init``)
# ---------------------------------------------------------------------------


def build_random_encoder(fm: str) -> tuple[nn.Module, encoders.EncoderSpec]:
    """Rebuild the FM architecture WITHOUT pretrained weights."""
    spec = encoders.FM_SPECS[fm]
    if fm.startswith("dinov2_"):
        model = torch.hub.load("facebookresearch/dinov2", fm, pretrained=False)
        wrapper: nn.Module = encoders._DINOv2Wrapper(model, spec.feature_dim, spec.feature_hw)
    elif fm == "biomedclip":
        import open_clip

        model = open_clip.create_model(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            pretrained=None,
        )
        visual = model.visual
        trunk = visual.trunk if hasattr(visual, "trunk") else visual
        for m in trunk.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        wrapper = encoders._BiomedCLIPWrapper(trunk, spec.feature_dim, spec.feature_hw)
    elif fm == "clip_vitl14":
        import open_clip

        model = open_clip.create_model("ViT-L-14", pretrained=None)
        for m in model.visual.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        wrapper = encoders._CLIPViTLWrapper(model.visual, spec.feature_dim, spec.feature_hw)
    elif fm in ("convnext_tiny", "efficientnet_b0", "resnet50", "resnet18"):
        import torchvision.models as tvm

        ctor = {
            "convnext_tiny": tvm.convnext_tiny,
            "efficientnet_b0": tvm.efficientnet_b0,
            "resnet50": tvm.resnet50,
            "resnet18": tvm.resnet18,
        }[fm]
        model = ctor(weights=None)
        trunk = nn.Sequential(*list(model.children())[:-2])
        wrapper = encoders._CNNWrapper(trunk, spec.feature_dim, spec.feature_hw)
    else:
        raise KeyError(f"Unknown fm {fm!r}")
    return wrapper, spec


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from train_frozen_fm.py to avoid touching it)
# ---------------------------------------------------------------------------


def make_normaliser(
    spec: encoders.EncoderSpec,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec.norm_std, dtype=torch.float32).view(3, 1, 1)

    def _fn(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"normalise_fn expects [3,H,W], got {tuple(image.shape)}")
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


def _resize_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    resized = F.interpolate(
        mask.float().unsqueeze(0), size=INPUT_HW, mode="nearest"
    ).squeeze(0)
    return (resized > 0.5).float()


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor, str]]):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([_resize_mask(item[1]) for item in batch], dim=0)
    ids = [item[2] for item in batch]
    return imgs, masks, ids


def _seed_worker(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    import random as _r

    _r.seed(base + worker_id)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _train(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    encoder_train_mode: bool,
    use_bf16: bool,
) -> None:
    params = [p for p in encoder.parameters() if p.requires_grad]
    params += [p for p in decoder.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters in encoder + decoder")
    opt = torch.optim.Adam(params, lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    encoder.train() if encoder_train_mode else encoder.eval()
    decoder.train()

    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_enabled = use_bf16 and device.type == "cuda"

    for epoch in range(epochs):
        n_batches = 0
        running = 0.0
        for imgs, masks, _ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                feats = encoder(imgs)
                logits = decoder(feats)
                loss = loss_fn(logits.float(), masks.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        logger.info(
            "epoch %d/%d loss=%.4f", epoch + 1, epochs, running / max(n_batches, 1)
        )


@torch.no_grad()
def _evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> tuple[list[str], np.ndarray]:
    encoder.eval()
    decoder.eval()
    from fmpool.estimators import dice_per_case

    all_ids: list[str] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_enabled = use_bf16 and device.type == "cuda"
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks_np = (masks.numpy() > 0.5).astype(bool)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled
        ):
            feats = encoder(imgs)
            logits = decoder(feats)
        preds = (torch.sigmoid(logits.float()).cpu().numpy() > 0.5).astype(bool)
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
# Atomic IO
# ---------------------------------------------------------------------------


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


def _atomic_save(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptation factorial trainer (paper §8).")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", choices=("lora", "full_ft", "rand_init"), required=True)
    p.add_argument("--lora-rank", type=int, default=None, choices=(8, 16, 64))
    p.add_argument("--lora-blocks", type=int, default=None, choices=(2, 4, 8, 12))
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Per-case-Dice ROOT (e.g. results/per_case_dice). Subdir tree is appended.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


def _config_tag(args: argparse.Namespace) -> str:
    if args.regime == "lora":
        if args.lora_rank is None or args.lora_blocks is None:
            raise SystemExit("--lora-rank and --lora-blocks are required for regime=lora")
        return f"lora_r{args.lora_rank}_b{args.lora_blocks}"
    if args.regime == "full_ft":
        return f"ft_e{args.epochs}"
    return "rand_init"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        if args.regime == "rand_init":
            encoder, spec = build_random_encoder(args.fm)
        else:
            encoder, spec = encoders.build_encoder(args.fm)
    except Exception as exc:  # noqa: BLE001
        logger.error("FM %s unloadable: %s", args.fm, exc)
        print(f"[train_adapted] ERROR: cannot load FM {args.fm!r}: {exc}", file=sys.stderr)
        return 2
    encoder = encoder.to(device)

    if args.regime == "lora":
        if args.lora_rank is None or args.lora_blocks is None:
            print(
                "[train_adapted] ERROR: lora regime needs --lora-rank/--lora-blocks",
                file=sys.stderr,
            )
            return 4
        encoder.eval()
        apply_lora(encoder, args.fm, rank=args.lora_rank, n_last_blocks=args.lora_blocks)
        # LoRA params were instantiated on CPU; move freshly-added modules to device.
        encoder = encoder.to(device)
        encoder_train_mode = False  # keep frozen layers in eval mode (no dropout/BN drift).
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        encoder_train_mode = True

    normalise_fn = make_normaliser(spec)
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
            f"[train_adapted] ERROR: dataset {args.task!r} unavailable: {exc}",
            file=sys.stderr,
        )
        return 3

    loader_kwargs: dict = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    decoder = LinearSegHead(
        in_dim=spec.feature_dim, num_classes=1, out_size=INPUT_HW
    ).to(device)

    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_trainable += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info("regime=%s n_trainable_params=%d", args.regime, n_trainable)

    t0 = time.time()
    _train(
        encoder=encoder,
        decoder=decoder,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        encoder_train_mode=encoder_train_mode,
        use_bf16=args.bf16,
    )
    train_elapsed = time.time() - t0

    test_ids, per_case = _evaluate(
        encoder=encoder,
        decoder=decoder,
        loader=test_loader,
        device=device,
        use_bf16=args.bf16,
    )

    config_tag = _config_tag(args)
    out_root = Path(args.out)
    out_dir = out_root / args.regime / config_tag / args.task / args.fm
    out_path = out_dir / f"seed_{args.seed}.json"

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path("results/checkpoints") / args.regime / config_tag / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"

    if args.regime == "lora":
        state = {
            k: v.detach().cpu()
            for k, v in encoder.state_dict().items()
            if "lora_" in k
        }
        state.update(
            {f"decoder.{k}": v.detach().cpu() for k, v in decoder.state_dict().items()}
        )
    else:
        state = {f"encoder.{k}": v.detach().cpu() for k, v in encoder.state_dict().items()}
        state.update(
            {f"decoder.{k}": v.detach().cpu() for k, v in decoder.state_dict().items()}
        )
    _atomic_save(state, ckpt_path)
    ckpt_sha = determinism.sha256_file(ckpt_path)

    payload = {
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "regime": args.regime,
        "lora_rank": int(args.lora_rank) if args.lora_rank is not None else None,
        "lora_blocks": int(args.lora_blocks) if args.lora_blocks is not None else None,
        "epochs": int(args.epochs),
        "n_trainable_params": int(n_trainable),
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
