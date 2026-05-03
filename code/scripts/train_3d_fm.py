#!/usr/bin/env python
"""Train a 3D BraTS foreground segmentation model with MONAI architectures (M21).

This is the 3D segmentation-architecture experiment referenced in Appendix ``app:3dfm``.
Four architectures are trained on full 3D BraTS volumes (NOT 2D axial
slices) for the binary foreground (label > 0) task:

* ``segresnet``  — :class:`monai.networks.nets.SegResNet`. Optionally
  initialised from the MONAI Model Zoo ``brats_mri_segmentation`` bundle
  (BraTS multi-class pretrained weights), then adapted to a 1-channel foreground
  head. The bundle's encoder body is reused; the final 1x1x1 conv is
  replaced and trained from scratch.
* ``dynunet``    — :class:`monai.networks.nets.DynUNet`. Random init.
* ``unetr``      — :class:`monai.networks.nets.UNETR`. Random init.
* ``swinunetr``  — :class:`monai.networks.nets.SwinUNETR`. Random init.
  The paper documents this cell as ``failed`` and it is excluded from
  Δ-statistics, but the cell is still attempted so the failure is on
  record.

Per-case Dice on the 3D test set is written to::

    {out}/seed_{seed}.json

with the SPEC §8 schema (extra fields: ``arch``, ``n_test``,
``checkpoint_sha256``, ``training_elapsed_s``).

Wallclock estimate (A100, batch=1, sliding-window inference):

* ~10–15 min/epoch × 100 epochs ≈ 16–25 GPU-h per cell.
* 4 arch × 4 seeds = 16 cells (paper) → 256–400 GPU-h. With the
  recommended 2-seed first pass {42, 43}, 4 arch × 2 seeds = 8 cells →
  ~150 GPU-h on a single A100.

CLI
~~~

::

    python scripts/train_3d_fm.py \
        --arch {segresnet,dynunet,unetr,swinunetr} \
        --seed SEED --epochs 100 \
        --out results/per_case_dice_3d/{arch}/

Exit codes
~~~~~~~~~~

* ``0`` — success.
* ``2`` — pretrained bundle download/load failure for SegResNet
  (only when ``--use-pretrained`` is set; otherwise random init).
* ``3`` — dataset / split unavailable.
* ``4`` — MONAI not installed.

Pretrained weights
~~~~~~~~~~~~~~~~~~

* Bundle name: ``brats_mri_segmentation``
* Source: MONAI Model Zoo (``Project-MONAI/model-zoo``)
* Catalog: https://catalog.ngc.nvidia.com/orgs/monaihosting/teams/monaitoolkit/models/monai-model-zoo
* Download API: ``monai.bundle.download(name="brats_mri_segmentation",
  bundle_dir=BUNDLE_DIR, source="monaihosting")``
* Loaded model file: ``brats_mri_segmentation/models/model.pt`` (4-channel
  in, 3-class out — TC/WT/ET). For our binary foreground head we load encoder
  weights only, then replace ``conv_final`` with a fresh 1-channel conv.

MONAI version requirement: ``monai>=1.3,<2.0`` (verified for the bundle
download API and ``SegResNet``/``UNETR``/``SwinUNETR``/``DynUNet``
constructors used here).
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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ``fmpool`` utilities are reused for split resolution, determinism, Dice.
from fmpool import datasets as fmpool_datasets
from fmpool import determinism
from fmpool.estimators import dice_per_case

logger = logging.getLogger("fmpool.train_3d_fm")

# --- 3D protocol constants (paper Appx app:3dfm) ----------------------------
TARGET_SHAPE: tuple[int, int, int] = (240, 240, 155)
BRATS_MODALITIES: tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")  # 4-channel
LR: float = 1e-3
ROI_SIZE: tuple[int, int, int] = (128, 128, 128)  # sliding window for inference
BUNDLE_NAME = "brats_mri_segmentation"
BUNDLE_SOURCE = "monaihosting"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a MONAI 3D segmentation network on BraTS foreground (M21).",
    )
    p.add_argument(
        "--arch",
        required=True,
        choices=("segresnet", "dynunet", "unetr", "swinunetr"),
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-case Dice JSON.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override default results/checkpoints_3d/{arch}/ path.",
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("results/monai_bundles"),
        help="Where to cache the MONAI Model Zoo bundle.",
    )
    p.add_argument(
        "--use-pretrained",
        action="store_true",
        help=(
            "For --arch=segresnet, initialise from the brats_mri_segmentation "
            "bundle. Ignored for other architectures."
        ),
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override FMPOOL_DATA_ROOT (defaults to configs/paths.yaml).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device ('cuda', 'cpu'). Auto-detect if None.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the network, run a forward pass on a small dummy 3D "
            "tensor, and exit. Does not touch the dataset or train."
        ),
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 3D BraTS dataset
# ---------------------------------------------------------------------------


class BraTSWT3DDataset(Dataset):
    """Full-3D BraTS foreground loader, reusing the 2D split's subject IDs.

    ``brats_wt_split.json`` stores per-slice ``case_ids`` of the form
    ``{subject_id}_slice{axi:03d}``; we reduce to unique subject IDs and
    load each subject's full 4-channel volume (T1N, T1C, T2W, T2F) plus
    binary foreground mask (seg > 0).

    Returns
    -------
    image : torch.FloatTensor of shape ``(4, 240, 240, 155)``
        Per-channel z-score normalised within the brain mask
        (image > 0); zeros outside.
    mask : torch.BoolTensor of shape ``(1, 240, 240, 155)``
    case_id : str (subject id)
    """

    def __init__(self, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"3D BraTS split must be train/val/test, got {split!r}"
            )
        committed = fmpool_datasets.load_committed_split("brats_wt")
        if committed is None:
            raise FileNotFoundError(
                "brats_wt_split.json missing; run the 2D BraTS loader once "
                "to materialise the split before training the 3D pool."
            )
        # Map subject_id -> 2D-split assignment (deduplicated).
        subject_split: dict[str, str] = {}
        for cid in committed.get("train", []):
            sub = cid.rsplit("_slice", 1)[0]
            subject_split.setdefault(sub, "train")
        for cid in committed.get("test", []):
            sub = cid.rsplit("_slice", 1)[0]
            subject_split.setdefault(sub, "test")
        self.subject_ids: list[str] = sorted(
            sub for sub, sp in subject_split.items() if sp == split
        )

        # Resolve subject -> on-disk path.
        from fmpool.datasets import resolve_task_root, require_dir

        self.root = resolve_task_root("brats")
        require_dir(self.root, dataset="brats")
        self._dirs: dict[str, Path] = {}
        for parent in (
            self.root / "training" / "training_data1_v2",
            self.root / "additional" / "training_data_additional",
        ):
            if not parent.is_dir():
                continue
            for sub_dir in parent.iterdir():
                if sub_dir.is_dir() and sub_dir.name in subject_split:
                    self._dirs[sub_dir.name] = sub_dir
        missing = [s for s in self.subject_ids if s not in self._dirs]
        if missing:
            raise FileNotFoundError(
                f"3D BraTS: {len(missing)} subject directories missing "
                f"(first 3: {missing[:3]})."
            )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        import nibabel as nib  # local: heavy optional dep

        sub_id = self.subject_ids[idx]
        sub_dir = self._dirs[sub_id]
        channels: list[np.ndarray] = []
        for mod in BRATS_MODALITIES:
            arr = nib.load(
                str(sub_dir / f"{sub_id}-{mod}.nii.gz")
            ).get_fdata().astype(np.float32)
            mask = arr > 0
            if mask.any():
                mean = float(arr[mask].mean())
                std = float(arr[mask].std()) + 1e-6
                arr = (arr - mean) / std
                arr[~mask] = 0.0
            channels.append(arr)
        image = np.stack(channels, axis=0)  # (4, H, W, D)
        seg = nib.load(
            str(sub_dir / f"{sub_id}-seg.nii.gz")
        ).get_fdata()
        mask_bool = (seg > 0)[None, ...]  # (1, H, W, D)
        if image.shape[1:] != TARGET_SHAPE:
            raise ValueError(
                f"BraTS volume {sub_id}: expected {TARGET_SHAPE} "
                f"got {image.shape[1:]}"
            )
        img_t = torch.from_numpy(image).contiguous()
        mask_t = torch.from_numpy(mask_bool).contiguous()
        return img_t, mask_t, sub_id


def _collate_3d(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0).float()
    ids = [b[2] for b in batch]
    return imgs, masks, ids


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(arch: str) -> nn.Module:
    """Construct one of the four MONAI 3D networks for binary WT."""
    from monai.networks.nets import (
        DynUNet,
        SegResNet,
        SwinUNETR,
        UNETR,
    )

    if arch == "segresnet":
        # Default config matches MONAI brats_mri_segmentation bundle.
        return SegResNet(
            spatial_dims=3,
            init_filters=8,
            in_channels=4,
            out_channels=1,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )
    if arch == "dynunet":
        # Conservative, BraTS-style nnU-Net config.
        kernels = [[3, 3, 3]] * 5
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        return DynUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=1,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    if arch == "unetr":
        return UNETR(
            in_channels=4,
            out_channels=1,
            img_size=ROI_SIZE,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            dropout_rate=0.0,
        )
    if arch == "swinunetr":
        return SwinUNETR(
            img_size=ROI_SIZE,
            in_channels=4,
            out_channels=1,
            feature_size=48,
            use_checkpoint=False,
        )
    raise ValueError(f"unknown arch: {arch!r}")


def _maybe_load_segresnet_pretrained(
    model: nn.Module, bundle_dir: Path
) -> str:
    """Download + load brats_mri_segmentation encoder into a SegResNet.

    The bundle ships a 4-channel-in / 3-class-out SegResNet checkpoint
    (TC/WT/ET). For binary WT we load encoder/decoder weights with
    ``strict=False`` — the final ``conv_final.conv`` (3-channel out) is
    silently skipped, leaving our fresh 1-channel head untouched.

    Returns the absolute path of the loaded ``model.pt`` for provenance.
    """
    from monai.bundle import download as bundle_download

    bundle_dir = Path(bundle_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / BUNDLE_NAME
    weights_path = bundle_path / "models" / "model.pt"
    if not weights_path.is_file():
        logger.info(
            "downloading MONAI bundle %r to %s", BUNDLE_NAME, bundle_dir
        )
        bundle_download(
            name=BUNDLE_NAME,
            bundle_dir=str(bundle_dir),
            source=BUNDLE_SOURCE,
        )
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"after bundle_download, weights still missing: {weights_path}"
        )
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(
        state["model"], dict
    ):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        "loaded %s (missing=%d unexpected=%d)",
        weights_path,
        len(missing),
        len(unexpected),
    )
    return str(weights_path)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def _train_one(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for imgs, masks, _ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            loss = bce(logits, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        avg = running / max(n_batches, 1)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    arch: str,
) -> tuple[list[str], np.ndarray]:
    """Per-case 3D Dice using sliding-window inference for ROI archs."""
    from monai.inferers import sliding_window_inference

    model.eval()
    all_ids: list[str] = []
    preds_flat: list[np.ndarray] = []
    targets_flat: list[np.ndarray] = []
    use_window = arch in {"unetr", "swinunetr"}
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        if use_window:
            logits = sliding_window_inference(
                imgs, ROI_SIZE, sw_batch_size=1, predictor=model, overlap=0.5
            )
        else:
            logits = model(imgs)
        pred = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(bool)
        tgt = (masks.numpy() > 0.5).astype(bool)
        all_ids.extend(ids)
        preds_flat.append(pred.reshape(pred.shape[0], -1))
        targets_flat.append(tgt.reshape(tgt.shape[0], -1))
    if not preds_flat:
        return [], np.zeros((0,), dtype=np.float64)
    return all_ids, dice_per_case(
        np.concatenate(preds_flat, 0),
        np.concatenate(targets_flat, 0),
    )


# ---------------------------------------------------------------------------
# Atomic IO (mirrors train_frozen_fm.py)
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


def _atomic_save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def _dry_run(arch: str, device: torch.device) -> int:
    """Build the network and run a forward pass on a small dummy 3D tensor."""
    try:
        import monai  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        logger.error("monai not installed: %s", exc)
        return 4
    model = _build_model(arch).to(device)
    model.eval()
    # ROI-sized dummy input — UNETR/SwinUNETR require img_size match.
    x = torch.randn(1, 4, *ROI_SIZE, device=device)
    with torch.no_grad():
        y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "[dry-run] arch=%s in=%s out=%s n_params=%d",
        arch,
        tuple(x.shape),
        tuple(y.shape),
        n_params,
    )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        import monai  # noqa: F401
    except ImportError as exc:
        logger.error("monai not installed: %s", exc)
        print(
            "[train_3d_fm] ERROR: MONAI not installed. "
            "Install: pip install 'monai>=1.3,<2.0' nibabel",
            file=sys.stderr,
        )
        return 4

    determinism.set_seed(args.seed)
    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.dry_run:
        return _dry_run(args.arch, device)

    # 1. Build model.
    model = _build_model(args.arch).to(device)
    pretrained_src: str | None = None
    if args.arch == "segresnet" and args.use_pretrained:
        try:
            pretrained_src = _maybe_load_segresnet_pretrained(
                model, args.bundle_dir
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("pretrained bundle load failed: %s", exc)
            print(
                f"[train_3d_fm] ERROR: pretrained bundle load failed: {exc}",
                file=sys.stderr,
            )
            return 2

    # 2. Build datasets.
    try:
        train_ds = BraTSWT3DDataset(split="train")
        test_ds = BraTSWT3DDataset(split="test")
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("dataset unavailable: %s", exc)
        print(
            f"[train_3d_fm] ERROR: BraTS 3D dataset unavailable: {exc}",
            file=sys.stderr,
        )
        return 3

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_3d,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_3d,
        drop_last=False,
    )

    # 3. Train.
    t0 = time.time()
    _train_one(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
    )
    train_elapsed = time.time() - t0

    # 4. Evaluate (3D per-case Dice).
    test_ids, per_case = _evaluate(
        model=model,
        loader=test_loader,
        device=device,
        arch=args.arch,
    )

    # 5. Atomic checkpoint.
    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path("results/checkpoints_3d") / args.arch
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_save_checkpoint(model, ckpt_path)
    ckpt_sha = determinism.sha256_file(ckpt_path)

    # 6. Per-case Dice JSON.
    out_path = Path(args.out) / f"seed_{args.seed}.json"
    payload: dict[str, Any] = {
        "task": "brats_wt_3d",
        "arch": args.arch,
        "arch_name": args.arch,
        "seed": int(args.seed),
        "n_test": int(len(per_case)),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(np.mean(per_case)) if per_case.size else float("nan"),
        "checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "pretrained_source": pretrained_src,
        "epochs": int(args.epochs),
        "lr": LR,
        "batch_size": int(args.batch_size),
        "roi_size": list(ROI_SIZE),
        "target_shape": list(TARGET_SHAPE),
        "modalities": list(BRATS_MODALITIES),
    }
    _atomic_write_bytes(out_path, json.dumps(payload, indent=2).encode("utf-8"))
    logger.info(
        "wrote %s (n_test=%d mean_dice=%.4f)",
        out_path,
        payload["n_test"],
        payload["mean_dice"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
