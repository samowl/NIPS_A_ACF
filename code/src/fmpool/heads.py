"""M3 decoder capacity sweep heads.

Provides 4 head designs that all share the same forward signature:

    head(spatial: Tensor[B, C, h, w]) -> logits: Tensor[B, num_classes, H, W]

where ``(H, W)`` is fixed to ``out_size`` (typically 224x224). All heads
return logits at the target resolution; BCE-with-logits is applied
externally by the trainer.

Designs
-------
1. ``linear_1x1`` -- SPEC §4 default (1x1 conv + bilinear upsample).
2. ``mlp_2layer`` -- 1x1 -> ReLU -> 1x1 (hidden=256) + bilinear upsample.
3. ``unet_lite``  -- 2-stage upsample with ConvBNReLU at each stage.
4. ``transformer_decoder`` -- single TransformerEncoderLayer over the
   spatial tokens, then 1x1 conv + bilinear upsample.

All modules expose ``out_size`` and ``in_dim`` for parity with
``LinearSegHead``. Use :func:`build_head` to instantiate by name.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HEAD_DESIGNS: tuple[str, ...] = (
    "linear_1x1",
    "mlp_2layer",
    "unet_lite",
    "transformer_decoder",
)


def _validate_init(in_dim: int, num_classes: int, out_size: Tuple[int, int]) -> None:
    if in_dim <= 0:
        raise ValueError(f"in_dim must be positive, got {in_dim}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if len(out_size) != 2 or any(s <= 0 for s in out_size):
        raise ValueError(f"out_size must be a positive (H, W) pair, got {out_size}")


class Linear1x1Head(nn.Module):
    """SPEC §4 baseline: 1x1 conv + bilinear upsample."""

    def __init__(
        self, in_dim: int, num_classes: int, out_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        _validate_init(in_dim, num_classes, out_size)
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.out_size: Tuple[int, int] = (int(out_size[0]), int(out_size[1]))
        self.proj = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        logits = self.proj(spatial)
        return F.interpolate(
            logits, size=self.out_size, mode="bilinear", align_corners=False
        )


class MLP2LayerHead(nn.Module):
    """1x1 -> ReLU -> 1x1 (hidden=256) + bilinear upsample."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        out_size: Tuple[int, int],
        hidden: int = 256,
    ) -> None:
        super().__init__()
        _validate_init(in_dim, num_classes, out_size)
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.out_size: Tuple[int, int] = (int(out_size[0]), int(out_size[1]))
        self.fc1 = nn.Conv2d(in_dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        x = self.fc1(spatial)
        x = F.relu(x, inplace=True)
        logits = self.fc2(x)
        return F.interpolate(
            logits, size=self.out_size, mode="bilinear", align_corners=False
        )


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class UNetLiteHead(nn.Module):
    """2-stage upsampling decoder.

    Each stage: bilinear x2 upsample -> ConvBNReLU. After two stages the
    feature is bilinear-resized to ``out_size`` and projected via 1x1 conv.
    Channels go in_dim -> in_dim/2 -> in_dim/4 (clamped to >= num_classes).
    """

    def __init__(
        self, in_dim: int, num_classes: int, out_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        _validate_init(in_dim, num_classes, out_size)
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.out_size: Tuple[int, int] = (int(out_size[0]), int(out_size[1]))
        c1 = max(in_dim // 2, max(num_classes, 16))
        c2 = max(in_dim // 4, max(num_classes, 16))
        self.stage1 = _ConvBNReLU(in_dim, c1)
        self.stage2 = _ConvBNReLU(c1, c2)
        self.proj = nn.Conv2d(c2, num_classes, kernel_size=1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(spatial, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return self.proj(x)


class TransformerDecoderHead(nn.Module):
    """Single TransformerEncoderLayer over spatial tokens, then 1x1 + upsample.

    Treats each of the ``h * w`` spatial locations as a token of dimension
    ``in_dim``. ``nhead`` is auto-fallback to a divisor of ``in_dim`` if 8
    does not divide cleanly (e.g. some FMs have non-multiple-of-8 channels).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        out_size: Tuple[int, int],
        nhead: int = 8,
        dim_feedforward: int = 1024,
    ) -> None:
        super().__init__()
        _validate_init(in_dim, num_classes, out_size)
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.out_size: Tuple[int, int] = (int(out_size[0]), int(out_size[1]))
        if nhead <= 0:
            raise ValueError(f"nhead must be positive, got {nhead}")
        if in_dim % nhead != 0:
            raise ValueError(
                f"in_dim={in_dim} must be divisible by nhead={nhead} "
                f"for TransformerEncoderLayer"
            )
        self.nhead = int(nhead)
        self.layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.proj = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        b, c, h, w = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2).contiguous()
        tokens = self.layer(tokens)
        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        logits = self.proj(x)
        return F.interpolate(
            logits, size=self.out_size, mode="bilinear", align_corners=False
        )


def build_head(
    name: str,
    in_dim: int,
    out_channels: int = 1,
    out_size: Tuple[int, int] = (224, 224),
) -> nn.Module:
    """Factory for M3 head designs.

    Parameters
    ----------
    name:
        One of :data:`HEAD_DESIGNS`.
    in_dim:
        Channel dimension of the cached feature map.
    out_channels:
        Number of logit channels (binary tasks use 1).
    out_size:
        Target spatial size (H, W) for the upsampled logits.
    """
    key = name.lower()
    if key == "linear_1x1":
        return Linear1x1Head(in_dim, out_channels, out_size)
    if key == "mlp_2layer":
        return MLP2LayerHead(in_dim, out_channels, out_size)
    if key == "unet_lite":
        return UNetLiteHead(in_dim, out_channels, out_size)
    if key == "transformer_decoder":
        return TransformerDecoderHead(in_dim, out_channels, out_size)
    raise ValueError(
        f"unknown head design {name!r}; expected one of {HEAD_DESIGNS}"
    )


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


__all__ = [
    "HEAD_DESIGNS",
    "Linear1x1Head",
    "MLP2LayerHead",
    "UNetLiteHead",
    "TransformerDecoderHead",
    "build_head",
    "count_trainable_params",
]
