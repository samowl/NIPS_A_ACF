"""M4 UNet-skip decoder head.

DEVIATION FROM SPEC §4
----------------------
SPEC §4 defines a 1x1 conv decoder over the FINAL feature map. M4 swaps in
a UNet-style decoder that consumes 4 multi-scale FM features (see
:mod:`fmpool.multiscale`) connected via additive skip paths. This file is
the head; the encoder lives in :mod:`fmpool.multiscale`. Used only by
``scripts/train_head_unet_skip.py``.

Shape contract
--------------
``forward(features: list[Tensor]) -> Tensor[B, num_classes, out_h, out_w]``

``features`` is a 4-element list ordered SHALLOW -> DEEP (level 0 has the
largest spatial size, level 3 the smallest). The decoder unrolls
deepest-first, projecting each level to a common ``decoder_channels`` width
via 1x1 conv, then upsampling to the next-shallower level's spatial size
and adding the projected skip. Two ``ConvBNReLU`` blocks fuse each merged
feature. Final stage upsamples to ``out_size`` and applies a 1x1 conv to
``num_classes``.

This head deliberately uses additive skip merges (not concat) so input
channels stay at ``decoder_channels`` throughout the decoder, keeping the
parameter count comparable across the 5 FMs in the M4 grid.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class UNetSkipHead(nn.Module):
    """UNet-style decoder over 4 multi-scale FM features.

    Parameters
    ----------
    level_dims:
        Channel dims of the 4 input scales in shallow->deep order.
    num_classes:
        Output channels (1 for binary segmentation).
    out_size:
        Final ``(H, W)``, typically ``(224, 224)``.
    decoder_channels:
        Width of the decoder lateral / fusion convs. Default 128.
    """

    def __init__(
        self,
        level_dims: Sequence[int],
        num_classes: int,
        out_size: tuple[int, int],
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(level_dims) != 4:
            raise ValueError(
                f"UNetSkipHead expects 4 input scales; got {len(level_dims)}"
            )
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if len(out_size) != 2 or any(s <= 0 for s in out_size):
            raise ValueError(f"out_size must be positive (H, W); got {out_size}")
        if decoder_channels <= 0:
            raise ValueError(
                f"decoder_channels must be positive; got {decoder_channels}"
            )

        self.level_dims: tuple[int, ...] = tuple(int(d) for d in level_dims)
        self.num_classes = int(num_classes)
        self.out_size: tuple[int, int] = (int(out_size[0]), int(out_size[1]))
        self.decoder_channels = int(decoder_channels)

        c = self.decoder_channels
        # Lateral 1x1 conv per scale -> common decoder width.
        self.lateral = nn.ModuleList(
            [nn.Conv2d(int(d), c, kernel_size=1) for d in self.level_dims]
        )
        # Fusion convs after each upsample+add merge. We merge 3 times
        # (deep->mid->shallow->shallowest); add (not concat) keeps channels
        # at c so input/output of each fusion stays c.
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(_ConvBNReLU(c, c), _ConvBNReLU(c, c))
                for _ in range(3)
            ]
        )
        # Final stage after upsample to out_size.
        self.final_fuse = _ConvBNReLU(c, c)
        self.classifier = nn.Conv2d(c, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise RuntimeError(
                f"UNetSkipHead expects 4 feature scales; got {len(features)}"
            )
        # Lateral projections to common decoder width.
        lats = [lat(f) for lat, f in zip(self.lateral, features)]
        # Top-down path: start from deepest (level 3), upsample to level i's
        # spatial size, add skip, fuse; for i in (2, 1, 0).
        x = lats[3]
        for fuse_idx, i in enumerate((2, 1, 0)):
            target_hw = lats[i].shape[-2:]
            x = F.interpolate(
                x, size=target_hw, mode="bilinear", align_corners=False
            )
            x = x + lats[i]
            x = self.fuse[fuse_idx](x)
        # Upsample to out_size and classify.
        x = F.interpolate(
            x, size=self.out_size, mode="bilinear", align_corners=False
        )
        x = self.final_fuse(x)
        return self.classifier(x)


def build_unet_skip_head(
    level_dims: Sequence[int],
    num_classes: int = 1,
    out_size: tuple[int, int] = (224, 224),
    decoder_channels: int = 128,
) -> UNetSkipHead:
    """Factory mirroring :func:`fmpool.heads.build_head` style."""
    return UNetSkipHead(
        level_dims=level_dims,
        num_classes=num_classes,
        out_size=out_size,
        decoder_channels=decoder_channels,
    )


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


__all__ = [
    "UNetSkipHead",
    "build_unet_skip_head",
    "count_trainable_params",
]
