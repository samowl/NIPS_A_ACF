"""Linear segmentation probe head (SPEC §4).

A single 1x1 conv on top of frozen FM features, followed by bilinear
upsampling to the target output size. This is the ONLY decoder architecture
used for frozen-FM probes - SPEC §4 calls it out explicitly and the same head
is trained across every (task, FM, seed) combination.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSegHead(nn.Module):
    """SPEC §4 linear probe: 1x1 conv + bilinear upsample.

    Parameters
    ----------
    in_dim:
        Channel dimension of the encoder feature map ``[B, in_dim, h, w]``.
    num_classes:
        Number of output logit channels. Binary tasks use ``num_classes=1``
        (BCE-with-logits).
    out_size:
        ``(H, W)`` target spatial size for bilinear upsampling. For the 224 x
        224 probe this is ``(224, 224)``.
    """

    def __init__(
        self, in_dim: int, num_classes: int, out_size: tuple[int, int]
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if len(out_size) != 2 or any(s <= 0 for s in out_size):
            raise ValueError(
                f"out_size must be a positive (H, W) pair, got {out_size}"
            )
        self.proj = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        self.out_size: tuple[int, int] = (int(out_size[0]), int(out_size[1]))

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        logits = self.proj(spatial)
        return F.interpolate(
            logits, size=self.out_size, mode="bilinear", align_corners=False
        )


__all__ = ["LinearSegHead"]
