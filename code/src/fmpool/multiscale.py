"""Multi-scale feature extractors for the M4 UNet-skip decoder sweep.

DEVIATION FROM SPEC §4
----------------------
SPEC §4 mandates a 1x1 conv dense decoder over the FINAL feature map of each
FM. M4 is an explicit sensitivity probe for that choice: we expose four
intermediate scales per backbone and feed them into a UNet-style decoder
with skip connections (see :mod:`fmpool.heads_m4`). This module adds the
multi-scale plumbing without altering the SPEC §4 frozen-pool invariants
used by every other M-matrix.

Per family
~~~~~~~~~~
- DINOv2 ViT-B/14 / ViT-S/14 : intermediate ``blocks[3,6,9,11]`` patch tokens
  (CLS-stripped) reshaped to ``[B, D, 16, 16]``.
- BiomedCLIP                 : timm trunk ``blocks[3,6,9,11]`` patch tokens
  reshaped to ``[B, 768, 14, 14]``.
- ResNet50                   : ``layer1`` (256 ch / 56), ``layer2`` (512 ch /
  28), ``layer3`` (1024 ch / 14), ``layer4`` (2048 ch / 7).
- ConvNeXt-T                 : ``features[1,3,5,7]`` outputs which correspond
  to the 4 stage outputs (96/56, 192/28, 384/14, 768/7).

All ``forward`` implementations return a list of 4 tensors in
SHALLOW -> DEEP order (level 0 = shallowest / largest spatial,
level 3 = deepest / smallest spatial). The decoder consumes this list.

The encoders here are CONSTRUCTED FRESH from the same official upstreams
used by :mod:`fmpool.encoders`. We keep them frozen and in eval mode and
return a metadata struct describing each level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from fmpool import encoders

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiScaleSpec:
    """Per-FM metadata describing the 4 multi-scale levels."""

    fm_id: str
    family: str
    # Channel dim per level in shallow->deep order.
    level_dims: tuple[int, int, int, int]
    # (H, W) per level in the same order.
    level_hw: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ]
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]


# Block indices for the four ViT backbones (12-block ViT-B; we tap blocks
# 3/6/9/11). 0-indexed; the LAST tap is always the final block so that
# level 3 == final-feature parity with SPEC §4.
_VIT_TAP_INDICES: tuple[int, int, int, int] = (3, 6, 9, 11)


# ---------------------------------------------------------------------------
# ViT multi-scale wrappers
# ---------------------------------------------------------------------------


class _DINOv2MultiScale(nn.Module):
    """DINOv2 multi-scale wrapper.

    Calls ``model.get_intermediate_layers(x, n=tap_indices, reshape=True,
    return_class_token=False)``, the official DINOv2 helper that returns
    NCHW tensors with the CLS token already stripped. Shape per layer:
    ``[B, D, 16, 16]`` at 224 input.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        grid: tuple[int, int],
        tap_indices: tuple[int, int, int, int] = _VIT_TAP_INDICES,
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = int(feature_dim)
        self.grid = grid
        self.tap_indices = tap_indices

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.model.get_intermediate_layers(
            x, n=self.tap_indices, reshape=True, return_class_token=False
        )
        return [f.contiguous() for f in feats]


class _TimmViTMultiScale(nn.Module):
    """timm/open_clip ViT multi-scale wrapper.

    Replays ``patch_embed -> +pos_embed -> blocks`` manually so we can
    capture intermediate token sequences. Strips CLS (and any register)
    tokens and reshapes to NCHW. Used for BiomedCLIP.
    """

    def __init__(
        self,
        trunk: nn.Module,
        feature_dim: int,
        grid: tuple[int, int],
        tap_indices: tuple[int, int, int, int] = _VIT_TAP_INDICES,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.feature_dim = int(feature_dim)
        self.grid = grid
        self.tap_indices = tap_indices

    def _strip_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Drop CLS (and any register) tokens to leave [B, h*w, D]."""
        h, w = self.grid
        n_expected = h * w
        _b, t, _d = tokens.shape
        if t == n_expected:
            return tokens
        if t > n_expected:
            return tokens[:, t - n_expected :]
        raise RuntimeError(
            f"timm ViT token count {t} smaller than expected patch count {n_expected}"
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        m = self.trunk
        x = m.patch_embed(x)
        if hasattr(m, "_pos_embed"):
            x = m._pos_embed(x)
        else:
            cls = m.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
            x = x + m.pos_embed
        if hasattr(m, "patch_drop"):
            x = m.patch_drop(x)
        if hasattr(m, "norm_pre"):
            x = m.norm_pre(x)

        outs: list[torch.Tensor] = []
        tap_set = set(self.tap_indices)
        last_tap = max(self.tap_indices)
        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if i in tap_set:
                tokens = x
                # Apply the trunk's final norm to the LAST tap only so it
                # numerically matches the SPEC §4 frozen-pool feature.
                if i == last_tap and hasattr(m, "norm"):
                    tokens = m.norm(tokens)
                patch = self._strip_prefix_tokens(tokens)
                b, _n, d = patch.shape
                h, w = self.grid
                outs.append(
                    patch.transpose(1, 2).reshape(b, d, h, w).contiguous()
                )
            if i >= last_tap:
                break
        return outs


# ---------------------------------------------------------------------------
# CNN multi-scale wrappers
# ---------------------------------------------------------------------------


class _ResNetMultiScale(nn.Module):
    """ResNet50 multi-scale wrapper.

    Returns ``[layer1, layer2, layer3, layer4]`` outputs. At 224 input these
    have channels 256/512/1024/2048 and grids 56/28/14/7.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class _ConvNeXtMultiScale(nn.Module):
    """ConvNeXt-T multi-scale wrapper.

    torchvision ``convnext_tiny.features`` is an 8-element ``Sequential`` of
    alternating ``Conv2dNormActivation`` (downsample) and ``CNBlock`` stages.
    Indices 1/3/5/7 are the outputs of the four stages with channels
    96/192/384/768 and grids 56/28/14/7 at 224 input.
    """

    _STAGE_INDICES: tuple[int, int, int, int] = (1, 3, 5, 7)

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        wanted = set(self._STAGE_INDICES)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in wanted:
                outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# Specs (must match the wrappers above)
# ---------------------------------------------------------------------------


MULTISCALE_SPECS: dict[str, MultiScaleSpec] = {
    "dinov2_vitb14": MultiScaleSpec(
        fm_id="dinov2_vitb14",
        family="DINOv2",
        level_dims=(768, 768, 768, 768),
        level_hw=((16, 16), (16, 16), (16, 16), (16, 16)),
        norm_mean=encoders.FM_SPECS["dinov2_vitb14"].norm_mean,
        norm_std=encoders.FM_SPECS["dinov2_vitb14"].norm_std,
    ),
    "dinov2_vits14": MultiScaleSpec(
        fm_id="dinov2_vits14",
        family="DINOv2",
        level_dims=(384, 384, 384, 384),
        level_hw=((16, 16), (16, 16), (16, 16), (16, 16)),
        norm_mean=encoders.FM_SPECS["dinov2_vits14"].norm_mean,
        norm_std=encoders.FM_SPECS["dinov2_vits14"].norm_std,
    ),
    "biomedclip": MultiScaleSpec(
        fm_id="biomedclip",
        family="BiomedCLIP",
        level_dims=(768, 768, 768, 768),
        level_hw=((14, 14), (14, 14), (14, 14), (14, 14)),
        norm_mean=encoders.FM_SPECS["biomedclip"].norm_mean,
        norm_std=encoders.FM_SPECS["biomedclip"].norm_std,
    ),
    "resnet50": MultiScaleSpec(
        fm_id="resnet50",
        family="ResNet",
        level_dims=(256, 512, 1024, 2048),
        level_hw=((56, 56), (28, 28), (14, 14), (7, 7)),
        norm_mean=encoders.FM_SPECS["resnet50"].norm_mean,
        norm_std=encoders.FM_SPECS["resnet50"].norm_std,
    ),
    "convnext_tiny": MultiScaleSpec(
        fm_id="convnext_tiny",
        family="ConvNeXt",
        level_dims=(96, 192, 384, 768),
        level_hw=((56, 56), (28, 28), (14, 14), (7, 7)),
        norm_mean=encoders.FM_SPECS["convnext_tiny"].norm_mean,
        norm_std=encoders.FM_SPECS["convnext_tiny"].norm_std,
    ),
}


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_multiscale_encoder(
    fm_id: str,
) -> tuple[nn.Module, MultiScaleSpec]:
    """Return a frozen multi-scale encoder + its :class:`MultiScaleSpec`."""

    if fm_id not in MULTISCALE_SPECS:
        raise KeyError(
            f"M4 multi-scale not configured for {fm_id!r}. "
            f"Supported: {sorted(MULTISCALE_SPECS.keys())}"
        )
    spec = MULTISCALE_SPECS[fm_id]

    if fm_id.startswith("dinov2_"):
        model = torch.hub.load("facebookresearch/dinov2", fm_id)
        wrapper: nn.Module = _DINOv2MultiScale(
            model,
            feature_dim=spec.level_dims[-1],
            grid=spec.level_hw[-1],
        )
    elif fm_id == "biomedclip":
        import open_clip

        model, _ = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        visual = model.visual
        trunk = visual.trunk if hasattr(visual, "trunk") else visual
        wrapper = _TimmViTMultiScale(
            trunk,
            feature_dim=spec.level_dims[-1],
            grid=spec.level_hw[-1],
        )
    elif fm_id == "resnet50":
        import torchvision.models as tvm

        rn = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        wrapper = _ResNetMultiScale(rn)
    elif fm_id == "convnext_tiny":
        import torchvision.models as tvm

        cn = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        wrapper = _ConvNeXtMultiScale(cn)
    else:  # pragma: no cover (guarded above)
        raise KeyError(fm_id)

    for p in wrapper.parameters():
        p.requires_grad = False
    wrapper.eval()

    logger.info(
        "built M4 multi-scale encoder fm=%s level_dims=%s level_hw=%s",
        fm_id,
        spec.level_dims,
        spec.level_hw,
    )
    return wrapper, spec


@torch.no_grad()
def extract_multiscale_features(
    model: nn.Module, fm_id: str, x: torch.Tensor
) -> list[torch.Tensor]:
    """Run the multi-scale encoder and validate the output shapes."""

    if fm_id not in MULTISCALE_SPECS:
        raise KeyError(fm_id)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W] input; got {tuple(x.shape)}")
    spec = MULTISCALE_SPECS[fm_id]
    feats = model(x)
    if not isinstance(feats, list) or len(feats) != 4:
        n = len(feats) if hasattr(feats, "__len__") else "NA"
        raise RuntimeError(
            f"{fm_id}: multi-scale forward returned {type(feats).__name__} "
            f"of len={n}; expected list[Tensor] of length 4"
        )
    for i, f in enumerate(feats):
        exp_c = spec.level_dims[i]
        exp_hw = spec.level_hw[i]
        got = tuple(f.shape[1:])
        expected = (exp_c, *exp_hw)
        if got != expected:
            raise RuntimeError(
                f"{fm_id}: level {i} shape {got} != expected {expected}"
            )
    return feats


__all__ = [
    "MultiScaleSpec",
    "MULTISCALE_SPECS",
    "build_multiscale_encoder",
    "extract_multiscale_features",
]
