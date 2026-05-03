"""Frozen foundation-model encoder wrappers.

Implements SPEC §3 dense-feature API for 19 FM backbones. Loaders use public
upstream hubs/checkpoints or documented mirrors where available; several
wrappers deliberately expose local dense-probe features rather than vendor
inference APIs. Checkpoint-gated entries require local credentials or explicit
checkpoint paths:

- DINOv2 ViT-S/B/L/14: ``torch.hub.load('facebookresearch/dinov2', ...)``
- BiomedCLIP, CLIP ViT-L/14, CLIP ViT-B/16, CLIP ViT-B/32: ``open_clip``
- ConvNeXt-T/S/B, EfficientNet-B0, ResNet18/34/50/101: ``torchvision``
- MAE ViT-B/16: ``timm.create_model('vit_base_patch16_224.mae', pretrained=True)``
  (official MAE ImageNet weights, mirrored on the timm hub).
- DeiT-B/16 (non-distilled): ``timm.create_model('deit_base_patch16_224',
  pretrained=True)`` (official ``dl.fbaipublicfiles.com/deit/...`` checkpoint).
- RETFound ViT-L/16 CFP: HuggingFace ``YukunZhou/RETFound_mae_natureCFP``
  (gated). Checkpoint can be supplied via ``RETFOUND_CFP_CHECKPOINT`` env var.
- MedSAM ViT-B/16 (image_encoder only, neck stripped): ``sam_model_registry``
  from ``segment_anything``. Local checkpoint via ``MEDSAM_CHECKPOINT`` env var.

All encoders expose :func:`extract_features` returning ``[B, D, h, w]`` dense
features for a ``[B, 3, 224, 224]`` input. They are frozen (``requires_grad
=False``) and in ``eval`` mode at build time. Device management is the
caller's responsibility.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation constants (SPEC §3)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_OPENAI_CLIP_MEAN: tuple[float, float, float] = (
    0.48145466,
    0.4578275,
    0.40821073,
)
_OPENAI_CLIP_STD: tuple[float, float, float] = (
    0.26862954,
    0.26130258,
    0.27577711,
)

# MedSAM local-probe normalisation. This benchmark feeds the MedSAM image
# encoder through the same 224-pixel frozen-feature path as the other encoders.
# We therefore use ImageNet-style normalisation for the local probe. This should
# not be read as official prompt-conditioned MedSAM preprocessing, whose public
# inference scripts resize to 1024 and use the full SAM prompt/mask-decoder path.


# ---------------------------------------------------------------------------
# Spec table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    """Static metadata for a single FM encoder."""

    fm_id: str
    feature_dim: int
    feature_hw: tuple[int, int]
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    family: str


FM_SPECS: dict[str, EncoderSpec] = {
    "dinov2_vitb14": EncoderSpec(
        fm_id="dinov2_vitb14",
        feature_dim=768,
        feature_hw=(16, 16),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DINOv2",
    ),
    "dinov2_vitl14": EncoderSpec(
        fm_id="dinov2_vitl14",
        feature_dim=1024,
        feature_hw=(16, 16),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DINOv2",
    ),
    "dinov2_vits14": EncoderSpec(
        fm_id="dinov2_vits14",
        feature_dim=384,
        feature_hw=(16, 16),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DINOv2",
    ),
    "biomedclip": EncoderSpec(
        fm_id="biomedclip",
        feature_dim=768,
        feature_hw=(14, 14),
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="BiomedCLIP",
    ),
    "clip_vitl14": EncoderSpec(
        fm_id="clip_vitl14",
        feature_dim=1024,
        feature_hw=(16, 16),
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="CLIP",
    ),
    "convnext_tiny": EncoderSpec(
        fm_id="convnext_tiny",
        feature_dim=768,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ConvNeXt",
    ),
    "convnext_small": EncoderSpec(
        fm_id="convnext_small",
        feature_dim=768,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ConvNeXt",
    ),
    "convnext_base": EncoderSpec(
        fm_id="convnext_base",
        feature_dim=1024,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ConvNeXt",
    ),
    "efficientnet_b0": EncoderSpec(
        fm_id="efficientnet_b0",
        feature_dim=1280,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="EfficientNet",
    ),
    "resnet50": EncoderSpec(
        fm_id="resnet50",
        feature_dim=2048,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    "resnet101": EncoderSpec(
        fm_id="resnet101",
        feature_dim=2048,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    "resnet18": EncoderSpec(
        fm_id="resnet18",
        feature_dim=512,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    "resnet34": EncoderSpec(
        fm_id="resnet34",
        feature_dim=512,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    # M2: 5-ViT same-arch row -------------------------------------------------
    "mae_vitb16": EncoderSpec(
        fm_id="mae_vitb16",
        feature_dim=768,
        feature_hw=(14, 14),  # 224 / 16 = 14
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="MAE",
    ),
    "deit_vitb16": EncoderSpec(
        fm_id="deit_vitb16",
        feature_dim=768,
        feature_hw=(14, 14),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DeiT",
    ),
    "clip_vitb16": EncoderSpec(
        fm_id="clip_vitb16",
        feature_dim=768,
        feature_hw=(14, 14),  # 224 / 16 = 14
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="CLIP",
    ),
    "clip_vitb32": EncoderSpec(
        fm_id="clip_vitb32",
        feature_dim=768,
        feature_hw=(7, 7),  # 224 / 32 = 7
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="CLIP",
    ),
    # M5: dense-medical FMs ---------------------------------------------------
    "retfound_vitl16": EncoderSpec(
        fm_id="retfound_vitl16",
        feature_dim=1024,
        feature_hw=(14, 14),
        norm_mean=_IMAGENET_MEAN,  # RETFound MAE uses IMAGENET_DEFAULT_MEAN/STD
        norm_std=_IMAGENET_STD,
        family="RETFound",
    ),
    "medsam_vitb16": EncoderSpec(
        fm_id="medsam_vitb16",
        feature_dim=768,  # pre-neck embed_dim (neck is stripped per SPEC)
        feature_hw=(14, 14),  # at 224 input, after pos_embed interpolation
        norm_mean=_IMAGENET_MEAN,  # local 224-pixel encoder-probe normalisation
        norm_std=_IMAGENET_STD,
        family="MedSAM",
    ),
}


# ---------------------------------------------------------------------------
# Thin wrapper modules per family
# ---------------------------------------------------------------------------


class _DINOv2Wrapper(nn.Module):
    """Wrap a DINOv2 ViT so ``forward`` returns ``[B, D, 16, 16]``.

    SPEC §3: ``model.forward_features(x)["x_norm_patchtokens"]`` reshape.
    """

    def __init__(self, model: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        patch = feats["x_norm_patchtokens"]  # [B, N, D]
        b, n, d = patch.shape
        h, w = self.grid
        if n != h * w:
            raise RuntimeError(
                f"DINOv2 patch count mismatch: got N={n}, expected {h}*{w}={h * w}. "
                "Input resolution must be 224 so the 14-patch grid yields 16x16."
            )
        return patch.transpose(1, 2).reshape(b, d, h, w).contiguous()


class _BiomedCLIPWrapper(nn.Module):
    """BiomedCLIP visual trunk (timm ViT-B/16) → ``[B, 768, 14, 14]``.

    SPEC §3: ``trunk.forward_features(x)[:, 1:]`` then reshape. The timm trunk
    returns ``[B, 1+N, D]`` with the CLS token at index 0.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.trunk = trunk
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk.forward_features(x)  # [B, 1+N, D]
        patch = feats[:, 1:]
        b, n, d = patch.shape
        h, w = self.grid
        if n != h * w:
            raise RuntimeError(
                f"BiomedCLIP patch count mismatch: got N={n}, expected {h}*{w}={h * w}."
            )
        return patch.transpose(1, 2).reshape(b, d, h, w).contiguous()


class _CLIPViTLWrapper(nn.Module):
    """CLIP ViT-L/14 visual tower → ``[B, 1024, 16, 16]`` pre-projection.

    SPEC §3: ``visual.forward_intermediates(x, indices=[-1],
    output_fmt='NCHW', intermediates_only=True)[0]``. This bypasses CLIP's
    attention-pool / projection, delivering the final block's dense tokens in
    NCHW layout. Using ``encode_image`` would pool to ``[B, D]`` and break the
    dense probe — the SPEC explicitly flags this.
    """

    def __init__(self, visual: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.visual = visual
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.visual.forward_intermediates(
            x,
            indices=[-1],
            output_fmt="NCHW",
            intermediates_only=True,
        )
        # open_clip>=3 returns ``{"image_intermediates": [tensor, ...]}``;
        # older versions returned a bare list/tuple. Handle both without
        # breaking the SPEC-documented call signature.
        if isinstance(out, dict):
            intermediates = out.get("image_intermediates")
            if intermediates is None:
                raise RuntimeError(
                    "CLIP ViT-L/14 forward_intermediates returned dict without "
                    f"'image_intermediates'; keys={list(out.keys())}"
                )
        else:
            intermediates = out
        feat = intermediates[-1]  # [B, D, h, w]
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"CLIP ViT-L/14 feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


class _CNNWrapper(nn.Module):
    """Generic torchvision CNN trunk → ``[B, D, 7, 7]`` at 224 input.

    SPEC §3: ``nn.Sequential(*list(model.children())[:-2])``. Slicing ``[:-2]``
    drops the final avgpool and classifier for ConvNeXt / EfficientNet / ResNet
    alike, leaving the last conv-stage feature map.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.trunk = trunk
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"CNN feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


class _TimmViTWrapper(nn.Module):
    """Wrap a timm VisionTransformer so ``forward`` returns ``[B, D, h, w]``.

    Used for MAE (``vit_base_patch16_224.mae``) and DeiT
    (``deit_base_patch16_224``). Both models expose
    ``forward_features(x) -> [B, 1+N, D]`` with the CLS token at index 0
    (DeiT-base, non-distilled, has a single CLS token; the distilled variant
    has two extra tokens — we deliberately do NOT use that variant).
    """

    def __init__(self, model: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)  # [B, 1+N, D] for ViT-B
        if feats.ndim != 3:
            raise RuntimeError(
                f"timm ViT forward_features returned ndim={feats.ndim}; expected 3"
            )
        h, w = self.grid
        n_expected = h * w
        # Drop the CLS token (index 0). If the model already returns the
        # post-pool [B, D] (num_classes=0 plus pre_logits), fail loudly.
        b, t, d = feats.shape
        if t == n_expected + 1:
            patch = feats[:, 1:]
        elif t == n_expected:
            patch = feats
        else:
            raise RuntimeError(
                f"timm ViT token count mismatch: got T={t}, expected {n_expected}"
                f" or {n_expected + 1}."
            )
        if d != self.feature_dim:
            raise RuntimeError(
                f"timm ViT feature_dim mismatch: got D={d}, expected {self.feature_dim}"
            )
        return patch.transpose(1, 2).reshape(b, d, h, w).contiguous()


class _CLIPViTBWrapper(nn.Module):
    """OpenAI CLIP ViT-B visual tower → dense pre-projection patch features.

    Same pattern as :class:`_CLIPViTLWrapper`. ``forward_intermediates`` with
    ``output_fmt='NCHW'`` and ``intermediates_only=True`` bypasses the CLIP
    attention-pool / projection, returning the final block's dense tokens
    in NCHW layout.
    """

    def __init__(self, visual: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.visual = visual
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.visual.forward_intermediates(
            x,
            indices=[-1],
            output_fmt="NCHW",
            intermediates_only=True,
        )
        if isinstance(out, dict):
            intermediates = out.get("image_intermediates")
            if intermediates is None:
                raise RuntimeError(
                    "CLIP ViT-B forward_intermediates returned dict without "
                    f"'image_intermediates'; keys={list(out.keys())}"
                )
        else:
            intermediates = out
        feat = intermediates[-1]
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"CLIP ViT-B feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


class _RETFoundWrapper(nn.Module):
    """RETFound ViT-L/16 → ``[B, 1024, 14, 14]`` dense patch tokens.

    Wraps the official ``models_vit.VisionTransformer`` from
    rmaphoh/RETFound_MAE. Calls ``forward_features`` if available, otherwise
    replays the standard ViT trunk (patch_embed → +pos_embed → blocks → norm)
    and drops the CLS token.
    """

    def __init__(self, model: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        # RETFound's models_vit defines forward_features that returns the CLS
        # token only when global_pool=False. To get patch tokens we replay the
        # trunk manually.
        b = x.shape[0]
        x = m.patch_embed(x)
        cls_tokens = m.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + m.pos_embed
        x = m.pos_drop(x)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        # Drop CLS token at position 0 → [B, N, D]
        patch = x[:, 1:]
        bn, n, d = patch.shape
        h, w = self.grid
        if n != h * w:
            raise RuntimeError(
                f"RETFound patch count mismatch: got N={n}, expected {h * w}."
            )
        if d != self.feature_dim:
            raise RuntimeError(
                f"RETFound feature_dim mismatch: got D={d}, expected {self.feature_dim}"
            )
        return patch.transpose(1, 2).reshape(bn, d, h, w).contiguous()


class _MedSAMWrapper(nn.Module):
    """MedSAM/SAM ViT-B image_encoder → ``[B, 768, 14, 14]`` pre-neck.

    The official ``segment_anything.modeling.image_encoder.ImageEncoderViT``
    runs ``patch_embed`` (stride=16), adds ``pos_embed``, applies the 12 ViT
    blocks, then permutes to NCHW and runs the ``neck`` (1x1+3x3 conv → 256).

    We strip the neck (per SPEC: "Feature dim=768"). We also interpolate the
    abs ``pos_embed`` from its training grid (1024/16 = 64) to whatever grid
    the input produces (224/16 = 14). All other parameters (patch_embed,
    blocks, rel_pos in attention) work at any resolution because the only
    resolution-coupled parameter is ``pos_embed``.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        feature_dim: int,
        grid: tuple[int, int],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.feature_dim = feature_dim
        self.grid = grid

    def _interp_pos_embed(self, target_hw: tuple[int, int]) -> torch.Tensor:
        # Original pos_embed is (1, 64, 64, embed_dim) for 1024 input.
        pos = self.image_encoder.pos_embed  # type: ignore[union-attr]
        if pos.shape[1:3] == target_hw:
            return pos
        # (1, H, W, D) -> (1, D, H, W) -> bicubic to target -> (1, H', W', D)
        pos_chw = pos.permute(0, 3, 1, 2)
        pos_chw = F.interpolate(
            pos_chw, size=target_hw, mode="bicubic", align_corners=False
        )
        return pos_chw.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.image_encoder
        # Step 1: patch_embed → (B, h, w, D)
        x = enc.patch_embed(x)
        # Step 2: pos_embed (interpolate if input grid != training grid)
        if enc.pos_embed is not None:
            target_hw = (x.shape[1], x.shape[2])
            pos = self._interp_pos_embed(target_hw)
            x = x + pos
        # Step 3: blocks
        for blk in enc.blocks:
            x = blk(x)
        # Step 4: permute to NCHW WITHOUT running the neck (which projects
        # from 768 → 256 and is trained for 1024 input).
        feat = x.permute(0, 3, 1, 2).contiguous()  # (B, 768, h, w)
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"MedSAM feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


# ---------------------------------------------------------------------------
# Per-FM loaders: public upstream/hub checkpoints plus documented local
# dense-probe wrappers.


def _load_dinov2(fm_id: str) -> nn.Module:
    # torch.hub.load yields the DINOv2 ViT with ``forward_features`` returning
    # the dict that contains ``x_norm_patchtokens``.
    model = torch.hub.load("facebookresearch/dinov2", fm_id)
    spec = FM_SPECS[fm_id]
    return _DINOv2Wrapper(model, spec.feature_dim, spec.feature_hw)


def _load_biomedclip() -> nn.Module:
    import open_clip

    model, _ = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    # open_clip CustomTextCLIP exposes the timm trunk via ``visual.trunk``.
    visual = model.visual
    trunk = visual.trunk if hasattr(visual, "trunk") else visual
    spec = FM_SPECS["biomedclip"]
    return _BiomedCLIPWrapper(trunk, spec.feature_dim, spec.feature_hw)


def _load_clip_vitl14() -> nn.Module:
    """Load the OpenAI CLIP ViT-L/14 visual tower via open_clip.

    Source: ``open_clip.create_model_and_transforms('ViT-L-14',
    pretrained='openai')``. open_clip also exposes
    ``ViT-L-14-quickgelu/openai`` and documents that many OpenAI checkpoints
    use QuickGELU; the released traces intentionally preserve this
    version-pinned ``ViT-L-14/openai`` loader for reproducibility rather than
    claiming parity with OpenAI's official CLIP inference API. The wrapper
    exposes the pre-projection 16x16 dense patch map at 224 input.
    """
    import open_clip

    # Keep the version-pinned open_clip name used for the released traces.
    # Strict OpenAI-CLIP parity would use the QuickGELU model definition; this
    # local dense-probe wrapper keeps the released trace path unchanged.
    out = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = out[0]
    spec = FM_SPECS["clip_vitl14"]
    return _CLIPViTLWrapper(model.visual, spec.feature_dim, spec.feature_hw)


def _load_torchvision_cnn(fm_id: str) -> nn.Module:
    import torchvision.models as tvm

    if fm_id == "convnext_tiny":
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif fm_id == "convnext_small":
        model = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif fm_id == "convnext_base":
        model = tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif fm_id == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet101":
        model = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet34":
        model = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown torchvision FM id: {fm_id!r}")

    trunk = nn.Sequential(*list(model.children())[:-2])
    spec = FM_SPECS[fm_id]
    return _CNNWrapper(trunk, spec.feature_dim, spec.feature_hw)


def _load_mae_vitb16() -> nn.Module:
    """Load MAE ViT-B/16 ImageNet-pretrained weights via timm.

    Source: ``timm.create_model('vit_base_patch16_224.mae', pretrained=True)``.
    The timm hub mirrors the official MAE checkpoint
    (``https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth``,
    sha256 unpublished by upstream). Setting ``num_classes=0`` removes the
    classifier head; we still call ``forward_features`` to obtain dense tokens.
    """

    import timm

    model = timm.create_model(
        "vit_base_patch16_224.mae", pretrained=True, num_classes=0
    )
    spec = FM_SPECS["mae_vitb16"]
    return _TimmViTWrapper(model, spec.feature_dim, spec.feature_hw)


def _load_deit_vitb16() -> nn.Module:
    """Load DeiT-B/16 (non-distilled) via timm.

    Source: ``timm.create_model('deit_base_patch16_224', pretrained=True)``,
    backed by the official checkpoint
    ``https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth``
    (filename hash ``b5f2ef4d`` is part of the URL, not a checksum). This is
    the NON-distilled DeiT-B/16.
    """

    import timm

    model = timm.create_model(
        "deit_base_patch16_224", pretrained=True, num_classes=0
    )
    spec = FM_SPECS["deit_vitb16"]
    return _TimmViTWrapper(model, spec.feature_dim, spec.feature_hw)


def _load_clip_vitb16() -> nn.Module:
    """Load OpenAI CLIP ViT-B/16 visual tower via open_clip.

    Source: ``open_clip.create_model_and_transforms('ViT-B-16-quickgelu',
    pretrained='openai')``. The ``-quickgelu`` suffix matches OpenAI's
    original activation (per open_clip README: "OpenAI pretrained weights
    will always default to QuickGELU"). The visual tower's pre-projection
    feature dim is 768 and the patch-token grid at 224 input is 14x14.
    """

    import open_clip

    out = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai"
    )
    model = out[0]
    spec = FM_SPECS["clip_vitb16"]
    return _CLIPViTBWrapper(model.visual, spec.feature_dim, spec.feature_hw)


def _load_clip_vitb32() -> nn.Module:
    """Load OpenAI CLIP ViT-B/32 visual tower via open_clip.

    Source: ``open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
    pretrained='openai')``. We use the QuickGELU model definition for the
    original OpenAI checkpoint and expose the pre-projection 7x7 patch map at
    224 input.
    """

    import open_clip

    out = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    model = out[0]
    spec = FM_SPECS["clip_vitb32"]
    return _CLIPViTBWrapper(model.visual, spec.feature_dim, spec.feature_hw)


def _load_retfound_vitl16() -> nn.Module:
    """Load RETFound ViT-L/16 (CFP, gated HuggingFace checkpoint).

    Source: ``YukunZhou/RETFound_mae_natureCFP`` on HuggingFace (gated). To
    use this loader you must either:

    1. Set ``RETFOUND_CFP_CHECKPOINT`` to a local ``.pth`` checkpoint path, OR
    2. Be authenticated to HF (``huggingface-cli login``) so that
       ``hf_hub_download`` can fetch ``RETFound_mae_natureCFP.pth``.

    Architecture matches the official ``models_vit.RETFound_mae``:
    embed_dim=1024, depth=24, num_heads=16, patch_size=16, mlp_ratio=4,
    qkv_bias=True. We build the trunk locally (using timm's
    VisionTransformer) and load the official state_dict with strict=False
    (the checkpoint contains an MAE decoder which we drop).
    """

    import timm

    # Build the encoder skeleton with the exact RETFound architecture.
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",  # keep CLS+patch tokens
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    # Compatibility: RETFound's official trunk lives in ``models_vit`` and uses
    # ``pos_drop`` + 24 ``blocks`` + ``norm`` layer names — these match timm's
    # VisionTransformer exactly, so the wrapper's forward replay is valid.

    ckpt_path = os.environ.get("RETFOUND_CFP_CHECKPOINT")
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="YukunZhou/RETFound_mae_natureCFP",
            filename="RETFound_mae_natureCFP.pth",
        )

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # The MAE checkpoint contains 'decoder_*' and 'mask_token' keys we don't
    # need for the encoder. strict=False allows those extras, but missing
    # encoder keys are treated as a hard preflight failure.
    missing, unexpected = model.load_state_dict(state, strict=False)
    encoder_missing = [k for k in missing if not k.startswith("head")]
    if encoder_missing:
        raise RuntimeError(
            "RETFound checkpoint is missing encoder keys "
            f"(n={len(encoder_missing)}, sample={encoder_missing[:5]}). "
            "Refusing to run the encoder-only probe with a partial trunk."
        )
    ignored_unexpected = [
        k for k in unexpected
        if k.startswith("decoder_") or k in {"mask_token"} or k.startswith("head")
    ]
    unexpected_encoder = sorted(set(unexpected) - set(ignored_unexpected))
    if unexpected_encoder:
        logger.info(
            "RETFound checkpoint had %d non-loaded non-decoder keys; sample=%s",
            len(unexpected_encoder),
            unexpected_encoder[:5],
        )
    spec = FM_SPECS["retfound_vitl16"]
    return _RETFoundWrapper(model, spec.feature_dim, spec.feature_hw)


def _load_medsam_vitb16() -> nn.Module:
    """Load MedSAM ViT-B image_encoder (neck stripped).

    Source: ``segment_anything.sam_model_registry['vit_b']`` populated with
    the official MedSAM checkpoint ``medsam_vit_b.pth`` (Google-Drive hosted
    by the upstream repo; no stable direct-download URL). The local checkpoint path must
    be supplied via the ``MEDSAM_CHECKPOINT`` env var.

    We strip everything except ``image_encoder`` and bypass the ``neck`` so
    the output is the pre-neck 768-dim patch token map. The abs ``pos_embed``
    (trained for 1024x1024 → 64x64 grid) is bicubically interpolated to the
    14x14 grid produced by 224x224 input.
    """

    from segment_anything import sam_model_registry

    ckpt_path = os.environ.get("MEDSAM_CHECKPOINT")
    if ckpt_path is None:
        raise RuntimeError(
            "MedSAM checkpoint path must be supplied via the "
            "'MEDSAM_CHECKPOINT' environment variable. Download "
            "'medsam_vit_b.pth' from "
            "https://github.com/bowang-lab/MedSAM (Google-Drive link in "
            "their README) and point MEDSAM_CHECKPOINT at it."
        )
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    image_encoder = sam.image_encoder

    spec = FM_SPECS["medsam_vitb16"]
    return _MedSAMWrapper(image_encoder, spec.feature_dim, spec.feature_hw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_encoder(fm_id: str) -> tuple[nn.Module, EncoderSpec]:
    """Load a frozen FM encoder from its documented hub/checkpoint path.

    Parameters
    ----------
    fm_id:
        One of the ``FM_SPECS`` identifiers.

    Returns
    -------
    (model, spec):
        ``model`` is already in ``eval()`` mode with all parameters frozen.
        The caller is responsible for moving it to a device.
    """

    if fm_id not in FM_SPECS:
        raise KeyError(
            f"Unknown fm_id {fm_id!r}. Supported: {sorted(FM_SPECS.keys())}"
        )
    spec = FM_SPECS[fm_id]

    if fm_id.startswith("dinov2_"):
        model = _load_dinov2(fm_id)
    elif fm_id == "biomedclip":
        model = _load_biomedclip()
    elif fm_id == "clip_vitl14":
        model = _load_clip_vitl14()
    elif fm_id == "mae_vitb16":
        model = _load_mae_vitb16()
    elif fm_id == "deit_vitb16":
        model = _load_deit_vitb16()
    elif fm_id == "clip_vitb16":
        model = _load_clip_vitb16()
    elif fm_id == "clip_vitb32":
        model = _load_clip_vitb32()
    elif fm_id == "retfound_vitl16":
        model = _load_retfound_vitl16()
    elif fm_id == "medsam_vitb16":
        model = _load_medsam_vitb16()
    else:
        model = _load_torchvision_cnn(fm_id)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    logger.info("Built encoder fm_id=%s feature_dim=%d", fm_id, spec.feature_dim)
    return model, spec


@torch.no_grad()
def extract_features(
    model: nn.Module, fm_id: str, x: torch.Tensor
) -> torch.Tensor:
    """Run the encoder on ``x`` and return ``[B, D, h, w]`` dense features.

    ``x`` must already be normalised with the per-FM mean/std returned by
    :func:`get_normalisation`. No pooling is applied.
    """

    if fm_id not in FM_SPECS:
        raise KeyError(f"Unknown fm_id {fm_id!r}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            f"Expected input of shape [B, 3, H, W]; got {tuple(x.shape)}"
        )

    spec = FM_SPECS[fm_id]
    feat = model(x)

    expected_shape = (x.shape[0], spec.feature_dim, *spec.feature_hw)
    if tuple(feat.shape) != expected_shape:
        raise RuntimeError(
            f"{fm_id}: extract_features produced {tuple(feat.shape)}, "
            f"expected {expected_shape}"
        )
    return feat


def get_normalisation(fm_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, std)`` as 1-D tensors of length 3 for ``fm_id``."""

    if fm_id not in FM_SPECS:
        raise KeyError(f"Unknown fm_id {fm_id!r}")
    spec = FM_SPECS[fm_id]
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32)
    std = torch.tensor(spec.norm_std, dtype=torch.float32)
    return mean, std


__all__ = [
    "EncoderSpec",
    "FM_SPECS",
    "build_encoder",
    "extract_features",
    "get_normalisation",
]
