# New Encoders (2026-04-25 extension)

Five frozen FM encoders added to `src/fmpool/encoders.py`. All share the
same `build_encoder(fm_id) -> (nn.Module, EncoderSpec)` interface as the
original eight; `extract_features(model, fm_id, x)` returns
`[B, D, h, w]` dense features for a `[B, 3, 224, 224]` input.

| `fm_id`            | Family   | Feature dim | Grid    | Norm        | Source                                                                                       |
|--------------------|----------|-------------|---------|-------------|----------------------------------------------------------------------------------------------|
| `mae_vitb16`       | MAE      | 768         | 14×14   | ImageNet    | `timm.create_model('vit_base_patch16_224.mae', pretrained=True)`                             |
| `deit_vitb16`      | DeiT     | 768         | 14×14   | ImageNet    | `timm.create_model('deit_base_patch16_224', pretrained=True)`                                |
| `clip_vitb16`      | CLIP     | 768         | 14×14   | OpenAI CLIP | `open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='openai')`           |
| `retfound_vitl16`  | RETFound | 1024        | 14×14   | ImageNet    | HF `YukunZhou/RETFound_mae_natureCFP` (gated) — checkpoint via `RETFOUND_CFP_CHECKPOINT` env |
| `medsam_vitb16`    | MedSAM   | 768         | 14×14   | ImageNet*   | `segment_anything.sam_model_registry['vit_b']` + `MEDSAM_CHECKPOINT` env (Google-Drive)      |

*MedSAM normalisation: this benchmark uses a common 224-pixel frozen-encoder
probe with ImageNet-style normalisation. Official MedSAM inference instead
resizes to 1024 and feeds the full prompt-conditioned SAM path; the released
MedSAM row is intentionally an encoder-only local probe, not official MedSAM
inference or official MedSAM preprocessing.

## Checkpoint provenance

### `mae_vitb16`
- **Upstream**: `facebookresearch/mae`. Original URL
  `https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth`.
- **Loaded via**: timm hub mirror `vit_base_patch16_224.mae`
  (https://huggingface.co/timm/vit_base_patch16_224.mae). 85.8M params.
- **Architecture**: ViT-B/16, embed_dim=768, depth=12, num_heads=12,
  mlp_ratio=4.
- **License**: CC-BY-NC 4.0 (research-only; matches the rest of the
  fmpool benchmark).
- **SHA256**: not published by upstream. timm hub stores its own copy with
  a content hash visible in the HF UI.

### `deit_vitb16`
- **Upstream**: `facebookresearch/deit`. Official non-distilled checkpoint
  `https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth`
  (filename hash `b5f2ef4d` is included in the URL by upstream; this is
  not a checksum).
- **Loaded via**: `timm.create_model('deit_base_patch16_224', pretrained=True)`.
- **Architecture**: ViT-B/16, embed_dim=768, depth=12, num_heads=12, qkv_bias.
- **License**: Apache-2.0.

### `clip_vitb16`
- **Upstream**: OpenAI CLIP weights, redistributed by
  `mlfoundations/open_clip`.
- **Loaded via**: `open_clip.create_model_and_transforms('ViT-B-16-quickgelu',
  pretrained='openai')`. The `-quickgelu` suffix matches OpenAI's training
  activation (per open_clip README). The visual tower's pre-projection dim is 768.
- **Architecture**: ViT-B/16, 14×14 token grid at 224 input.
- **License**: MIT (open_clip code) / OpenAI CLIP weights (OpenAI license).

### `retfound_vitl16`
- **Upstream**: `rmaphoh/RETFound_MAE` (Zhou et al. *Nature* 2023).
  Architecture matches `models_vit.RETFound_mae`: embed_dim=1024,
  depth=24, num_heads=16, patch_size=16, mlp_ratio=4, qkv_bias=True.
- **Checkpoint**: HuggingFace **gated** repo
  `YukunZhou/RETFound_mae_natureCFP`, file `RETFound_mae_natureCFP.pth`.
- **Loading path**: builds the trunk via `timm.create_model('vit_large_patch16_224',
  pretrained=False, ...)` with RETFound-compatible hyperparameters, then
  loads a gated MAE state-dict with `strict=False` (the checkpoint contains a
  `decoder_*` / `mask_token` block we drop). This path was not materialized in
  the inspected artifact because no RETFound checkpoint/authentication
  was available.
- **Auth**: requires `huggingface-cli login` OR set
  `RETFOUND_CFP_CHECKPOINT=/path/to/RETFound_mae_natureCFP.pth` to load
  from a local file.
- **Scope restriction (per paper SPEC)**: this FM is restricted to
  RIGA / Kvasir / ACDC and **must not** be evaluated on BraTS.
- **License**: CC-BY-NC 4.0 / research-use gated checkpoint terms.

### `medsam_vitb16`
- **Upstream**: `bowang-lab/MedSAM` (Ma et al. *Nature Communications*
  2024). The official `medsam_vit_b.pth` is hosted on Google Drive
  (https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN);
  no stable direct-download URL is published.
- **Loading path**: `segment_anything.sam_model_registry['vit_b']
  (checkpoint=$MEDSAM_CHECKPOINT)` → keep `image_encoder`, drop prompt
  encoder + mask decoder. The pre-trained `pos_embed` (64×64 grid for
  1024×1024 input) is bicubically interpolated to the 14×14 grid produced
  by 224×224 input. The 1×1+3×3 conv `neck` (768→256) is **stripped** so
  the output retains the 768-dim ViT representation, matching the
  paper SPEC and the other ViT-B FMs in the benchmark.
- **Architecture**: ViT-B/16, embed_dim=768, depth=12, num_heads=12,
  global_attn_indexes=(2, 5, 8, 11), patch_size=16.
- **Required env**: `MEDSAM_CHECKPOINT=/path/to/medsam_vit_b.pth`.
- **License**: Apache-2.0.

## Interface contract (unchanged from original 8)

```python
from fmpool.encoders import build_encoder, extract_features, get_normalisation

model, spec = build_encoder("mae_vitb16")
mean, std = get_normalisation("mae_vitb16")
# x: [B, 3, 224, 224], normalised with (mean, std)
feat = extract_features(model, "mae_vitb16", x)  # [B, 768, 14, 14]
```

## Artifact Checks

The supplementary bundle does not include a standalone `tests/` tree. Encoder
registration, metadata, normalisation routing, and checkpoint-gated build paths
can be inspected through `src/fmpool/encoders.py`; raw build/forward checks
require the corresponding upstream packages and gated checkpoints. The released
artifact smoke test focuses on CPU-only scorer reproducibility from bundled
per-case JSON traces rather than network/checkpoint-dependent encoder loading.
