# Artifact Manifest

Submission: "Auditing Correlated Failures in Frozen-Feature
Pretrained-Encoder Pools for Medical Segmentation" (NeurIPS 2026 Evaluations &
Datasets).

This manifest describes the submission artifact rooted at this
repository. The artifact is organized around `code/`, `paper/`,
`results/_merged/`, and `metadata/`.

## Included Top-Level Files

| Path | Role |
|---|---|
| `README.md` | Bundle overview, headline numbers, regeneration command. |
| `LICENSE` | Apache-2.0 licence for released code/metadata authored for this bundle. |
| `ARTIFACT_README.md` | Short trace-level reproduction and governance guide. |
| `DATA_CARD.md` | Derived trace resource data card for intended use, non-use, and upstream constraints. |
| `environment.yml` | Conda environment helper for artifact inspection. |
| `requirements.txt` | Root pip requirements shim for lightweight artifact checks. |

## Paper

| Path | Role |
|---|---|
| `paper/main_ED9.pdf` | Built main manuscript. |
| `paper/main_ED9.tex` | Main LaTeX source. |
| `paper/appendix_ED9.tex` | Appendix source. |
| `paper/checklist.tex` | NeurIPS checklist source included by the manuscript. |
| `paper/references.bib` | Bibliography. |
| `paper/neurips_2026.sty` | Venue style file used for local build. |

## Figures

| Path | Role |
|---|---|
| `figures/fig1_hero.pdf` | Figure 1 source included by the LaTeX manuscript. |
| `figures/fig_adaptation.pdf` | Adaptation/control figure source. |
| `figures/fig_phase3.pdf` | Robustness/distribution-shift figure source. |

## Code

| Path | Role |
|---|---|
| `code/src/fmpool/` | Core library: encoders, heads, determinism utilities, estimators. |
| `code/scripts/compute_all.py` | Canonical scorer for released JSON traces. |
| `code/scripts/compute_audit_sensitivity.py` | CPU-only estimator-robustness scorer for functional-floor, ICC, leave-one-out, and permutation checks. |
| `code/scripts/compute_cross_checkpoint_family.py` | CPU-only diagnostic scorer that separates same-bundle seed, same-family cross-checkpoint, and cross-family correlations. |
| `code/scripts/compute_quality_controlled_pairs.py` | CPU-only pair-level diagnostic scorer controlling correlation for marginal quality and variance features. |
| `code/scripts/compute_item_difficulty_robustness.py` | CPU-only all-row item-difficulty residualisation scorer. |
| `code/scripts/compute_failure_event_summary.py` | CPU-only five-task binary failure-event same/cross diagnostic scorer. |
| `code/scripts/compute_distshift.py` | CPU-only RIGA ID/MESSIDOR fixed-pool distribution-shift scorer. |
| `code/scripts/compute_tau_sweep.py` | CPU-only appendix conditional-recovery tau-sweep scorer. |
| `code/scripts/compute_threshold_table.py` | CPU-only RIGA Cup empirical-threshold failure-correlation scorer. |
| `code/scripts/compute_m15_cv_summary.py` | CPU-only single-seed 5-fold split-reslicing scorer. |
| `code/scripts/compute_nnunet_case_identical.py` | CPU-only scorer for the released RIGA case-identical nnU-Net scope check. |
| `code/scripts/compute_m4_table.py` | Appendix UNet-skip table scorer from released two-seed traces. |
| `code/scripts/compute_late_brats_summaries.py` | BraTS train-size and multimodal appendix scorer for released traces. |
| `code/scripts/compute_m14_summary.py` | Loss/augmentation sensitivity scorer for the released task-level traces. |
| `code/scripts/verify_artifact.py` | One-command verification for the released summary artifact. |
| `code/scripts/install_nnunet_seeded_trainers.py` | Optional no-training setup/verification helper that exposes seeded nnU-Net trainers through official nnU-Net discovery. |
| `code/scripts/run_nnunet_case_worker.sh` | Optional raw-data launcher for the RIGA case-identical nnU-Net scope check. |
| `code/scripts/train_*.py`, `code/scripts/extract_features.py` | Training and feature-extraction provenance scripts. |
| `code/jobs_m*.txt`, `code/scripts/run_worker*.sh` | Matrix job lists and worker launch scripts. |
| `code/requirements-repro.txt` | Minimal dependencies for reproducing released JSON summaries. |
| `code/requirements-training.txt` | Additional dependencies for frozen-head training/reproduction from local datasets. |
| `code/requirements-3d-nnunet.txt` | Optional MONAI/nnU-Net dependencies for 3D/full-pipeline complements. |
| `code/configs/paths.yaml` | Local dataset/cache path template. |
| `code/docs/SPEC.md` | Current estimator/data contract. |
| `code/docs/DATA_NOTES.md` | Dataset redistribution and split notes. |
| `code/docs/NEW_ENCODERS.md` | Encoder provenance notes for the added FM backbones. |

## Results

All paper-table primary rows are generated from `results/_merged/`.

| Path | Count | Role |
|---|---:|---|
| `results/_merged/per_case_dice/` | 392 JSONs | Primary and selected appendix/diagnostic per-case Dice traces: 220 primary JSONs plus 120 same-family cross-checkpoint traces, 32 RIGA OOD, and 20 BraTS multimodal JSONs. |
| `results/_merged/within_cross/*.json` | 5 JSONs | Primary within/cross/gap summaries and bootstrap intervals. |
| `results/_merged/m_eff/*.json` | 5 JSONs | Mono-pool and functional diverse 4-bundle `M_eff` summaries, including mean and conservative maximum-`rho_fail` readouts. |
| `results/_merged/subject_level/*.json` | 2 JSONs | ACDC/BraTS patient-or-subject cluster sensitivity. |
| `results/_merged/calibration_split/*.json` | 5 JSONs | Split-half sensitivity only; not the primary protocol. |
| `results/_merged/nnunet/*.json` | 4 JSONs | Released nnU-Net v2 2D/3D complement summaries used as limited-scope evidence, including the RIGA case-identical held-out scope summary. |
| `results/_merged/nnunet_case_riga_2d_100ep/*.json` | 10 JSONs | Per-model RIGA case-identical nnU-Net held-out traces with hashed case IDs, consumed by `compute_nnunet_case_identical.py`. |
| `results/_merged/diagnostics/` | 28 JSONs | Selected aggregate JSONs for secondary pixel-error, matched-lift, architecture, same-family cross-checkpoint, quality-controlled pair regression, item-difficulty, binary failure-event, distribution-shift, tau-sweep, threshold, split-reslicing, held-out, MedSAM-probe, and estimator-sensitivity diagnostics. Raw masks, NPZ traces, and full prediction caches are not redistributed. |
| `results/_merged/per_case_dice_heldout_11fm/` | 220 JSONs | Released held-out stress-test per-case Dice traces consumed by `compute_heldout_11fm_summary.py`. |
| `results/_merged/paper_table.json` | 1 JSON | Consolidated table-ready primary rows. |
| `results/_merged/provenance_summary.json` | 1 JSON | Task-level source counts, case units, cluster units, and analytic pools. |
| `results/_merged/per_case_dice_m14/` | 112 JSONs | Dice+BCE+augmentation loss/augmentation sensitivity traces: RIGA Cup plus task-level extensions on RIGA Disc, ACDC LV, and a local 400-image ISIC-2018 diagnostic split. |
| `results/_merged/m14_clinical_summary.json` and `results/_merged/m14_clinical_summary_{riga_cup,riga_disc,acdc_lv,isic2018}.json` | 5 JSONs | Loss/augmentation task-specific summaries; the RIGA Cup filename is retained for backward-compatible verification. |
| `results/_merged/m14_clinical_extension_summary.json` | 1 JSON | Combined loss/augmentation extension summary over the released task-level diagnostic splits. |
| Additional appendix trace directories under `results/_merged/` | Appendix JSONs | Robustness and scope-analysis traces; adapted traces are limited to the released full-fine-tuning/random-init files. |

Current primary functional pools after the Dice >= 0.30 functional floor:

| Task | Test cases | Higher-level units | Source JSONs | Functional pool |
|---|---:|---:|---:|---|
| `kvasir` | 120 | 120 images | 44 | 11-bundle x 4-seed (44 members) |
| `acdc_lv` | 361 | 20 patients | 44 | 10-bundle x 4-seed (40 members) |
| `brats_wt` | 3,228 | 324 subjects | 44 | 10-bundle x 4-seed (40 members) |
| `riga_cup` | 95 | 95 images | 44 | 9-bundle x 4-seed (36 members) |
| `riga_disc` | 95 | 95 images | 44 | 11-bundle x 4-seed (44 members) |

## Metadata

| Path | Role |
|---|---|
| `metadata/SHA256SUMS` | SHA-256 digests for released artifact files, excluding itself, local reports, local `code/results/**` cache trees, and Python bytecode. |
| `metadata/MANIFEST.md` | This manifest. |
| `ARTIFACT_README.md` | Reviewer-facing reproduction guide. |
| `DATA_CARD.md` | Derived trace resource documentation. |
| `environment.yml`, `requirements.txt` | Lightweight environment entry points. |

Verify integrity from the bundle root. The hash manifest intentionally omits
itself, local `reports/**` files, any local `code/results/**` cache tree if
present, and Python bytecode:

```bash
sha256sum -c metadata/SHA256SUMS
```

## Not Included

| Item | Reason |
|---|---|
| Raw BraTS images and voxel labels | Synapse/BraTS DUA; users must obtain the dataset separately. |
| Raw ACDC/RIGA/Kvasir/ISIC/MSD data | Upstream licences and size; this bundle releases derived traces and loaders/scripts. |
| FM checkpoints | Public upstream distribution via torch.hub, HuggingFace, OpenCLIP, or project-specific sources. |
| Feature caches and raw prediction caches | Large intermediate artifacts; not needed to verify released summary JSONs. |
| Upstream FM checkpoints | Users must obtain checkpoints from their original providers and terms. |
| Author and affiliation metadata | Omitted from the submitted artifact. |
| Non-artifact notes | Excluded from the released archive; not needed to reproduce released summaries. |

## Licence Notes

Released code and metadata authored for this submission bundle are Apache-2.0.
Derived scalar traces are redistributed only to the extent permitted by the
upstream dataset/model terms. RIGA+ is CC BY-NC 4.0 via Deep Blue; Kvasir-SEG
is used via Simula's upstream research/educational terms; ACDC, BraTS, MSD, and
ISIC resources require their challenge/research licences, terms, or DUAs. Model
checkpoints are not redistributed:
DINOv2 and SAM/MedSAM are Apache-2.0, MAE and RETFound checkpoints are
non-commercial/research-use, OpenAI CLIP weights use the OpenAI CLIP licence,
BiomedCLIP follows its model-card terms, and torchvision/open_clip/timm code
retains the upstream package terms.

## Regeneration Contract

From the bundle root:

```bash
PYTHONPATH=code/src python3 code/scripts/verify_artifact.py
```

This single command regenerates and compares the released CPU-only summary
JSONs covered by the verification path. Individual scorer commands are listed below
for targeted inspection:

```bash
OUT_DIR="$(mktemp -d)"
PYTHONPATH=code/src python3 code/scripts/compute_all.py \
  --results-root results/_merged \
  --out "$OUT_DIR"
```

This read-only verification path regenerates the primary summary JSONs from the
released per-case Dice traces without mutating the bundle. Use
`--out results/_merged` only when intentionally refreshing bundled outputs and
integrity hashes. Training from raw images requires local dataset paths and is
outside the minimal artifact-verification path.

The appendix UNet-skip summary is recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_m4_table.py \
  --results-root results/_merged \
  --out "$OUT_DIR/m4_unet_summary.json"
```

The BraTS train-size and multimodal summaries are recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_late_brats_summaries.py \
  --results-root results/_merged \
  --out "$OUT_DIR/late_brats_summaries.json"
```

The loss/augmentation sensitivity summaries are recomputed task-by-task by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_m14_summary.py \
  --results-root results/_merged --task riga_cup \
  --out "$OUT_DIR/m14_clinical_summary_riga_cup.json"
PYTHONPATH=code/src python3 code/scripts/compute_m14_summary.py \
  --results-root results/_merged --task riga_disc \
  --out "$OUT_DIR/m14_clinical_summary_riga_disc.json"
PYTHONPATH=code/src python3 code/scripts/compute_m14_summary.py \
  --results-root results/_merged --task acdc_lv \
  --out "$OUT_DIR/m14_clinical_summary_acdc_lv.json"
PYTHONPATH=code/src python3 code/scripts/compute_m14_summary.py \
  --results-root results/_merged --task isic2018 \
  --out "$OUT_DIR/m14_clinical_summary_isic2018.json"
```

The estimator-sensitivity appendix artifact is recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_audit_sensitivity.py \
  --results-root results/_merged \
  --out "$OUT_DIR/audit_sensitivity.json" \
  --n-perm 1000 --seed 0
```

The quality-controlled pairwise-correlation diagnostic is recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_quality_controlled_pairs.py \
  --results-root results/_merged \
  --out "$OUT_DIR/quality_controlled_pairs.json" \
  --n-boot 1000 --seed 0
```

The all-row cross-fitted item-difficulty residualisation diagnostic is
recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_item_difficulty_robustness.py \
  --results-root results/_merged \
  --out "$OUT_DIR/item_difficulty_robustness.json" \
  --n-boot 2000 --seed 0
```

The binary failure-event diagnostic is recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_failure_event_summary.py \
  --results-root results/_merged \
  --out "$OUT_DIR/failure_event_summary.json" \
  --n-boot 1000 --seed 0
```

The distribution-shift, tau-sweep, threshold, fold-reslicing, and held-out
stress-test diagnostics are recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_distshift.py \
  --results-root results/_merged \
  --out "$OUT_DIR/distshift_riga_messidor.json" \
  --n-boot 500 --seed 0
PYTHONPATH=code/src python3 code/scripts/compute_tau_sweep.py \
  --results-root results/_merged \
  --out "$OUT_DIR/tau_sweep.json"
PYTHONPATH=code/src python3 code/scripts/compute_threshold_table.py \
  --results-root results/_merged \
  --out "$OUT_DIR/threshold_table_riga_cup.json"
PYTHONPATH=code/src python3 code/scripts/compute_m15_cv_summary.py \
  --results-root results/_merged \
  --out "$OUT_DIR/m15_cv_summary.json"
PYTHONPATH=code/src python3 code/scripts/compute_heldout_11fm_summary.py \
  --per-case-root results/_merged/per_case_dice_heldout_11fm \
  --out "$OUT_DIR/heldout_11fm_summary.json"
```
