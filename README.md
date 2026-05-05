# Auditing Correlated Failures in Frozen-Feature Pretrained-Encoder Pools for Medical Segmentation

NeurIPS 2026 Evaluations & Datasets submission artifact.

This repository is the submission artifact for the paper, code, metadata,
and released result JSONs. The five primary task-row numerical claims in the
current paper are driven by JSON artifacts under `results/_merged/` and by the
analysis script `code/scripts/compute_all.py`; held-out stress-test claims are
supported by selected aggregate diagnostic JSONs, while raw held-out prediction
arrays and full caches are not redistributed in this artifact.

## Headline Numbers

Source: `results/_merged/paper_table.json` and `results/_merged/m_eff/*.json`.
`M_eff` is a tau=0.85 failure-indicator equicorrelation diagnostic, not a
literal independent-member count or standalone pool-selection objective.

| Task | Filtered functional pool | Test units | Gap Delta | Diverse M=4 M_eff (diagnostic) |
|---|---:|---:|---:|---:|
| Kvasir polyp | 11 bundles / 44 members | 120 images | 0.261 | 2.05 |
| ACDC LV | 10 bundles / 40 members | 361 slices / 20 patients | 0.321 | 2.14 |
| BraTS foreground | 10 bundles / 40 members | 3,228 slices / 324 subjects | 0.315 | 2.24 |
| RIGA Cup | 9 bundles / 36 members | 95 images | 0.473 | 2.43 |
| RIGA Disc | 11 bundles / 44 members | 95 images | 0.638 | 2.97 |

Additional audit outputs:

- `results/_merged/subject_level/`: ACDC/BraTS patient-or-subject cluster checks.
- `results/_merged/calibration_split/`: split-half sensitivity only; not the
  primary protocol.
- `results/_merged/nnunet/`: seeded nnU-Net v2 2D/3D task-level
  complements used as limited-scope diagnostics; these use custom
  seed-injection trainers and are not official nnU-Net benchmark results or
  case-identical baselines for the primary frozen-encoder rows.
- `results/_merged/per_case_dice_m14/`: Dice+BCE+augmentation frozen-encoder
  loss/augmentation sensitivity traces for RIGA Cup plus task-level extensions
  on RIGA Disc, ACDC LV, and a staged 400-image ISIC-2018 diagnostic split.
- `results/_merged/diagnostics/`: selected aggregate JSONs for secondary
  pixel-error, matched-lift, architecture, difficulty, held-out, MedSAM-probe,
  quality-controlled pair regression, item-difficulty, binary failure-event,
  distribution-shift, tau-sweep, threshold, split-reslicing, and
  estimator-sensitivity diagnostics.
- `results/_merged/per_case_dice_heldout_11fm/`: released held-out stress-test
  per-case Dice traces consumed by `compute_heldout_11fm_summary.py`.
- `results/_merged/provenance_summary.json`: per-task source-file counts and
  filtered functional pools.

## Directory Layout

```
.
├── README.md
├── ARTIFACT_README.md
├── DATA_CARD.md
├── environment.yml
├── requirements.txt
├── LICENSE
├── paper/
│   ├── main_ED9.pdf
│   ├── main_ED9.tex
│   ├── appendix_ED9.tex
│   ├── checklist.tex
│   ├── references.bib
│   └── neurips_2026.sty
├── figures/
│   ├── fig1_hero.pdf
│   ├── fig_adaptation.pdf
│   └── fig_phase3.pdf
├── code/
│   ├── README.md
│   ├── requirements-repro.txt
│   ├── requirements-training.txt
│   ├── requirements-3d-nnunet.txt
│   ├── src/fmpool/
│   ├── scripts/
│   ├── configs/paths.yaml
│   └── docs/
├── results/_merged/
│   ├── paper_table.json
│   ├── provenance_summary.json
│   ├── within_cross/
│   ├── m_eff/
│   ├── subject_level/
│   ├── calibration_split/
│   ├── per_case_dice/
│   ├── per_case_dice_heldout_11fm/
│   ├── per_case_dice_adapted/
│   ├── nnunet/
│   ├── diagnostics/
│   └── additional appendix trace directories
├── metadata/
│   ├── MANIFEST.md
│   └── SHA256SUMS
```

## Reproducing the Released Summaries

Single-command artifact verification:

```bash
PYTHONPATH=code/src python3 code/scripts/verify_artifact.py
```

The bundle ships the per-case Dice JSONs consumed by the paper. The primary
analysis scorer is pinned by default to the paper's 11-bundle generic source pool;
additional same-family cross-checkpoint traces under `per_case_dice/` are used
only by their dedicated appendix diagnostic scorer. In a fresh
environment, install the lightweight CPU-only dependencies first:

```bash
python3 -m venv .venv-review
. .venv-review/bin/activate
python3 -m pip install -r code/requirements-repro.txt
```

Then regenerate the merged summaries from the released traces:

```bash
OUT_DIR="$(mktemp -d)"
PYTHONPATH=code/src python3 code/scripts/compute_all.py \
  --results-root results/_merged \
  --out "$OUT_DIR"
```

Use `--out results/_merged` only when intentionally refreshing the bundled
generated JSONs and their integrity hashes.

This regenerates:

- `within_cross/<task>.json`
- `m_eff/<task>.json`
- `subject_level/{acdc_lv,brats_wt}.json`
- `calibration_split/<task>.json`
- `paper_table.json`
- `provenance_summary.json`

The appendix UNet-skip diagnostic table is recomputed separately from the
released two-seed traces:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_m4_table.py \
  --results-root results/_merged \
  --out "$OUT_DIR/m4_unet_summary.json"
```

The BraTS appendix train-size and multimodal RGB diagnostics are recomputed
from their released traces with:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_late_brats_summaries.py \
  --results-root results/_merged \
  --out "$OUT_DIR/late_brats_summaries.json"
```

The loss/augmentation sensitivity appendix diagnostic is recomputed from
released traces task-by-task. The `m14_clinical_summary.json` file is the
RIGA Cup summary; task-specific summaries are also released for RIGA
Disc, ACDC LV, and the staged ISIC-2018 diagnostic split:

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

The estimator-sensitivity appendix diagnostic is recomputed from the same
released per-case traces with:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_audit_sensitivity.py \
  --results-root results/_merged \
  --out "$OUT_DIR/audit_sensitivity.json" \
  --n-perm 1000 --seed 0
```

The same-family cross-checkpoint diagnostic, which separates same-bundle decoder
seeds from distinct checkpoints within broad upstream families, is recomputed
with:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_cross_checkpoint_family.py \
  --results-root results/_merged \
  --out "$OUT_DIR/cross_checkpoint_family.json"
```

The quality-controlled pairwise-correlation diagnostic controls pairwise
Pearson correlations for pair mean Dice, absolute mean-Dice difference, average
per-case variance, and absolute variance difference:

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

The five-task binary failure-event diagnostic is recomputed by:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_failure_event_summary.py \
  --results-root results/_merged \
  --out "$OUT_DIR/failure_event_summary.json" \
  --n-boot 1000 --seed 0
```

For a single artifact verification covering the primary scorer and released
appendix summary scorers, run:

```bash
PYTHONPATH=code/src python3 code/scripts/verify_artifact.py
```

Training from raw images requires obtaining the upstream datasets under their
own licences/DUAs and editing `code/configs/paths.yaml`. Raw medical images,
voxel masks, FM checkpoints, feature caches, and large prediction caches are not
redistributed in this artifact.

## Responsible Use and Privacy

This artifact is for retrospective evaluation research. It releases derived
per-case Dice traces, aggregate summaries, metadata, hashes, and scorer code;
it does not release raw medical images, voxel labels, upstream checkpoints, full
prediction caches, or patient-facing services. Some trace records retain
de-identified benchmark alignment IDs or source labels, such as RIGA source
paths and ACDC/BraTS subject or slice IDs, so users can verify split alignment
without reconstructing raw images. These IDs must not be used for
re-identification, patient profiling, or clinical triage. Kvasir traces expose
image-level file IDs only; the artifact does not contain patient/video grouping
IDs for patient-level duplicate clustering. BraTS `brats_wt` uses 3,228 test
slices from 324 subjects because three held-out subjects contribute fewer than
ten eligible foreground-positive slices after the 64-pixel filter. Any raw-data
reproduction requires obtaining the original datasets and model checkpoints
under their upstream licences, research terms, or DUAs.

## Integrity

Verify released artifacts with the SHA-256 manifest. The manifest intentionally
omits itself, local `reports/**` files, any local `code/results/**` cache tree
if present, and Python bytecode:

```bash
sha256sum -c metadata/SHA256SUMS
```

`metadata/MANIFEST.md` explains what is included, what is excluded, and which
JSON files drive the paper tables.

## Claim-to-Artifact Map

| Paper claim/readout | Released source artifacts | Recompute command |
|---|---|---|
| Primary same-vs-cross Table 1 gaps, CIs, filtered functional pools | `results/_merged/paper_table.json`, `results/_merged/within_cross/*.json`, `results/_merged/per_case_dice/*.json` | `code/scripts/compute_all.py` |
| ACDC/BraTS subject-level sensitivity and split-half sensitivity | `results/_merged/subject_level/*.json`, `results/_merged/calibration_split/*.json` | `code/scripts/compute_all.py` |
| Functional-floor sweep, ICC(A,1), leave-one-out, permutation stress test | `results/_merged/diagnostics/audit_sensitivity.json` | `code/scripts/compute_audit_sensitivity.py` |
| Quality-controlled pair regression in the main robustness table | `results/_merged/diagnostics/quality_controlled_pairs.json` | `code/scripts/compute_quality_controlled_pairs.py` |
| Cross-fitted item-difficulty residualisation | `results/_merged/diagnostics/item_difficulty_robustness.json` | `code/scripts/compute_item_difficulty_robustness.py` |
| Binary failure-event same/cross diagnostic | `results/_merged/diagnostics/failure_event_summary.json` | `code/scripts/compute_failure_event_summary.py` |
| Same-family cross-checkpoint decomposition | `results/_merged/diagnostics/cross_checkpoint_family.json` and expanded `per_case_dice/` traces | `code/scripts/compute_cross_checkpoint_family.py` |
| Distribution-shift diagnostic | `results/_merged/diagnostics/distshift_riga_messidor.json`, `results/_merged/per_case_dice/riga_cup_ood_messidor/*.json` | `code/scripts/compute_distshift.py` |
| Tau-sweep and RIGA threshold diagnostics | `results/_merged/diagnostics/tau_sweep.json`, `results/_merged/diagnostics/threshold_table_riga_cup.json` | `code/scripts/compute_tau_sweep.py`; `code/scripts/compute_threshold_table.py` |
| Fold-reslicing diagnostic | `results/_merged/diagnostics/m15_cv_summary.json`, `results/_merged/per_case_dice_m15/**/*.json` | `code/scripts/compute_m15_cv_summary.py` |
| Held-out 11-FM stress test | `results/_merged/diagnostics/heldout_11fm_summary.json`, `results/_merged/per_case_dice_heldout_11fm/**/*.json` | `code/scripts/compute_heldout_11fm_summary.py` |
| Pixel-error and matched-lift diagnostics | `results/_merged/diagnostics/pixel_error/*.json`, `results/_merged/diagnostics/matched_lift/*.json` | Released aggregate JSONs only; raw masks/caches are not redistributed |
| Same-backbone and appendix scope diagnostics | Appendix trace directories and bundled summary JSONs under `results/_merged/` | Dedicated appendix scripts listed above |
| nnU-Net limited-scope complements | `results/_merged/nnunet/*.json` | Released task-level JSONs only; optional seeded trainers are documented under `code/scripts/` |

## License

Apache-2.0 for the released code and metadata in this artifact. Source datasets,
FM checkpoints, and derived traces retain their upstream licence constraints.
The artifact does not redistribute raw medical images, voxel labels, full
prediction caches, or upstream model checkpoints.
