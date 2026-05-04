# FMPool Clean Submission SPEC

Status: artifact truth source for the released code and result JSONs.
Last revised: 2026-04-30.

Every numerical paper claim should be traceable to a JSON file under
`results/_merged/` and to `code/scripts/compute_all.py` or a documented
appendix-only artefact. No raw medical images, labels, feature caches, or FM
checkpoints are redistributed in the supplementary artifact.

## Primary Tasks

| Task | Released test unit | Higher-level unit | Current test count |
|---|---|---:|---:|
| `kvasir` | endoscopic image | none | 120 images |
| `acdc_lv` | 2D LV-positive cardiac slice | ACDC patient | 361 slices / 20 patients |
| `brats_wt` | 2D FLAIR axial `seg > 0` foreground slice | BraTS subject | 3,228 slices / 324 subjects |
| `riga_cup` | fundus image | none | 95 images |
| `riga_disc` | fundus image | none | 95 images |

Dataset split notes:

- Kvasir-SEG uses the loader's deterministic 880/120 split when no upstream
  `train.txt`/`val.txt` is present.
- ACDC uses public training patients 001--100, fold-0 of a patient-level
  5-fold split, and LV binary masks.
- BraTS foreground is a BraTS 2024 GLI 2D derivative. The legacy task id is
  `brats_wt`, but the binary target is `seg > 0` and includes the GLI
  resection-cavity label, so it is not the official 3D BraTS WT target.
- RIGA+ uses majority-vote cup/disc masks with `sum(rater_i) >= 4`; the
  Magrabi Eye Center/Magrabia source is the held-out test domain.

## Foundation Models and Functional Floor

The regenerated primary source pool contains up to 11 generic frozen FMs:
DINOv2-B/14, DINOv2-S/14, BiomedCLIP, CLIP ViT-B/16, CLIP ViT-L/14,
ConvNeXt-T, EfficientNet-B0, ResNet-50, ResNet-18, MAE ViT-B/16, and
DeiT ViT-B/16. BraTS foreground now has the completed 11-FM source pool;
ResNet-18 falls below the Dice floor, leaving 10 retained FMs / 40 seed
members. RIGA Disc also uses the completed 11-FM source pool.

The paper applies a Dice >= 0.30 functional floor downstream in the estimator
with `scope="per_fm"`. Input `source_files` may therefore include more completed
JSONs than the retained functional pool.

Current retained primary pools:

| Task | Source JSONs | Analytic pool |
|---|---:|---|
| `kvasir` | 44 | 11-FM x 4-seed (44 members) |
| `acdc_lv` | 44 | 10-FM x 4-seed (40 members) |
| `brats_wt` | 44 | 10-FM x 4-seed (40 members) |
| `riga_cup` | 44 | 9-FM x 4-seed (36 members) |
| `riga_disc` | 44 | 11-FM x 4-seed (44 members) |

## Estimators

Estimator implementation: `code/src/fmpool/estimators.py`.

`within_cross_rbar` computes Pearson correlation over aligned per-case Dice
arrays. Same-FM different-seed pairs are within-family; cross-FM pairs are
cross-family, except DINOv2-B and DINOv2-S can be grouped as the same family via
`family_pairs_within={frozenset({"dinov2_vitb14","dinov2_vits14"})}`.
Low-standard-deviation pairs return NaN and are excluded from the mean.

`case_bootstrap` resamples test cases with replacement. `hierarchical_bootstrap`
resamples FM families and seeds. `subject_cluster_bootstrap` resamples ACDC
patients or BraTS subjects with replacement while retaining all slices in each
sampled unit.

`compute_m_eff` uses:

```
M_eff = M / (1 + (M - 1) * rho_fail)
```

where `rho_fail` is the mean pairwise Pearson correlation of binary failure
indicators `(Dice < tau)` at `tau=0.85`. Undefined zero-variance failure
correlations are not imputed. Diverse-pool summaries enumerate every 4-FM
composition of the retained functional pool and every available one-seed-per-FM
tuple.

## Analysis Outputs

Run:

```bash
OUT_DIR="$(mktemp -d)"
PYTHONPATH=code/src python3 code/scripts/compute_all.py \
  --results-root results/_merged \
  --out "$OUT_DIR"
```

Use `--out results/_merged` only when intentionally refreshing bundled outputs
and integrity hashes. The read-only smoke path outputs:

- `within_cross/<task>.json`: primary within/cross/gap rows and CIs.
- `m_eff/<task>.json`: mono-pool and diverse 4-FM `M_eff` summaries.
- `subject_level/{acdc_lv,brats_wt}.json`: cluster-bootstrap and
  patient/subject-aggregated sensitivity.
- `calibration_split/<task>.json`: deterministic split-half sensitivity; not
  the primary protocol.
- `paper_table.json`: table-ready primary rows.
- `provenance_summary.json`: task-level source counts and pool metadata.

The appendix estimator-sensitivity artefact is regenerated independently:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_audit_sensitivity.py \
  --results-root results/_merged \
  --out "$OUT_DIR/audit_sensitivity.json" \
  --n-perm 1000 --seed 0
```

It emits `audit_sensitivity.json`, containing a functional-floor sweep, Pearson
vs. ICC(A,1), leave-one-FM-out, leave-one-task-out, and random family-label
permutation checks derived only from released per-case Dice traces.

## Reproducibility Invariants

- Canonical `compute_all.py` summary JSONs have `schema_version`,
  `generated_at`, `commit_sha`, and `source_files`. Appendix-only summary
  artefacts use the lighter schemas documented by their scorer scripts and
  retain explicit source roots or source-file lists where applicable.
- Primary per-case JSONs retain `test_ids`; M15 fold-reslicing traces retain
  `val_ids`; the scorer validates alignment before aggregation.
- Bootstrap seeds are fixed at 0 in `compute_all.py`.
- `metadata/SHA256SUMS` verifies the released artifact files.
- Optional nnU-Net complements use `nnunetv2==2.7.0`. Before launching seeded
  nnU-Net jobs, run `PYTHONPATH=code/src python3
  code/scripts/install_nnunet_seeded_trainers.py` from the bundle root. This
  installs a small compatibility module into the active nnU-Net package so
  official trainer discovery can resolve the documented seeded trainer classes,
  including the 1000-epoch preset used by the released complement summaries.
