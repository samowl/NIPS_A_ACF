# fmpool code snapshot

This directory contains the code snapshot used to analyse the released
per-case Dice traces for the NeurIPS 2026 Evaluations & Datasets submission.

The released artifact includes the scalar per-case Dice JSONs under
`../results/_merged/`. The most important artifact-verification command is therefore the
analysis regeneration. The lightweight dependency file is CPU-only (`numpy` and
`scipy`); raw-data training dependencies are kept in
`requirements-training.txt`. In a fresh environment, install the CPU-only
dependencies first:

```bash
python3 -m venv .venv-review
. .venv-review/bin/activate
python3 -m pip install -r code/requirements-review.txt
```

Then run the primary scorer. By default it filters to the paper's 11-FM
generic source pool; extra same-family cross-checkpoint traces in
`results/_merged/per_case_dice/` are consumed only by
`compute_cross_checkpoint_family.py`.

```bash
OUT_DIR="$(mktemp -d)"
PYTHONPATH=code/src python3 code/scripts/compute_all.py \
  --results-root results/_merged \
  --out "$OUT_DIR"
```

Use `--out results/_merged` only when intentionally refreshing the bundled
generated JSONs and their integrity hashes.

The script emits `within_cross/`, `m_eff/`, `subject_level/`,
`calibration_split/`, `paper_table.json`, and `provenance_summary.json`.
The regenerated expected primary gaps are:

```text
kvasir=0.260642, acdc_lv=0.320525, brats_wt=0.315171,
riga_cup=0.473153, riga_disc=0.637896
```

The appendix estimator-sensitivity artefact is regenerated separately from the
same released traces:

```bash
OUT_DIR="$(mktemp -d)"
PYTHONPATH=code/src python3 code/scripts/compute_audit_sensitivity.py \
  --results-root results/_merged \
  --out "$OUT_DIR/audit_sensitivity.json" \
  --n-perm 1000 --seed 0
```

The same-family cross-checkpoint diagnostic separates the same-FM seed floor
from distinct-checkpoint same-family pairs:

```bash
PYTHONPATH=code/src python3 code/scripts/compute_cross_checkpoint_family.py \
  --results-root results/_merged \
  --out "$OUT_DIR/cross_checkpoint_family.json"
```

The combined smoke test runs all artifact-verification scorers and compares their stable
JSON outputs to the bundled summaries:

```bash
PYTHONPATH=code/src python3 code/scripts/smoke_artifact.py
```

## Code Contents

- `src/fmpool/estimators.py`: Dice, within/cross Pearson, functional floor,
  case/bootstrap estimators, subject-cluster bootstrap, and `M_eff`.
- `scripts/compute_all.py`: single analysis driver for released JSON traces.
- `scripts/smoke_artifact.py`: one-command smoke test for the released
  summary artifact.
- `scripts/compute_audit_sensitivity.py`: CPU-only functional-floor, ICC,
  leave-one-out, and permutation negative-control scorer for the appendix.
- `scripts/compute_cross_checkpoint_family.py`: CPU-only decomposition of
  same-FM seed, same-family cross-checkpoint, and cross-family correlations.
- `scripts/train_*.py`, `scripts/extract_features.py`, `scripts/run_worker*.sh`:
  training and matrix-launch scripts used to produce the per-case traces.
- `configs/paths.yaml`: local dataset/cache path template.
- `docs/`: specification and dataset/encoder provenance notes.

## Training From Raw Data

Raw medical images and labels are not redistributed. To rerun training, obtain
the source datasets from their distributors, edit `configs/paths.yaml`, then run
the relevant extractor/trainer scripts. A single-cell frozen-feature run has
the shape:

```bash
PYTHONPATH=code/src python3 code/scripts/extract_features.py \
  --task riga_cup \
  --fm dinov2_vitb14 \
  --data-root /path/to/raw-data-root \
  --cache-root results/feature_cache

PYTHONPATH=code/src python3 code/scripts/train_head_cached.py \
  --task riga_cup \
  --fm dinov2_vitb14 \
  --seed 42 \
  --cache-root results/feature_cache \
  --out results/_rerun/per_case_dice \
  --checkpoint-dir results/_rerun/checkpoints
```

Matrix job lists are in `jobs_m*.txt`; the corresponding runners are
`scripts/run_worker*.sh`. They are included for provenance, not as a requirement
for artifact verification.

For optional nnU-Net v2 complements, install `code/requirements-3d-nnunet.txt`
in the active environment and run the no-training trainer-discovery setup once:

```bash
PYTHONPATH=code/src python3 code/scripts/install_nnunet_seeded_trainers.py
```

This writes a small compatibility module into the installed `nnunetv2`
trainer tree so official `nnUNetv2_train -tr
nnUNetTrainer_{100,1000}epochs_Seed*` lookup can resolve the seeded trainers.
It also provides the 1000-epoch length preset used by the released complement
summaries under `results/_merged/nnunet/`.

## Released JSON Contract

Primary per-case Dice JSONs carry `task`, `fm`, `seed`, `test_ids`, and
`per_case_dice`; M15 fold-reslicing traces carry `val_ids` for held-out fold
alignment. The analysis outputs add `schema_version`,
`generated_at`, `commit_sha`, and `source_files` so each summary row can be
traced back to the exact input JSONs.
`results/_merged/diagnostics/audit_sensitivity.json` is derived from those same
per-case traces and does not require raw images or GPU training.

The regenerated primary rows currently cover:

- `kvasir`: 120 test images, 11 post-floor FMs.
- `acdc_lv`: 361 LV-positive slices from 20 patients, 10 post-floor FMs.
- `brats_wt`: 3,228 slices from 324 subjects, 10 post-floor FMs / 40 members.
- `riga_cup`: 95 images, 9 post-floor FMs.
- `riga_disc`: 95 images, 11 post-floor FMs / 44 members.
