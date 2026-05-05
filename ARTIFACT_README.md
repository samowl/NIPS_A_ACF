# Artifact README

## Purpose

Trace-level reproduction of the co-failure audit in "Auditing Correlated Failures in Frozen-Feature Pretrained-Encoder Pools for Medical Segmentation."

## What is included

- Primary per-case Dice traces.
- Same/cross summaries.
- Subject-level sensitivity summaries.
- Diagnostic aggregate JSONs.
- Scorer and table scripts.
- Environment and requirements files.
- SHA-256 manifest.
- Data-card documentation for the derived trace resource.

## What is not included

- Raw medical images.
- Raw masks.
- Full prediction caches.
- Upstream checkpoints.
- End-to-end retraining outputs for restricted datasets.

## Reproduce primary Table 1 and released summaries

Command:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=code/src python3 code/scripts/verify_artifact.py
```

Expected output:

- Recomputed primary same/cross summaries under `results/_merged/`.
- Verification that the released summary JSONs can be parsed and regenerated from the released trace artifacts.

Expected checksum:

```bash
sha256sum -c metadata/SHA256SUMS
```

## Reproduce Table 2 diagnostics

Command:

```bash
OUT_DIR="$(mktemp -d)"
PYTHONPATH=code/src python3 code/scripts/compute_audit_sensitivity.py \
  --results-root results/_merged \
  --out "$OUT_DIR/audit_sensitivity.json" \
  --n-perm 1000 --seed 0
PYTHONPATH=code/src python3 code/scripts/compute_failure_event_summary.py \
  --results-root results/_merged \
  --out "$OUT_DIR/failure_event_summary.json" \
  --n-boot 1000 --seed 0
PYTHONPATH=code/src python3 code/scripts/compute_quality_controlled_pairs.py \
  --results-root results/_merged \
  --out "$OUT_DIR/quality_controlled_pairs.json" \
  --n-boot 1000 --seed 0
PYTHONPATH=code/src python3 code/scripts/compute_item_difficulty_robustness.py \
  --results-root results/_merged \
  --out "$OUT_DIR/item_difficulty_robustness.json" \
  --n-boot 2000 --seed 0
PYTHONPATH=code/src python3 code/scripts/compute_cross_checkpoint_family.py \
  --results-root results/_merged \
  --out "$OUT_DIR/cross_checkpoint_family.json"
```

Expected output:

- Diagnostic JSONs under the temporary output directory named by `$OUT_DIR`.

## Data governance

The artifact releases derived scalar traces and aggregate summaries only. Raw upstream datasets remain governed by their original licenses, terms, and access procedures. Users who want raw-data retraining must separately obtain each upstream dataset and checkpoint under its original terms.

## Privacy and ethics

No re-identification, no patient profiling, and no clinical deployment claims are supported. Released IDs are retained only to align traces and reproduce subject-level resampling where applicable.
