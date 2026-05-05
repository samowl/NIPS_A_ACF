# Derived Trace Resource Data Card

## Resource Type

Derived trace resource for an evaluation audit. This is not a new raw medical-image dataset.

## Included Fields

- Per-case Dice vectors for released model traces.
- Task, bundle, seed, hashed case identifiers, and non-identifying split keys needed for deterministic alignment and released sensitivity checks.
- Same/cross summary JSONs, subject-level summaries, and selected diagnostic aggregate JSONs.
- SHA-256 hashes and manifest metadata.

## Upstream Data

The traces are derived from public or access-controlled medical segmentation benchmarks, including Kvasir-SEG, ACDC, BraTS 2024 GLI, and RIGA+. Raw images and masks are not redistributed here.

## Intended Use

The resource is intended for trace-level recomputation of the paper's co-failure audit statistics and for inspecting the released evaluation protocol.

## Prohibited or Unsupported Use

The resource must not be used for re-identification, patient profiling, protected-attribute inference, clinical diagnosis, clinical deployment decisions, or claims about patient safety. It also does not support full raw-data retraining unless users separately obtain upstream datasets and checkpoints.

## Privacy Considerations

Identifiers are retained only where needed for trace alignment and subject-level resampling. RIGA alignment keys are released as deterministic hashed identifiers with non-identifying source/task labels; raw upstream folder-name components are not needed for the analysis and are not redistributed as case identifiers. These identifiers are not validated demographic annotations and must not be used for subgroup, gender, or individual-level analysis.

## Access Restrictions

Upstream datasets and checkpoints remain subject to their original licenses, research-use terms, and data-use agreements. The derived scalar traces and authored code are distributed only within those constraints.

## Croissant Metadata

A Croissant JSON file describing this derived trace artifact is provided as an OpenReview dataset-metadata fallback. It includes the required Responsible AI metadata fields for intended use, limitations, sensitive-information scope, data collection, labels, preprocessing, and maintenance. It describes the trace/code artifact and should not be read as a claim that this submission introduces a new raw dataset.
