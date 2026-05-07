# Derived Trace Resource Data Card

## Resource Type

Derived trace resource for an evaluation audit. This is not a new raw medical-image dataset.

## Included Fields

- Per-case Dice vectors for released model traces.
- Task, bundle, seed, hashed case identifiers, and non-identifying split keys needed for deterministic alignment and released sensitivity checks.
- Same/cross summary JSONs, subject-level summaries, and selected diagnostic aggregate JSONs.
- SHA-256 hashes and manifest metadata.

## Upstream Data and Access Procedures

The traces are derived from public or access-controlled medical segmentation benchmarks. Raw images and masks are not redistributed here.

| Dataset | License / DUA | Access page |
|---|---|---|
| Kvasir-SEG | CC BY 4.0 | https://datasets.simula.no/kvasir-seg/ |
| ACDC | CC BY-NC-SA 4.0 (research) | https://www.creatis.insa-lyon.fr/Challenge/acdc/ |
| BraTS 2024 GLI | Challenge DUA | https://www.synapse.org/brats2024 |
| RIGA+ (BinRushed/MESSIDOR) | Per-source license | https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z |
| ISIC 2018 | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/ |
| MSD (Spleen, Heart, Hippocampus, Prostate) | CC BY-SA 4.0 | http://medicaldecathlon.com/ |

Pretrained encoder licenses (per `code/docs/NEW_ENCODERS.md`): DINOv2 (Apache-2.0), BiomedCLIP (MIT), MedSAM (Apache-2.0), CLIP/open_clip (MIT + OpenAI weight terms), MAE (CC BY-NC 4.0; research-only), RETFound (CC BY-NC 4.0), ConvNeXt/EfficientNet/ResNet/DeiT (BSD/MIT/Apache as per torchvision/timm).

Derived scalar traces inherit the most restrictive upstream license that applies to the encoder used and the dataset used (e.g., MAE-derived or RETFound-derived traces are CC BY-NC 4.0 research-only).

## PHI Handling

Released JSONs contain only per-case Dice scalars and deterministic hashed alignment IDs (16-hex SHA-256 prefixed by source token, e.g., `riga_magrabia_<hash>`). No DICOM headers, no demographic attributes, no raw filenames, no pixel-level outputs are released. Spot-check command: `grep -r "patient\|name=\|/data/\|P0[0-9][0-9]" results/_merged/per_case_dice/` returns zero hits.

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
