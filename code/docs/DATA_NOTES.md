# DATA_NOTES

This bundle redistributes model-derived scalar per-case Dice traces and summary
JSONs only. It does not redistribute raw medical images, voxel labels, FM
checkpoints, feature caches, or large prediction caches.

## Dataset Notes

| Dataset | Paper use | Redistribution note |
|---|---|---|
| Kvasir-SEG | Polyp binary segmentation, 120-image test split | Raw images/masks remain with the upstream Kvasir-SEG release. |
| ACDC | LV binary segmentation, public training patients 001--100, fold-0 held-out patients | Raw cine-MRI and masks are not mirrored. Released traces contain derived Dice only. |
| BraTS 2024 GLI | Non-background foreground (`seg > 0`) 2D FLAIR axial derivative, 3,228 test slices from 324 subjects; legacy task id `brats_wt` | BraTS raw images and voxel labels are DUA-restricted and are not redistributed. |
| RIGA+ | Cup/disc majority-vote segmentation; Magrabi Eye Center/Magrabia source held-out test domain | Raw fundus images/rater masks are not mirrored. |
| Held-out / matrix tasks | ISIC/MSD/RIGA OOD/BraTS variants in appendix or robustness matrices | Included only as derived traces where present in `results/_merged/`; source datasets must be obtained separately. |

## Split and Unit Conventions

- `test_ids` in each per-case JSON are the alignment authority used by
  `code/scripts/compute_all.py`.
- ACDC and BraTS have higher-level units embedded in `test_ids`; the scorer
  derives patient/subject IDs for `subject_level/*.json`.
- RIGA and Kvasir are analysed at image level.

## Functional Floor

The Dice >= 0.30 functional floor is applied downstream by the estimator, not by
deleting input traces. Consequently, source-file counts can exceed the analytic
functional pool reported in the paper.

Current retained functional pools:

- Kvasir: 11 FMs / 44 members.
- ACDC LV: 10 FMs / 40 members.
- BraTS foreground: 44 source JSONs across 11 FMs; 10 FMs / 40 members after the Dice >= 0.30 floor.
- RIGA Cup: 9 FMs / 36 members.
- RIGA Disc: 11 FMs / 44 members.

## Verification

Use `metadata/SHA256SUMS` at the bundle root to verify released artefacts. Use
`results/_merged/provenance_summary.json` and each output JSON's `source_files`
field to trace summary statistics back to per-case Dice files.
