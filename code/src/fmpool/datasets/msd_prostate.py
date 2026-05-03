"""MSD Task05_Prostate central-axial-slice loader (task ``msd_prostate``).

Dataset: 32 imagesTr 4D multi-modal MRI volumes ``[H,W,Z,M]`` with M=2
modalities (0: T2, 1: ADC). We keep modality 0 (T2) only. Labels
{0: background, 1: PZ, 2: TZ}; binary foreground = ``label > 0``
(PZ + TZ merged).
"""
from __future__ import annotations

from fmpool.datasets._msd_common import MSDCenterSliceDataset


class MSDProstateDataset(MSDCenterSliceDataset):
    """MSD Task05_Prostate T2 binary prostate (PZ+TZ) segmentation."""

    task_key = "msd_prostate"
    split_task_name = "msd_prostate"
    modality_index = 0  # T2
    provenance_extra = (
        "Task05_Prostate (32 imagesTr 4D MRI; T2 = modality 0, ADC = "
        "modality 1; we use T2 only). Binary foreground = (label > 0) "
        "merging PZ (1) and TZ (2)."
    )
