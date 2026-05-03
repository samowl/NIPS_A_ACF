"""MSD Task09_Spleen central-axial-slice loader (task ``msd_spleen``).

Dataset: 41 imagesTr 3D CT volumes. Binary mask = ``label > 0``.
"""
from __future__ import annotations

from fmpool.datasets._msd_common import MSDCenterSliceDataset


class MSDSpleenDataset(MSDCenterSliceDataset):
    """MSD Task09_Spleen binary spleen segmentation (CT)."""

    task_key = "msd_spleen"
    split_task_name = "msd_spleen"
    modality_index = None
    provenance_extra = (
        "Task09_Spleen (41 imagesTr 3D CT). Binary foreground = "
        "(label > 0)."
    )
