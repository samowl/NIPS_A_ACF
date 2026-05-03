"""MSD Task04_Hippocampus central-axial-slice loader (task ``msd_hippocampus``).

Dataset: 260 imagesTr 3D MRI volumes. Labels {0: background, 1: head,
2: body}; binary foreground = ``label > 0`` (head + body merged).
"""
from __future__ import annotations

from fmpool.datasets._msd_common import MSDCenterSliceDataset


class MSDHippocampusDataset(MSDCenterSliceDataset):
    """MSD Task04_Hippocampus binary head+body segmentation."""

    task_key = "msd_hippocampus"
    split_task_name = "msd_hippocampus"
    modality_index = None
    provenance_extra = (
        "Task04_Hippocampus (260 imagesTr). Binary foreground = "
        "(label > 0) merging head (1) and body (2)."
    )
