"""MSD Task02_Heart central-axial-slice loader (task ``msd_heart``).

Dataset: 20 imagesTr 3D cardiac MRI volumes. Binary mask = ``label > 0``
(left atrium foreground).
"""
from __future__ import annotations

from fmpool.datasets._msd_common import MSDCenterSliceDataset


class MSDHeartDataset(MSDCenterSliceDataset):
    """MSD Task02_Heart binary left-atrium segmentation."""

    task_key = "msd_heart"
    split_task_name = "msd_heart"
    modality_index = None
    provenance_extra = (
        "Task02_Heart (20 imagesTr cardiac MRI). Binary foreground = "
        "(label > 0)."
    )
