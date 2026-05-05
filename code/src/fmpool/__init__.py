"""fmpool -- frozen-feature FM-pool audit utilities.

The package keeps the NumPy/SciPy estimators importable without importing
Torch. Training-only objects such as ``LinearSegHead`` and deterministic Torch
RNG helpers are loaded lazily through ``__getattr__`` so artifact summary
scorers can run from ``requirements-repro.txt`` only.
"""
from __future__ import annotations

from fmpool.estimators import (
    case_bootstrap,
    compute_m_eff,
    dice_per_case,
    empty_empty_filter,
    family_pair_matrix,
    functional_floor_filter,
    hierarchical_bootstrap,
    icc_a1,
    paired_delta_bootstrap,
    pearson_rbar,
    sparse_class_sensitivity_sweep,
    within_cross_rbar,
)

__all__ = [
    # decoder
    "LinearSegHead",
    # determinism
    "set_seed",
    "get_rng_state",
    "set_rng_state",
    "sha256_file",
    # estimators
    "dice_per_case",
    "pearson_rbar",
    "icc_a1",
    "family_pair_matrix",
    "within_cross_rbar",
    "functional_floor_filter",
    "empty_empty_filter",
    "sparse_class_sensitivity_sweep",
    "compute_m_eff",
    "case_bootstrap",
    "hierarchical_bootstrap",
    "paired_delta_bootstrap",
]


def __getattr__(name: str):
    """Lazy training-only exports to avoid Torch in artifact scorers."""
    if name == "LinearSegHead":
        from fmpool.decoder import LinearSegHead

        return LinearSegHead
    if name in {"set_seed", "get_rng_state", "set_rng_state", "sha256_file"}:
        from fmpool import determinism

        return getattr(determinism, name)
    raise AttributeError(f"module 'fmpool' has no attribute {name!r}")
