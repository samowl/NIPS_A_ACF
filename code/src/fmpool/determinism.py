"""Seed / RNG-state / checkpoint-hash utilities (SPEC §9).

All training and evaluation scripts must call :func:`set_seed` before any
stochastic operation. :func:`get_rng_state` / :func:`set_rng_state` support
checkpointable randomness. :func:`sha256_file` hashes checkpoints for the
``checkpoint_sha256`` field in per-case-Dice JSONs (SPEC §8).
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 1 << 20  # 1 MiB
_CUBLAS_CONFIG = ":4096:8"


def set_seed(seed: int) -> None:
    """SPEC §9.1: set every RNG and enable deterministic kernels.

    Sets ``random``, ``numpy``, ``torch`` CPU / CUDA seeds, disables cuDNN
    benchmark, sets the cuBLAS workspace configuration needed by
    ``torch.use_deterministic_algorithms`` on Ampere+ GPUs, and flips
    ``torch.use_deterministic_algorithms(True, warn_only=True)``.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required for deterministic matmul on Ampere+.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug("set_seed: seed=%d CUDA=%s", seed, torch.cuda.is_available())


def get_rng_state() -> dict[str, Any]:
    """SPEC §9: snapshot of Python/NumPy/Torch RNG state for checkpointing."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """SPEC §9: restore RNG state produced by :func:`get_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def sha256_file(path: str | os.PathLike[str], chunk_size: int = _DEFAULT_CHUNK) -> str:
    """SPEC §9.3: SHA-256 of a file on disk (for checkpoint provenance).

    Reads ``path`` in streaming chunks so that multi-GB checkpoints hash
    without blowing memory. Returns the hex digest.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sha256_file: not a regular file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
