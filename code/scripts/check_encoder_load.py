#!/usr/bin/env python3
"""Preflight encoder loading and dense-feature shape checks.

Run this inside the same GPU/container environment before launching a job queue.
It catches missing checkpoints, incompatible OpenCLIP versions, and feature
shape mismatches before long workers begin.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmpool import encoders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fm", nargs="+", help="FM IDs to load and shape-check")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"ENCODER_PREFLIGHT device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    for fm in args.fm:
        model, spec = encoders.build_encoder(fm)
        model = model.to(device)
        x = torch.zeros(
            (args.batch_size, 3, 224, 224),
            dtype=torch.float32,
            device=device,
        )
        mean = torch.tensor(spec.norm_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(spec.norm_std, device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            y = encoders.extract_features(model, fm, (x - mean) / std)
        expected = (args.batch_size, spec.feature_dim, *spec.feature_hw)
        if tuple(y.shape) != expected:
            raise RuntimeError(f"{fm}: got {tuple(y.shape)}, expected {expected}")
        print(f"OK {fm} shape={tuple(y.shape)} family={spec.family}")
    print("ENCODER_PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
