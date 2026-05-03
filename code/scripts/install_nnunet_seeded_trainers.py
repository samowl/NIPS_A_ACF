#!/usr/bin/env python
"""Install or verify FMPool seeded nnU-Net trainer discovery.

Official nnU-Net v2 resolves custom trainers by recursively scanning the
installed ``nnunetv2.training.nnUNetTrainer`` package. Run this helper once in
the active environment before launching seeded nnU-Net training/prediction.
It writes a small compatibility module into the installed nnU-Net trainer tree
and verifies the documented seeded trainer class names without starting
training.
"""
from __future__ import annotations

import argparse
import json

from fmpool.nnunet_seeded import ensure_nnunet_discovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify existing discovery without writing the shim.",
    )
    args = parser.parse_args()
    status = ensure_nnunet_discovery(write=not args.check_only)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
