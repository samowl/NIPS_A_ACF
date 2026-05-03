"""Generate patient-level 5-fold CV split JSONs for M15.

For ``acdc_lv`` the split already exists on disk under
``data/splits/acdc_lv_split.json`` (created by the ACDC dataset on first
instantiation) with a top-level ``folds`` dict containing
``fold_{0..4}_{train,val}`` case-id lists. We do NOT regenerate it; this
script only verifies that the keys are present.

For ``riga_cup`` the committed split is a single train (BinRushed +
MESSIDOR) / test (Magrabia) partition without folds. We pool ALL case_ids
across both groups (RIGA exposes one image per fundus eye and no patient
grouping beyond the image filename, so image-level == patient-level for
this dataset's released metadata) and shuffle with
``numpy.random.default_rng(seed=42)``, then partition into 5 folds with
floor(N/5) sizes (remainder distributed to the first folds).

Output: ``data/splits/riga_cup_5fold.json`` with schema:
  {
    "task": "riga_cup",
    "provenance": "...",
    "seed": 42,
    "n_folds": 5,
    "_image_paths": {case_id: image_rel_path, ...},
    "folds": {
      "fold_0_train": [case_id, ...],
      "fold_0_val":   [case_id, ...],
      ...
      "fold_4_val":   [case_id, ...]
    }
  }

Run:
    python scripts/make_5fold_splits.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("fmpool.make_5fold_splits")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
N_FOLDS = 5
SEED = 42


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 5-fold CV splits for M15")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override FMPOOL_DATA_ROOT for RIGA enumeration")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


def _generate_riga_cup_5fold(repo_root: Path) -> Path:
    """Build patient/image-level 5-fold for RIGA cup."""
    import sys

    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    from fmpool.datasets import load_committed_split

    base = load_committed_split("riga_cup")
    if base is None:
        raise FileNotFoundError(
            "data/splits/riga_cup_split.json missing; instantiate the "
            "RIGA cup dataset once first to materialise it."
        )
    image_paths: dict[str, str] = dict(base.get("_image_paths", {}))
    pool: list[str] = list(base.get("train", [])) + list(base.get("test", []))
    if not pool:
        raise RuntimeError("riga_cup base split has empty train+test pool")
    if not image_paths:
        raise RuntimeError(
            "riga_cup base split has no _image_paths; cannot persist."
        )

    pool_sorted = sorted(set(pool))
    rng = np.random.default_rng(SEED)
    shuffled = list(pool_sorted)
    rng.shuffle(shuffled)

    n = len(shuffled)
    base_size = n // N_FOLDS
    rem = n - base_size * N_FOLDS
    sizes = [base_size + (1 if k < rem else 0) for k in range(N_FOLDS)]

    fold_val: list[list[str]] = []
    cursor = 0
    for k in range(N_FOLDS):
        lo = cursor
        hi = cursor + sizes[k]
        fold_val.append(shuffled[lo:hi])
        cursor = hi

    folds: dict[str, list[str]] = {}
    for k in range(N_FOLDS):
        val_set = set(fold_val[k])
        folds[f"fold_{k}_val"] = list(fold_val[k])
        folds[f"fold_{k}_train"] = [c for c in shuffled if c not in val_set]

    provenance = (
        f"riga_cup 5-fold image-level CV. Pool = train "
        f"({len(base.get('train', []))}) + test "
        f"({len(base.get('test', []))}) case_ids from "
        f"data/splits/riga_cup_split.json (deduplicated, sorted). "
        f"numpy.random.default_rng(seed={SEED}).shuffle, then split into "
        f"{N_FOLDS} folds of sizes {sizes} (remainder distributed to "
        f"first folds). 'fold_k_val' lists are disjoint across k; "
        f"'fold_k_train' = pool \\ fold_k_val."
    )
    out_path = SPLITS_DIR / "riga_cup_5fold.json"
    payload = {
        "task": "riga_cup",
        "provenance": provenance,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "_image_paths": image_paths,
        "folds": folds,
    }
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_path, json.dumps(payload, indent=2))
    return out_path


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic write via mkstemp + os.replace (survives interrupted reruns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _patient_of(case_id: str) -> str:
    """Extract patient prefix from an ACDC slice case_id.

    Format: 'patient001_frame01_slice005' -> 'patient001'.
    """
    m = re.match(r"^(patient\d{3})_frame\d{2}_slice\d{3}$", case_id)
    if m is None:
        raise ValueError(
            f"unrecognised ACDC case_id format: {case_id!r}"
        )
    return m.group(1)


def _verify_acdc_5fold(repo_root: Path) -> Path:
    """Validate acdc_lv split has fold_{0..4}_{train,val} with patient-level
    train/val disjointness, fold-val disjointness, and full pool coverage.
    Raises if any invariant is violated.
    """
    p = SPLITS_DIR / "acdc_lv_split.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} missing; instantiate the ACDC dataset once first to "
            "materialise its 5-fold split."
        )
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds = data.get("folds", {})
    missing = []
    for k in range(N_FOLDS):
        for sub in ("train", "val"):
            key = f"fold_{k}_{sub}"
            if key not in folds:
                missing.append(key)
    if missing:
        raise RuntimeError(
            f"acdc_lv split missing keys: {missing}. Re-instantiate the "
            f"ACDC dataset to regenerate {p}."
        )

    # Per-fold: train and val patients must be disjoint.
    # Across folds: val patient sets must be pairwise disjoint and cover
    # the public-patient pool exactly once.
    fold_val_patients: list[set[str]] = []
    fold_train_patients: list[set[str]] = []
    for k in range(N_FOLDS):
        train_pts = {_patient_of(c) for c in folds[f"fold_{k}_train"]}
        val_pts = {_patient_of(c) for c in folds[f"fold_{k}_val"]}
        overlap = train_pts & val_pts
        if overlap:
            raise RuntimeError(
                f"acdc_lv fold={k} has {len(overlap)} patients in both train "
                f"and val (sample: {sorted(overlap)[:3]}); patient-level leak."
            )
        fold_train_patients.append(train_pts)
        fold_val_patients.append(val_pts)

    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            inter = fold_val_patients[i] & fold_val_patients[j]
            if inter:
                raise RuntimeError(
                    f"acdc_lv fold val sets {i} and {j} overlap on "
                    f"{len(inter)} patients (sample: {sorted(inter)[:3]})."
                )

    union_val = set().union(*fold_val_patients)
    expected_pool = fold_train_patients[0] | fold_val_patients[0]
    missing_in_val = expected_pool - union_val
    if missing_in_val:
        raise RuntimeError(
            f"acdc_lv 5-fold val coverage incomplete: {len(missing_in_val)} "
            f"patients never appear in any fold_k_val "
            f"(sample: {sorted(missing_in_val)[:3]})."
        )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    riga_path = _generate_riga_cup_5fold(REPO_ROOT)
    logger.info("wrote %s", riga_path)

    acdc_path = _verify_acdc_5fold(REPO_ROOT)
    logger.info("verified %s (folds present)", acdc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
