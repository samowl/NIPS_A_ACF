"""Regenerate late-stage BraTS appendix summaries (M19, M20)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fmpool.estimators import (
    case_bootstrap,
    functional_floor_filter,
    subject_cluster_bootstrap,
    within_cross_rbar,
)

SCHEMA_VERSION = "late_brats_summaries_v1"
DINOV2_FAMILY_PAIRS = frozenset(
    [frozenset({"dinov2_vitb14", "dinov2_vits14"})]
)
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
FUNCTIONAL_FLOOR = 0.30


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _ci_list(ci: tuple[float, float]) -> list[float]:
    return [float(ci[0]), float(ci[1])]


def _subject_aggregate(per_case: dict[tuple[str, int], np.ndarray], test_ids: list[str]) -> dict[tuple[str, int], np.ndarray]:
    subjects = [case_id.rsplit("_slice", 1)[0] for case_id in test_ids]
    unique = sorted(set(subjects))
    idx_by_subject = {
        subject: np.asarray([i for i, s in enumerate(subjects) if s == subject])
        for subject in unique
    }
    return {
        key: np.asarray([values[idx].mean() for idx in idx_by_subject.values()])
        for key, values in per_case.items()
    }


def summarize_m20(results_root: Path) -> dict:
    root = results_root / "per_case_dice" / "brats_wt_multimodal"
    paths = sorted(root.glob("*/seed_*.json"))
    if not paths:
        raise RuntimeError(f"no M20 JSONs under {root}")

    payloads = [_load(path) for path in paths]
    test_ids = payloads[0]["test_ids"]
    per_case = {}
    source_files = []
    for payload, path in zip(payloads, paths, strict=True):
        if payload["task"] != "brats_wt_multimodal":
            raise ValueError(f"{path}: unexpected task {payload['task']!r}")
        if payload["test_ids"] != test_ids:
            raise ValueError(f"{path}: test_ids mismatch")
        key = (str(payload["fm"]), int(payload["seed"]))
        per_case[key] = np.asarray(payload["per_case_dice"], dtype=float)
        source_files.append(_rel(path, results_root))

    per_case = functional_floor_filter(
        per_case, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm"
    )
    point = within_cross_rbar(
        per_case, family_pairs_within=set(DINOV2_FAMILY_PAIRS), stat="pearson"
    )
    boot = case_bootstrap(
        per_case,
        family_pairs_within=set(DINOV2_FAMILY_PAIRS),
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )
    subject_ids = [case_id.rsplit("_slice", 1)[0] for case_id in test_ids]
    subject_per_case = _subject_aggregate(per_case, test_ids)
    subject_point = within_cross_rbar(
        subject_per_case,
        family_pairs_within=set(DINOV2_FAMILY_PAIRS),
        stat="pearson",
    )
    subject_boot = case_bootstrap(
        subject_per_case,
        family_pairs_within=set(DINOV2_FAMILY_PAIRS),
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )
    cluster_boot = subject_cluster_bootstrap(
        per_case,
        case_to_subject=subject_ids,
        family_pairs_within=set(DINOV2_FAMILY_PAIRS),
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )

    by_fm: dict[str, list[float]] = {}
    for (fm, _seed), values in per_case.items():
        by_fm.setdefault(fm, []).append(float(np.mean(values)))
    per_fm = [
        {
            "fm": fm,
            "n_seeds": len(vals),
            "mean_dice": float(np.mean(vals)),
            "min_dice": float(np.min(vals)),
            "max_dice": float(np.max(vals)),
        }
        for fm, vals in sorted(by_fm.items())
    ]

    return {
        "task": "brats_wt_multimodal",
        "protocol": (
            "BraTS 2024 GLI 2D multimodal RGB derivative: R=t1c, G=t2w, "
            "B=t2f/FLAIR; same subject-level split and top-10 foreground-area slice "
            "selection as brats_wt."
        ),
        "n_test": int(len(test_ids)),
        "n_subjects": int(len({case_id.rsplit("_slice", 1)[0] for case_id in test_ids})),
        "pool": f"{len(by_fm)}-FM x 4-seed ({len(per_case)} members)",
        "per_fm": per_fm,
        "within_cross_case": {
            "within": float(point["within"]),
            "cross": float(point["cross"]),
            "gap": float(point["gap"]),
            "n_within_pairs": int(point["n_within_pairs"]),
            "n_cross_pairs": int(point["n_cross_pairs"]),
            "gap_ci95_case": _ci_list(boot["ci95"]["gap"]),
        },
        "within_cross_subject_aggregated": {
            "within": float(subject_point["within"]),
            "cross": float(subject_point["cross"]),
            "gap": float(subject_point["gap"]),
            "n_within_pairs": int(subject_point["n_within_pairs"]),
            "n_cross_pairs": int(subject_point["n_cross_pairs"]),
            "gap_ci95_subject_bootstrap": _ci_list(subject_boot["ci95"]["gap"]),
        },
        "within_cross_case_cluster_bootstrap": {
            "within": float(cluster_boot["point"]["within"]),
            "cross": float(cluster_boot["point"]["cross"]),
            "gap": float(cluster_boot["point"]["gap"]),
            "gap_ci95_cluster_bootstrap": _ci_list(cluster_boot["ci95"]["gap"]),
            "n_subjects": int(cluster_boot["n_subjects"]),
        },
        "source_files": source_files,
    }


def summarize_m19(results_root: Path) -> dict:
    root = results_root / "per_case_dice_m19"
    paths = sorted(root.glob("frac*/brats_wt/*/seed_*.json"))
    if not paths:
        raise RuntimeError(f"no M19 JSONs under {root}")

    cells = []
    by_frac: dict[float, list[float]] = {}
    for path in paths:
        payload = _load(path)
        frac = float(payload["train_frac"])
        mean_dice = float(payload["mean_dice"])
        by_frac.setdefault(frac, []).append(mean_dice)
        cells.append(
            {
                "train_frac": frac,
                "fm": str(payload["fm"]),
                "seed": int(payload["seed"]),
                "mean_dice": mean_dice,
                "n_train_used": int(payload["n_train_used"]),
                "n_train_total": int(payload["n_train_total"]),
                "source_file": _rel(path, results_root),
            }
        )

    per_frac = [
        {
            "train_frac": frac,
            "n_cells": len(vals),
            "mean_dice": float(np.mean(vals)),
            "min_dice": float(np.min(vals)),
            "max_dice": float(np.max(vals)),
        }
        for frac, vals in sorted(by_frac.items())
    ]
    return {
        "task": "brats_wt",
        "protocol": (
            "Cache-based single-seed training-size sweep over the five BraTS foreground "
            "FMs with completed feature caches."
        ),
        "pool": "5-FM completed-cache subset, seed 42",
        "note": (
            "M19 is an accuracy/saturation diagnostic only; with one seed per FM "
            "it is not a within-vs-cross dependence estimator."
        ),
        "per_fraction": per_frac,
        "cells": sorted(cells, key=lambda x: (x["train_frac"], x["fm"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/late_brats_summaries.json"),
    )
    args = parser.parse_args()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "functional_floor": FUNCTIONAL_FLOOR,
        "m20_multimodal": summarize_m20(args.results_root),
        "m19_train_size": summarize_m19(args.results_root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
