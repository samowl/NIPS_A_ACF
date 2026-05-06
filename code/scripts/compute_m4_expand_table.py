"""Aggregate the M4 UNet-skip 9-backbone x 5-task expansion grid.

Reads per-case Dice traces from the expansion run and the original M4
diagnostic, computes per-cell mean Dice (averaged over the two
released seeds) and the single seed-pair Pearson r, and emits a
JSON summary keyed by (task, fm) for use by the appendix table.

CPU only. Does not retrain anything.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASK_ORDER = (
    "kvasir",
    "acdc_lv",
    "riga_cup",
    "brats_wt",
    "riga_disc",
)

FM_ORDER: tuple[tuple[str, str], ...] = (
    ("dinov2_vitb14", "DINOv2-B"),
    ("dinov2_vits14", "DINOv2-S"),
    ("biomedclip", "BiomedCLIP"),
    ("convnext_tiny", "ConvNeXt-T"),
    ("resnet50", "ResNet-50"),
    ("resnet18", "ResNet-18"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("mae_vitb16", "MAE-B"),
    ("deit_vitb16", "DeiT-B"),
)
SEEDS: tuple[int, int] = (42, 43)
NEW_FMS = {"resnet18", "efficientnet_b0", "mae_vitb16", "deit_vitb16"}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_pair_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def cell_status(
    task: str, fm: str, primary_root: Path, expand_root: Path
) -> dict:
    if task == "brats_wt" and fm in NEW_FMS:
        return {
            "status": "na_data_layout",
            "note": (
                "BraTS expansion not run for the new-backbone column "
                "due to a host-specific data-root layout mismatch; "
                "the original 5-backbone brats_wt row from the main "
                "M4 grid is retained unchanged."
            ),
        }

    if fm in NEW_FMS or task == "riga_disc":
        source_root = expand_root
    else:
        source_root = primary_root

    paths = [source_root / task / fm / f"seed_{s}.json" for s in SEEDS]
    have = [p.exists() for p in paths]
    if not any(have):
        return {"status": "missing", "have_seeds": []}
    if not all(have):
        seed_have = [SEEDS[i] for i, ok in enumerate(have) if ok]
        return {"status": "tbd", "have_seeds": seed_have}

    payloads = [_load(p) for p in paths]
    test_ids = payloads[0]["test_ids"]
    for p, payload, seed in zip(paths, payloads, SEEDS):
        if payload.get("task") != task or payload.get("fm") != fm:
            raise ValueError(f"{p}: task/fm mismatch")
        if int(payload.get("seed")) != seed:
            raise ValueError(f"{p}: seed mismatch")
        if list(payload["test_ids"]) != list(test_ids):
            raise ValueError(f"{p}: test_ids mismatch with seed_42")

    dices = [np.asarray(payload["per_case_dice"], dtype=np.float64) for payload in payloads]
    means = [float(d.mean()) for d in dices]
    mean_avg = float(np.mean(means))
    pair_r = _seed_pair_pearson(dices[0], dices[1])
    return {
        "status": "ok",
        "have_seeds": list(SEEDS),
        "n_test": int(len(test_ids)),
        "mean_dice_per_seed": means,
        "mean_dice": mean_avg,
        "seed_pair_r": pair_r,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("results/_merged/per_case_dice_m4"),
    )
    parser.add_argument(
        "--expand-root",
        type=Path,
        default=Path("results/_merged/per_case_dice_m4_e1/per_case_dice_m4"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/m4_unet_expand_summary.json"),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for task in TASK_ORDER:
        for fm, label in FM_ORDER:
            cell = cell_status(task, fm, args.primary_root, args.expand_root)
            cell["task"] = task
            cell["fm"] = fm
            cell["fm_label"] = label
            rows.append(cell)

    counters = {"ok": 0, "tbd": 0, "na_data_layout": 0, "missing": 0}
    for row in rows:
        counters[row["status"]] += 1

    payload = {
        "schema_version": "m4_unet_expand_v1",
        "task_order": list(TASK_ORDER),
        "fm_order": [f for f, _ in FM_ORDER],
        "fm_labels": {fm: lab for fm, lab in FM_ORDER},
        "seeds": list(SEEDS),
        "rows": rows,
        "counters": counters,
        "notes": {
            "tbd_meaning": (
                "Cells where one of the two released seeds has not yet "
                "produced a per-case JSON; the cell will fill in after "
                "the running worker finalises that seed."
            ),
            "na_data_layout_meaning": (
                "BraTS column expansion for the four new backbones is "
                "skipped because the BraTS-2024-GLI subjects on the "
                "snu12 host live under training/training_data1_v2/, "
                "while fmpool.datasets.brats expects them at the "
                "data-root top level. The original 5-backbone brats_wt "
                "row from the main M4 grid is unchanged."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"wrote {args.out}: rows={len(rows)} ok={counters['ok']} "
        f"tbd={counters['tbd']} na={counters['na_data_layout']} "
        f"missing={counters['missing']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
