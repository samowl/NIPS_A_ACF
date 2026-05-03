#!/usr/bin/env python3
"""Decompose failure correlation into seed-floor, cross-checkpoint, and cross-family strata.

This diagnostic addresses the main structural caveat of the primary
within/cross estimator: the primary ``within`` column is dominated by same-FM
different-decoder-seed pairs. Here, distinct checkpoints from the same broad
upstream family (DINOv2-S/B, OpenAI CLIP-B/L, ResNet-18/50 when functional)
are scored separately from the same-FM seed floor and from cross-family pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fmpool.estimators import functional_floor_filter

TASKS = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")
FUNCTIONAL_FLOOR = 0.30
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0

FAMILY_GROUPS = {
    "dinov2_vitb14": "dinov2",
    "dinov2_vits14": "dinov2",
    "dinov2_vitl14": "dinov2",
    "dinov2_vitg14": "dinov2",
    "clip_vitb16": "openai_clip",
    "clip_vitb32": "openai_clip",
    "clip_vitl14": "openai_clip",
    "resnet18": "resnet",
    "resnet34": "resnet",
    "resnet50": "resnet",
    "resnet101": "resnet",
    "convnext_tiny": "convnext",
    "convnext_small": "convnext",
    "convnext_base": "convnext",
    "efficientnet_b0": "efficientnet",
    "efficientnet_b1": "efficientnet",
    "efficientnet_b2": "efficientnet",
    "efficientnet_b3": "efficientnet",
    "mae_vitb16": "mae",
    "mae_vitl16": "mae",
    "deit_vitb16": "deit",
    "deit_vits16": "deit",
    "deit_vitt16": "deit",
}


@dataclass(frozen=True)
class LoadedTrace:
    task: str
    fm: str
    seed: int
    test_ids: tuple[str, ...]
    per_case_dice: np.ndarray
    rel_path: str


def _family(fm: str) -> str:
    return FAMILY_GROUPS.get(fm, fm)


def _mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _ci(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return [float(lo), float(hi)]


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _load_trace(path: Path, root: Path) -> LoadedTrace:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    fm = str(payload.get("fm") or path.parent.name)
    return LoadedTrace(
        task=str(payload["task"]),
        fm=fm,
        seed=int(payload["seed"]),
        test_ids=tuple(str(x) for x in payload["test_ids"]),
        per_case_dice=np.asarray(payload["per_case_dice"], dtype=float),
        rel_path=str(path.relative_to(root)),
    )


def _collect(results_root: Path, task: str) -> list[LoadedTrace]:
    task_root = results_root / "per_case_dice" / task
    traces = [
        _load_trace(path, results_root)
        for path in sorted(task_root.glob("*/seed_*.json"))
    ]
    traces = [t for t in traces if t.task == task]
    if not traces:
        raise RuntimeError(f"no per-case traces for {task} under {task_root}")
    first_ids = traces[0].test_ids
    for trace in traces:
        if trace.test_ids != first_ids:
            raise ValueError(f"{trace.rel_path}: test_ids mismatch")
    return traces


def _case_units(task: str, test_ids: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    if task == "acdc_lv":
        return "ACDC patient", tuple(
            case_id.split("_frame", 1)[0] if "_frame" in case_id else case_id
            for case_id in test_ids
        )
    if task == "brats_wt":
        return "BraTS subject", tuple(
            case_id.rsplit("_slice", 1)[0] if "_slice" in case_id else case_id
            for case_id in test_ids
        )
    return None, test_ids


def _aggregate_by_unit(
    per_case: dict[tuple[str, int], np.ndarray],
    unit_ids: tuple[str, ...],
) -> dict[tuple[str, int], np.ndarray]:
    unit_arr = np.asarray(unit_ids)
    units = sorted(set(unit_ids))
    idxs = {unit: np.where(unit_arr == unit)[0] for unit in units}
    return {
        key: np.asarray([np.mean(values[idxs[unit]]) for unit in units])
        for key, values in per_case.items()
    }


def _keys_matrix(per_case: dict[tuple[str, int], np.ndarray]) -> tuple[list[tuple[str, int]], np.ndarray]:
    keys = sorted(per_case)
    matrix = np.stack([np.asarray(per_case[key], dtype=float) for key in keys], axis=0)
    return keys, matrix


def _corr_matrix(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=1))
    denom = np.outer(norm, norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (centered @ centered.T) / denom
    corr[denom < 1e-24] = np.nan
    return corr


def _bucket_pairs(keys: list[tuple[str, int]]) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[str]]]:
    values = {
        "same_fm_seed_floor": [],
        "same_family_cross_checkpoint": [],
        "cross_family": [],
    }
    pair_indices: dict[str, list[tuple[int, int]]] = {k: [] for k in values}
    pair_examples: dict[str, list[str]] = {k: [] for k in values}
    for idx, key_a in enumerate(keys):
        fm_a, seed_a = key_a
        fam_a = _family(fm_a)
        for key_b in keys[idx + 1 :]:
            fm_b, seed_b = key_b
            fam_b = _family(fm_b)
            if fm_a == fm_b and seed_a != seed_b:
                bucket = "same_fm_seed_floor"
            elif fm_a != fm_b and fam_a == fam_b:
                bucket = "same_family_cross_checkpoint"
            elif fam_a != fam_b:
                bucket = "cross_family"
            else:
                continue
            pair_indices[bucket].append((idx, keys.index(key_b)))
            if len(pair_examples[bucket]) < 8:
                pair_examples[bucket].append(f"{fm_a}:{seed_a}--{fm_b}:{seed_b}")
    return pair_indices, pair_examples


def _stat_from_matrix(
    keys: list[tuple[str, int]],
    matrix: np.ndarray,
    pair_indices: dict[str, list[tuple[int, int]]] | None = None,
    pair_examples: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if pair_indices is None or pair_examples is None:
        pair_indices, pair_examples = _bucket_pairs(keys)
    corr = _corr_matrix(matrix)
    values: dict[str, list[float]] = {}
    for bucket, pairs in pair_indices.items():
        values[bucket] = [float(corr[i, j]) for i, j in pairs]

    out: dict[str, Any] = {}
    for bucket, vals in values.items():
        finite = [v for v in vals if math.isfinite(v)]
        out[bucket] = {
            "rbar": _mean_or_nan(vals),
            "n_pairs": len(vals),
            "n_finite_pairs": len(finite),
            "example_pairs": pair_examples[bucket],
        }
    seed = out["same_fm_seed_floor"]["rbar"]
    xckpt = out["same_family_cross_checkpoint"]["rbar"]
    cross = out["cross_family"]["rbar"]
    out["gaps"] = {
        "seed_floor_minus_cross": seed - cross,
        "xcheckpoint_minus_cross": xckpt - cross,
        "seed_floor_minus_xcheckpoint": seed - xckpt,
    }
    return out


def _stat(per_case: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    keys, matrix = _keys_matrix(per_case)
    return _stat_from_matrix(keys, matrix)


def _bootstrap(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    keys, matrix = _keys_matrix(per_case)
    pair_indices, pair_examples = _bucket_pairs(keys)
    n = matrix.shape[1]
    draws = {
        "same_fm_seed_floor": np.empty(n_boot, dtype=float),
        "same_family_cross_checkpoint": np.empty(n_boot, dtype=float),
        "cross_family": np.empty(n_boot, dtype=float),
        "seed_floor_minus_cross": np.empty(n_boot, dtype=float),
        "xcheckpoint_minus_cross": np.empty(n_boot, dtype=float),
        "seed_floor_minus_xcheckpoint": np.empty(n_boot, dtype=float),
    }
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        point = _stat_from_matrix(
            keys,
            matrix[:, idx],
            pair_indices=pair_indices,
            pair_examples=pair_examples,
        )
        draws["same_fm_seed_floor"][b] = point["same_fm_seed_floor"]["rbar"]
        draws["same_family_cross_checkpoint"][b] = point["same_family_cross_checkpoint"]["rbar"]
        draws["cross_family"][b] = point["cross_family"]["rbar"]
        for gap_key, val in point["gaps"].items():
            draws[gap_key][b] = val
    return {key: _ci(vals) for key, vals in draws.items()}


def _task_summary(results_root: Path, task: str) -> dict[str, Any]:
    traces = _collect(results_root, task)
    per_case_all = {
        (trace.fm, trace.seed): trace.per_case_dice for trace in traces
    }
    per_case = functional_floor_filter(
        per_case_all, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm"
    )
    point = _stat(per_case)
    ci = _bootstrap(per_case, n_boot=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    cluster_unit, unit_ids = _case_units(task, traces[0].test_ids)
    subject_aggregated = None
    if cluster_unit is not None:
        agg = _aggregate_by_unit(per_case, unit_ids)
        subject_aggregated = {
            "cluster_unit": cluster_unit,
            "n_units": len(set(unit_ids)),
            "point": _stat(agg),
            "case_bootstrap_ci95": _bootstrap(
                agg, n_boot=BOOTSTRAP_B, seed=BOOTSTRAP_SEED
            ),
        }

    fms_kept = sorted({fm for fm, _seed in per_case})
    family_groups = {
        family: sorted(fm for fm in fms_kept if _family(fm) == family)
        for family in sorted({_family(fm) for fm in fms_kept})
    }
    return {
        "task": task,
        "n_test": len(traces[0].test_ids),
        "functional_floor": FUNCTIONAL_FLOOR,
        "source_file_count": len(traces),
        "source_files": sorted(trace.rel_path for trace in traces),
        "fms_kept_after_floor": fms_kept,
        "family_groups_after_floor": family_groups,
        "point": point,
        "case_bootstrap_ci95": ci,
        "subject_aggregated": subject_aggregated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/_merged"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = parser.parse_args(argv)

    rows = [_task_summary(args.results_root, task) for task in args.tasks]
    payload = {
        "schema_version": "cross_checkpoint_family_v1",
        "description": (
            "Diagnostic decomposition of per-case Dice correlation into "
            "same-FM decoder-seed floor, distinct-checkpoint same broad family, "
            "and cross-family strata. This is not the primary within/cross "
            "estimator; it tests the structural caveat that primary within "
            "pairs are dominated by same-FM seeds."
        ),
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "family_group_rule": (
            "DINOv2 variants, OpenAI CLIP variants, ResNet variants, ConvNeXt "
            "variants, EfficientNet variants, MAE variants, and DeiT variants "
            "are broad-family groups when multiple checkpoints are present; "
            "BiomedCLIP remains separate from OpenAI CLIP."
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
