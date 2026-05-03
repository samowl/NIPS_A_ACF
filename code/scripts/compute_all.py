"""SPEC §8 analysis pipeline: per-case Dice JSONs → paper result JSONs.

Reads ``results/per_case_dice/{task}/{fm}/seed_{seed}.json`` and
``results/per_case_dice/{task}/nnunet_fold_{f}/seed_{seed}.json``, then emits:

* ``results/within_cross/{task}.json``
* ``results/m_eff/{task}.json``
* ``results/nnunet_baseline/{task}.json``
* ``results/paper_table.json``

Every output JSON carries provenance fields (``source_files``,
``schema_version``, ``generated_at``, ``commit_sha``) to satisfy SPEC §9
reproducibility invariants. Bootstrap seeds are fixed at 0.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

# Make ``fmpool`` importable whether the script is run from repo root or scripts/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmpool.estimators import (  # noqa: E402
    case_bootstrap,
    compute_m_eff,
    functional_floor_filter,
    hierarchical_bootstrap,
    subject_cluster_bootstrap,
    within_cross_rbar,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "fmpool_clean_v1"
TASKS: tuple[str, ...] = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")
PRIMARY_FMS: tuple[str, ...] = (
    "biomedclip",
    "clip_vitb16",
    "clip_vitl14",
    "convnext_tiny",
    "deit_vitb16",
    "dinov2_vitb14",
    "dinov2_vits14",
    "efficientnet_b0",
    "mae_vitb16",
    "resnet18",
    "resnet50",
)
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
FUNCTIONAL_FLOOR = 0.30
TAU = 0.85
CALIBRATION_FRACTION = 0.35
DINOV2_FAMILY_PAIRS = frozenset(
    [frozenset({"dinov2_vitb14", "dinov2_vits14"})]
)

_NNUNET_FOLD_RE = re.compile(r"^nnunet_fold_(\d+)$")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_commit_sha(cwd: Path) -> str:
    """Return current git commit SHA, or ``"not_recorded_for_artifact"`` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "not_recorded_for_artifact"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return "not_recorded_for_artifact"


def _iso_utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel_paths(paths: Iterable[Path], root: Path) -> list[str]:
    rels: list[str] = []
    for p in paths:
        try:
            rels.append(str(p.resolve().relative_to(root.resolve())))
        except ValueError:
            rels.append(str(p.resolve()))
    return sorted(rels)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerCaseJSON:
    path: Path
    task: str
    group: str  # fm name or "nnunet_fold_{f}"
    seed: int
    per_case_dice: np.ndarray
    test_ids: tuple[str, ...]


def _load_per_case_json(path: Path) -> PerCaseJSON:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    required = {"task", "seed", "per_case_dice", "test_ids"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    group = payload.get("fm") or path.parent.name
    return PerCaseJSON(
        path=path,
        task=str(payload["task"]),
        group=str(group),
        seed=int(payload["seed"]),
        per_case_dice=np.asarray(payload["per_case_dice"], dtype=np.float64),
        test_ids=tuple(str(x) for x in payload["test_ids"]),
    )


def _collect_task_jsons(
    results_root: Path,
    task: str,
    *,
    allowed_fms: Iterable[str] | None = PRIMARY_FMS,
) -> tuple[list[PerCaseJSON], list[PerCaseJSON]]:
    """Return (fm_jsons, nnunet_jsons) for ``task``.

    Only well-formed directories ``{task}/{group}/seed_{seed}.json`` are
    considered; anything else is ignored. By default this collector is pinned
    to the paper's 11-FM primary source pool so artifact regeneration of
    Table 1 is not affected by additional diagnostic traces stored under the
    same ``per_case_dice`` tree. Pass ``allowed_fms=None`` only for diagnostics
    that intentionally consume all completed FMs.
    """
    task_root = results_root / "per_case_dice" / task
    if not task_root.is_dir():
        return [], []

    allowed = None if allowed_fms is None else set(allowed_fms)
    fm_jsons: list[PerCaseJSON] = []
    nnunet_jsons: list[PerCaseJSON] = []
    for group_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
        if (
            allowed is not None
            and group_dir.name not in allowed
            and not _NNUNET_FOLD_RE.match(group_dir.name)
        ):
            continue
        for seed_file in sorted(group_dir.glob("seed_*.json")):
            parsed = _load_per_case_json(seed_file)
            if parsed.task != task:
                raise ValueError(
                    f"{seed_file}: task mismatch "
                    f"(expected {task!r}, got {parsed.task!r})"
                )
            if _NNUNET_FOLD_RE.match(group_dir.name):
                nnunet_jsons.append(parsed)
            else:
                fm_jsons.append(parsed)
    return fm_jsons, nnunet_jsons


def _validate_alignment(jsons: list[PerCaseJSON]) -> None:
    """Ensure every pool member uses identical ``test_ids`` (SPEC §9.2)."""
    if not jsons:
        return
    first = jsons[0]
    n = len(first.per_case_dice)
    for j in jsons:
        if len(j.per_case_dice) != n:
            raise ValueError(
                f"Case-count mismatch: {j.path} has "
                f"{len(j.per_case_dice)} cases, expected {n}"
            )
        if j.test_ids != first.test_ids:
            raise ValueError(
                f"test_ids mismatch: {j.path} disagrees with {first.path}"
            )


def _as_per_case_map(
    jsons: list[PerCaseJSON],
) -> dict[tuple[str, int], np.ndarray]:
    """Convert a list of loaded JSONs into the estimator input map."""
    out: dict[tuple[str, int], np.ndarray] = {}
    for j in jsons:
        key = (j.group, j.seed)
        if key in out:
            raise ValueError(
                f"Duplicate (group, seed) key {key} from {j.path}"
            )
        out[key] = j.per_case_dice
    return out


def _pool_desc_from_keys(keys: Iterable[tuple[str, int]]) -> str:
    """Human-readable pool label, preserving incomplete FM/seed grids."""
    keys = list(keys)
    fms = sorted({fm for fm, _ in keys})
    seeds = sorted({seed for _, seed in keys})
    seed_counts = {fm: len({seed for fm2, seed in keys if fm2 == fm}) for fm in fms}
    if len(set(seed_counts.values())) == 1:
        return f"{len(fms)}-FM x {len(seeds)}-seed ({len(keys)} members)"
    return f"{len(fms)}-FM completed subset ({len(keys)} members)"


def _case_metadata(task: str, test_ids: tuple[str, ...]) -> dict[str, object]:
    """Infer case/unit metadata from released ``test_ids``.

    The release stores only scalar per-case Dice traces. For slice-derived
    tasks, the higher-level patient/subject key is encoded in the case ID.
    """
    if task == "acdc_lv":
        unit_ids = [
            case_id.split("_frame", 1)[0]
            if "_frame" in case_id else case_id
            for case_id in test_ids
        ]
        return {
            "case_unit": "2D cardiac slice",
            "cluster_unit": "ACDC patient",
            "unit_ids": tuple(unit_ids),
            "n_units": len(set(unit_ids)),
        }
    if task == "brats_wt":
        unit_ids = [
            case_id.rsplit("_slice", 1)[0]
            if "_slice" in case_id else case_id
            for case_id in test_ids
        ]
        return {
            "case_unit": "2D axial lesion-enriched slice",
            "cluster_unit": "BraTS subject",
            "unit_ids": tuple(unit_ids),
            "n_units": len(set(unit_ids)),
        }
    if task.startswith("riga_"):
        return {
            "case_unit": "fundus image",
            "cluster_unit": None,
            "unit_ids": tuple(test_ids),
            "n_units": len(test_ids),
        }
    if task == "kvasir":
        return {
            "case_unit": "endoscopic image",
            "cluster_unit": None,
            "unit_ids": tuple(test_ids),
            "n_units": len(test_ids),
        }
    return {
        "case_unit": "case",
        "cluster_unit": None,
        "unit_ids": tuple(test_ids),
        "n_units": len(test_ids),
    }


def _aggregate_by_unit(
    per_case: dict[tuple[str, int], np.ndarray],
    unit_ids: tuple[str, ...],
) -> tuple[dict[tuple[str, int], np.ndarray], list[str]]:
    """Average per-case Dice within each patient/subject unit."""
    unit_arr = np.asarray(unit_ids)
    units = sorted(set(unit_ids))
    unit_to_idx = {u: np.where(unit_arr == u)[0] for u in units}
    aggregated = {
        k: np.asarray([np.mean(np.asarray(v)[unit_to_idx[u]]) for u in units])
        for k, v in per_case.items()
    }
    return aggregated, units


def _stable_unit_split(
    task: str,
    unit_ids: tuple[str, ...],
    fraction: float = CALIBRATION_FRACTION,
) -> tuple[set[str], set[str]]:
    """Deterministic unit-level split for sensitivity-only calibration checks."""
    units = sorted(set(unit_ids))
    if len(units) < 2:
        raise ValueError(f"{task}: need at least two units for split-half check")
    keyed = sorted(
        (
            hashlib.sha256(f"{task}:{unit}".encode("utf-8")).hexdigest(),
            unit,
        )
        for unit in units
    )
    n_cal = int(round(len(units) * fraction))
    n_cal = max(1, min(len(units) - 1, n_cal))
    cal_units = {unit for _h, unit in keyed[:n_cal]}
    eval_units = set(units) - cal_units
    return cal_units, eval_units


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def _round_ci(ci: tuple[float, float]) -> list[float]:
    return [float(ci[0]), float(ci[1])]


def _relabel_for_hierarchical(
    per_case: dict[tuple[str, int], np.ndarray],
) -> dict[tuple[str, int], np.ndarray]:
    """Collapse DINOv2-B/S to a shared ``dinov2`` family label so the
    hierarchical bootstrap treats them as one family (SPEC §3).

    Seeds from DINOv2-S are offset by +10000 to avoid collisions with
    DINOv2-B when both sizes share the same seed set.
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for (fm, seed), arr in per_case.items():
        if fm == "dinov2_vitb14":
            out[("dinov2", int(seed))] = arr
        elif fm == "dinov2_vits14":
            out[("dinov2", int(seed) + 10_000)] = arr
        else:
            out[(fm, int(seed))] = arr
    return out


def compute_within_cross(
    fm_jsons: list[PerCaseJSON],
    *,
    apply_floor: bool = True,
) -> dict:
    """Compute within/cross r̄, gap, and bootstrap CIs for an FM pool."""
    task = fm_jsons[0].task
    meta = _case_metadata(task, fm_jsons[0].test_ids)
    per_case_all = _as_per_case_map(fm_jsons)

    if apply_floor:
        per_case = functional_floor_filter(
            per_case_all, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm"
        )
    else:
        per_case = per_case_all

    fms_kept = sorted({fm for fm, _ in per_case.keys()})
    pool_desc = _pool_desc_from_keys(per_case.keys())

    point = within_cross_rbar(
        per_case, family_pairs_within=DINOV2_FAMILY_PAIRS, stat="pearson"
    )

    case_bs = case_bootstrap(
        per_case,
        family_pairs_within=DINOV2_FAMILY_PAIRS,
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )
    hier_input = _relabel_for_hierarchical(per_case)
    hier_bs = hierarchical_bootstrap(
        hier_input,
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )

    return {
        "pool_desc": pool_desc,
        "n_test": int(len(fm_jsons[0].test_ids)),
        "case_unit": meta["case_unit"],
        "cluster_unit": meta["cluster_unit"],
        "n_units": int(meta["n_units"]),
        "fms_available_before_floor": sorted({fm for fm, _ in per_case_all.keys()}),
        "within_rbar": float(point["within"]),
        "cross_rbar": float(point["cross"]),
        "gap": float(point["gap"]),
        "n_within_pairs": int(point["n_within_pairs"]),
        "n_cross_pairs": int(point["n_cross_pairs"]),
        "case_bootstrap_95ci_gap": _round_ci(case_bs["ci95"]["gap"]),
        "hierarchical_bootstrap_95ci_gap": _round_ci(hier_bs["ci95"]["gap"]),
        "B": BOOTSTRAP_B,
        "fms_kept_after_floor": fms_kept,
    }


def compute_subject_level(
    fm_jsons: list[PerCaseJSON],
) -> dict | None:
    """Compute patient/subject-aware sensitivity outputs for slice tasks."""
    task = fm_jsons[0].task
    meta = _case_metadata(task, fm_jsons[0].test_ids)
    if meta["cluster_unit"] is None:
        return None

    per_case_all = _as_per_case_map(fm_jsons)
    per_case = functional_floor_filter(
        per_case_all, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm"
    )
    unit_ids = tuple(str(x) for x in meta["unit_ids"])
    aggregated, units = _aggregate_by_unit(per_case, unit_ids)

    unit_point = within_cross_rbar(
        aggregated, family_pairs_within=DINOV2_FAMILY_PAIRS, stat="pearson"
    )
    unit_bs = case_bootstrap(
        aggregated,
        family_pairs_within=DINOV2_FAMILY_PAIRS,
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )
    cluster_bs = subject_cluster_bootstrap(
        per_case,
        case_to_subject=unit_ids,
        family_pairs_within=DINOV2_FAMILY_PAIRS,
        n_boot=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
        stat="pearson",
    )

    return {
        "task": task,
        "case_unit": meta["case_unit"],
        "cluster_unit": meta["cluster_unit"],
        "n_test": int(len(fm_jsons[0].test_ids)),
        "n_subjects": int(len(units)),
        "pool": _pool_desc_from_keys(per_case.keys()),
        "fms_kept_after_floor": sorted({fm for fm, _ in per_case}),
        "subject_aggregated": {
            "within_rbar": float(unit_point["within"]),
            "cross_rbar": float(unit_point["cross"]),
            "gap": float(unit_point["gap"]),
            "n_within_pairs": int(unit_point["n_within_pairs"]),
            "n_cross_pairs": int(unit_point["n_cross_pairs"]),
            "bootstrap_95ci_gap": _round_ci(unit_bs["ci95"]["gap"]),
        },
        "case_cluster_bootstrap": {
            "within_rbar": float(cluster_bs["point"]["within"]),
            "cross_rbar": float(cluster_bs["point"]["cross"]),
            "gap": float(cluster_bs["point"]["gap"]),
            "cluster_bootstrap_95ci_gap": _round_ci(cluster_bs["ci95"]["gap"]),
            "n_subjects": int(cluster_bs["n_subjects"]),
        },
        "B": BOOTSTRAP_B,
    }


def compute_calibration_split(
    fm_jsons: list[PerCaseJSON],
) -> dict:
    """Split-half sensitivity: choose functional FMs on calibration units,
    then evaluate the within/cross gap on disjoint evaluation units.

    This is not a prospective validation protocol; it is a leakage-sensitivity
    check for the functional-floor convention using only released traces.
    """
    task = fm_jsons[0].task
    meta = _case_metadata(task, fm_jsons[0].test_ids)
    unit_ids = tuple(str(x) for x in meta["unit_ids"])
    unit_arr = np.asarray(unit_ids)
    cal_units, eval_units = _stable_unit_split(task, unit_ids)
    cal_mask = np.asarray([u in cal_units for u in unit_ids], dtype=bool)
    eval_mask = np.asarray([u in eval_units for u in unit_ids], dtype=bool)

    per_case_all = _as_per_case_map(fm_jsons)
    by_fm: dict[str, list[np.ndarray]] = {}
    for (fm, _seed), arr in per_case_all.items():
        by_fm.setdefault(fm, []).append(np.asarray(arr, dtype=np.float64))

    calibration_mean_dice_by_fm = {
        fm: float(np.mean([float(np.mean(arr[cal_mask])) for arr in arrays]))
        for fm, arrays in by_fm.items()
    }
    fms_selected = sorted(
        fm for fm, mean in calibration_mean_dice_by_fm.items()
        if mean >= FUNCTIONAL_FLOOR
    )
    eval_per_case = {
        k: np.asarray(v)[eval_mask]
        for k, v in per_case_all.items()
        if k[0] in fms_selected
    }

    if len({fm for fm, _ in eval_per_case}) < 2:
        point = {
            "within": float("nan"),
            "cross": float("nan"),
            "gap": float("nan"),
            "n_within_pairs": 0,
            "n_cross_pairs": 0,
        }
        ci_gap = [float("nan"), float("nan")]
    else:
        point = within_cross_rbar(
            eval_per_case,
            family_pairs_within=DINOV2_FAMILY_PAIRS,
            stat="pearson",
        )
        bs = case_bootstrap(
            eval_per_case,
            family_pairs_within=DINOV2_FAMILY_PAIRS,
            n_boot=BOOTSTRAP_B,
            seed=BOOTSTRAP_SEED,
            stat="pearson",
        )
        ci_gap = _round_ci(bs["ci95"]["gap"])

    out: dict[str, object] = {
        "task": task,
        "analysis_type": "split-half sensitivity; not the primary protocol",
        "functional_floor": FUNCTIONAL_FLOOR,
        "calibration_fraction": CALIBRATION_FRACTION,
        "case_unit": meta["case_unit"],
        "cluster_unit": meta["cluster_unit"],
        "n_test": int(len(unit_ids)),
        "n_units": int(meta["n_units"]),
        "n_calibration_units": int(len(cal_units)),
        "n_evaluation_units": int(len(eval_units)),
        "n_calibration_cases": int(cal_mask.sum()),
        "n_evaluation_cases": int(eval_mask.sum()),
        "fms_available_before_floor": sorted(by_fm),
        "fms_selected_on_calibration": fms_selected,
        "calibration_mean_dice_by_fm": calibration_mean_dice_by_fm,
        "evaluation": {
            "within_rbar": float(point["within"]),
            "cross_rbar": float(point["cross"]),
            "gap": float(point["gap"]),
            "n_within_pairs": int(point["n_within_pairs"]),
            "n_cross_pairs": int(point["n_cross_pairs"]),
            "case_bootstrap_95ci_gap": ci_gap,
        },
        "B": BOOTSTRAP_B,
    }

    if meta["cluster_unit"] is not None and eval_per_case:
        eval_unit_ids = tuple(unit_arr[eval_mask].tolist())
        eval_aggregated, eval_units_sorted = _aggregate_by_unit(
            eval_per_case, eval_unit_ids
        )
        unit_point = within_cross_rbar(
            eval_aggregated,
            family_pairs_within=DINOV2_FAMILY_PAIRS,
            stat="pearson",
        )
        out["evaluation_unit_aggregated"] = {
            "n_units": int(len(eval_units_sorted)),
            "within_rbar": float(unit_point["within"]),
            "cross_rbar": float(unit_point["cross"]),
            "gap": float(unit_point["gap"]),
            "n_within_pairs": int(unit_point["n_within_pairs"]),
            "n_cross_pairs": int(unit_point["n_cross_pairs"]),
        }

    return out


def compute_m_eff_pools(fm_jsons: list[PerCaseJSON]) -> dict:
    """Mono-pool and 4-FM diverse-pool M_eff summary at tau=0.85.

    Diverse compositions follow the paper's selection unit: choose four
    distinct FMs, then choose one seed-trace from each FM. This avoids mixing
    the old pairwise/full-pool diagnostic with the claimed 4-FM composition
    estimand.
    """
    per_case_all = _as_per_case_map(fm_jsons)
    per_case = functional_floor_filter(
        per_case_all, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm"
    )

    mono_pools: list[dict] = []
    by_fm: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    for (fm, seed), arr in per_case.items():
        by_fm.setdefault(fm, {})[(fm, seed)] = arr

    for fm in sorted(by_fm):
        stats = compute_m_eff(by_fm[fm], tau=TAU)
        mono_pools.append(
            {
                "fm": fm,
                "rho_fail": float(stats["rho_fail"]),
                "M_eff": float(stats["M_eff"]),
                "n_fm_seed": int(stats["n_fm_seed"]),
            }
        )

    diverse_rhos: list[float] = []
    diverse_meffs: list[float] = []
    n_fm_compositions = 0
    n_seed_compositions = 0
    fms_sorted = sorted(by_fm)

    for fm_combo in itertools.combinations(fms_sorted, 4):
        seed_choices = [
            sorted(by_fm[fm].items(), key=lambda kv: kv[0][1])
            for fm in fm_combo
        ]
        if any(not choices for choices in seed_choices):
            continue
        n_fm_compositions += 1
        for seed_tuple in itertools.product(*seed_choices):
            combo = {key: arr for key, arr in seed_tuple}
            s = compute_m_eff(combo, tau=TAU)
            n_seed_compositions += 1
            if not np.isnan(s["rho_fail"]):
                diverse_rhos.append(float(s["rho_fail"]))
            if not np.isnan(s["M_eff"]):
                diverse_meffs.append(float(s["M_eff"]))

    full_pool_summary: dict[str, float | int] | None = None
    if len(fms_sorted) >= 2:
        s_full = compute_m_eff(per_case, tau=TAU)
        full_pool_summary = {
            "rho_fail": float(s_full["rho_fail"]),
            "M_eff": float(s_full["M_eff"]),
            "n_fm_seed": int(s_full["n_fm_seed"]),
        }

    return {
        "mono_pools": mono_pools,
        "diverse_pools_summary": {
            "pool_size_fms": 4,
            "n_fm_compositions": int(n_fm_compositions),
            "n_seed_compositions": int(n_seed_compositions),
            "n_valid_compositions": int(len(diverse_meffs)),
            "rho_fail_mean": (
                float(np.mean(diverse_rhos)) if diverse_rhos else float("nan")
            ),
            "rho_fail_std": (
                float(np.std(diverse_rhos, ddof=1))
                if len(diverse_rhos) > 1 else float("nan")
            ),
            "M_eff_mean": (
                float(np.mean(diverse_meffs)) if diverse_meffs else float("nan")
            ),
            "M_eff_std": (
                float(np.std(diverse_meffs, ddof=1))
                if len(diverse_meffs) > 1 else float("nan")
            ),
        },
        "full_pool_diagnostic": full_pool_summary,
    }


def compute_nnunet_within_cross(
    nnunet_jsons: list[PerCaseJSON],
) -> dict | None:
    """nnU-Net within-fold vs cross-fold correlation gap."""
    if not nnunet_jsons:
        return None
    per_case = _as_per_case_map(nnunet_jsons)

    point = within_cross_rbar(per_case, stat="pearson")
    case_bs = case_bootstrap(
        per_case, n_boot=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, stat="pearson"
    )
    hier_bs = hierarchical_bootstrap(
        per_case, n_boot=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, stat="pearson"
    )

    folds = sorted({fold for fold, _ in per_case.keys()})
    seeds = sorted({s for _, s in per_case.keys()})
    return {
        "pool_desc": f"{len(folds)}-fold x {len(seeds)}-seed ({len(per_case)} members)",
        "within_rbar": float(point["within"]),
        "cross_rbar": float(point["cross"]),
        "gap": float(point["gap"]),
        "n_within_pairs": int(point["n_within_pairs"]),
        "n_cross_pairs": int(point["n_cross_pairs"]),
        "case_bootstrap_95ci_gap": _round_ci(case_bs["ci95"]["gap"]),
        "hierarchical_bootstrap_95ci_gap": _round_ci(hier_bs["ci95"]["gap"]),
        "B": BOOTSTRAP_B,
        "folds": folds,
        "seeds": seeds,
    }


# ---------------------------------------------------------------------------
# JSON writers
# ---------------------------------------------------------------------------


def _strict_json_value(value):
    """Return a JSON-standard representation with non-finite floats as null."""
    if isinstance(value, dict):
        return {k: _strict_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            _strict_json_value(payload),
            fh,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        fh.write("\n")


def write_within_cross(
    task: str,
    fm_jsons: list[PerCaseJSON],
    out_root: Path,
    results_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict:
    result = compute_within_cross(fm_jsons, apply_floor=True)
    source_files = _rel_paths((j.path for j in fm_jsons), results_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        "task": task,
        "pool": result["pool_desc"],
        "n_test": result["n_test"],
        "case_unit": result["case_unit"],
        "cluster_unit": result["cluster_unit"],
        "n_units": result["n_units"],
        "within_rbar": result["within_rbar"],
        "cross_rbar": result["cross_rbar"],
        "gap": result["gap"],
        "n_within_pairs": result["n_within_pairs"],
        "n_cross_pairs": result["n_cross_pairs"],
        "case_bootstrap_95ci_gap": result["case_bootstrap_95ci_gap"],
        "hierarchical_bootstrap_95ci_gap": result["hierarchical_bootstrap_95ci_gap"],
        "B": result["B"],
        "functional_floor_0.30_applied": True,
        "fms_available_before_floor": result["fms_available_before_floor"],
        "fms_kept_after_floor": result["fms_kept_after_floor"],
        "source_files": source_files,
    }
    _write_json(out_root / "within_cross" / f"{task}.json", payload)
    return payload


def write_m_eff(
    task: str,
    fm_jsons: list[PerCaseJSON],
    out_root: Path,
    results_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict:
    body = compute_m_eff_pools(fm_jsons)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        "task": task,
        "tau": TAU,
        "mono_pools": body["mono_pools"],
        "diverse_pools_summary": body["diverse_pools_summary"],
        "full_pool_diagnostic": body["full_pool_diagnostic"],
        "source_files": _rel_paths((j.path for j in fm_jsons), results_root),
    }
    _write_json(out_root / "m_eff" / f"{task}.json", payload)
    return payload


def write_subject_level(
    task: str,
    fm_jsons: list[PerCaseJSON],
    out_root: Path,
    results_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict | None:
    body = compute_subject_level(fm_jsons)
    if body is None:
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        **body,
        "source_files": _rel_paths((j.path for j in fm_jsons), results_root),
    }
    _write_json(out_root / "subject_level" / f"{task}.json", payload)
    return payload


def write_calibration_split(
    task: str,
    fm_jsons: list[PerCaseJSON],
    out_root: Path,
    results_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict:
    body = compute_calibration_split(fm_jsons)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        **body,
        "source_files": _rel_paths((j.path for j in fm_jsons), results_root),
    }
    _write_json(out_root / "calibration_split" / f"{task}.json", payload)
    return payload


def write_nnunet(
    task: str,
    nnunet_jsons: list[PerCaseJSON],
    out_root: Path,
    results_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict | None:
    body = compute_nnunet_within_cross(nnunet_jsons)
    if body is None:
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        "task": task,
        "pool": body["pool_desc"],
        "within_rbar": body["within_rbar"],
        "cross_rbar": body["cross_rbar"],
        "gap": body["gap"],
        "n_within_pairs": body["n_within_pairs"],
        "n_cross_pairs": body["n_cross_pairs"],
        "case_bootstrap_95ci_gap": body["case_bootstrap_95ci_gap"],
        "hierarchical_bootstrap_95ci_gap": body["hierarchical_bootstrap_95ci_gap"],
        "B": body["B"],
        "folds": body["folds"],
        "seeds": body["seeds"],
        "source_files": _rel_paths((j.path for j in nnunet_jsons), results_root),
    }
    _write_json(out_root / "nnunet_baseline" / f"{task}.json", payload)
    return payload


def write_paper_table(
    within_cross_rows: list[dict],
    nnunet_rows: list[dict],
    out_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict:
    rows: list[dict] = []
    for r in within_cross_rows:
        rows.append(
            {
                "task": r["task"],
                "pool": r["pool"],
                "n_test": r["n_test"],
                "case_unit": r["case_unit"],
                "cluster_unit": r["cluster_unit"],
                "n_units": r["n_units"],
                "fms_kept_after_floor": r["fms_kept_after_floor"],
                "within_rbar": r["within_rbar"],
                "cross_rbar": r["cross_rbar"],
                "gap": r["gap"],
                "gap_ci95_case": r["case_bootstrap_95ci_gap"],
                "gap_ci95_hier": r["hierarchical_bootstrap_95ci_gap"],
                "source_files": r["source_files"],
            }
        )
    nnu_rows: list[dict] = []
    for r in nnunet_rows:
        nnu_rows.append(
            {
                "task": r["task"],
                "within_rbar": r["within_rbar"],
                "cross_rbar": r["cross_rbar"],
                "gap": r["gap"],
                "gap_ci95_case": r["case_bootstrap_95ci_gap"],
                "gap_ci95_hier": r["hierarchical_bootstrap_95ci_gap"],
                "source_files": r["source_files"],
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        "rows": rows,
        "nnunet_rows": nnu_rows,
    }
    _write_json(out_root / "paper_table.json", payload)
    return payload


def write_provenance_summary(
    within_cross_rows: list[dict],
    subject_rows: list[dict],
    calibration_rows: list[dict],
    out_root: Path,
    commit_sha: str,
    generated_at: str,
) -> dict:
    subject_by_task = {r["task"]: r for r in subject_rows}
    calibration_by_task = {r["task"]: r for r in calibration_rows}
    tasks: list[dict[str, object]] = []
    for row in within_cross_rows:
        task = row["task"]
        tasks.append({
            "task": task,
            "n_test": row["n_test"],
            "case_unit": row["case_unit"],
            "cluster_unit": row["cluster_unit"],
            "n_units": row["n_units"],
            "pool": row["pool"],
            "fms_available_before_floor": row["fms_available_before_floor"],
            "fms_kept_after_floor": row["fms_kept_after_floor"],
            "within_cross_json": f"within_cross/{task}.json",
            "m_eff_json": f"m_eff/{task}.json",
            "subject_level_json": (
                f"subject_level/{task}.json"
                if task in subject_by_task else None
            ),
            "calibration_split_json": (
                f"calibration_split/{task}.json"
                if task in calibration_by_task else None
            ),
            "source_file_count": len(row["source_files"]),
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "commit_sha": commit_sha,
        "functional_floor": FUNCTIONAL_FLOOR,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tau": TAU,
        "tasks": tasks,
    }
    _write_json(out_root / "provenance_summary.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    results_root: Path,
    out_root: Path,
    tasks: Iterable[str] = TASKS,
    *,
    only_within_cross: bool = False,
    only_m_eff: bool = False,
    only_nnunet: bool = False,
    include_all_fms: bool = False,
    commit_sha: str | None = None,
    generated_at: str | None = None,
) -> dict:
    """Run the full analysis pipeline and return the paper table payload."""
    commit_sha = commit_sha or _git_commit_sha(_REPO_ROOT)
    generated_at = generated_at or _iso_utc_now()

    any_only = only_within_cross or only_m_eff or only_nnunet
    do_wc = only_within_cross or not any_only
    do_meff = only_m_eff or not any_only
    do_nnu = only_nnunet or not any_only
    do_release_sensitivity = not any_only

    wc_rows: list[dict] = []
    nnu_rows: list[dict] = []
    subject_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for task in tasks:
        fm_jsons, nnunet_jsons = _collect_task_jsons(
            results_root,
            task,
            allowed_fms=None if include_all_fms else PRIMARY_FMS,
        )
        _validate_alignment(fm_jsons)
        _validate_alignment(nnunet_jsons)

        if do_wc and fm_jsons:
            payload = write_within_cross(
                task, fm_jsons, out_root, results_root, commit_sha, generated_at
            )
            wc_rows.append(payload)

        if do_meff and fm_jsons:
            write_m_eff(
                task, fm_jsons, out_root, results_root, commit_sha, generated_at
            )

        if do_nnu and nnunet_jsons:
            payload = write_nnunet(
                task,
                nnunet_jsons,
                out_root,
                results_root,
                commit_sha,
                generated_at,
            )
            if payload is not None:
                nnu_rows.append(payload)

        if do_release_sensitivity and fm_jsons:
            subject_payload = write_subject_level(
                task, fm_jsons, out_root, results_root, commit_sha, generated_at
            )
            if subject_payload is not None:
                subject_rows.append(subject_payload)
            calibration_rows.append(
                write_calibration_split(
                    task, fm_jsons, out_root, results_root, commit_sha, generated_at
                )
            )

    table = write_paper_table(
        wc_rows, nnu_rows, out_root, commit_sha, generated_at
    )
    if do_release_sensitivity:
        write_provenance_summary(
            wc_rows, subject_rows, calibration_rows, out_root, commit_sha, generated_at
        )
    return table


def _read_json_tree(root: Path) -> dict[str, object]:
    """Read every *.json file under ``root`` keyed by relative path.

    Strips the volatile ``generated_at`` field so two back-to-back runs with
    a shared timestamp are byte-comparable for SPEC §9 determinism checks.
    """
    out: dict[str, object] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if k != "generated_at"}
        out[str(path.relative_to(root))] = payload
    return out


def _verify_determinism(
    results_root: Path,
    out_root: Path,
    tasks: Iterable[str],
    *,
    include_all_fms: bool = False,
) -> None:
    """Run the pipeline twice and assert bit-identical outputs."""
    fixed_time = _iso_utc_now()
    sha = _git_commit_sha(_REPO_ROOT)
    run_pipeline(
        results_root,
        out_root,
        tasks=tasks,
        commit_sha=sha,
        generated_at=fixed_time,
        include_all_fms=include_all_fms,
    )
    snapshot_a = _read_json_tree(out_root)
    run_pipeline(
        results_root,
        out_root,
        tasks=tasks,
        commit_sha=sha,
        generated_at=fixed_time,
        include_all_fms=include_all_fms,
    )
    snapshot_b = _read_json_tree(out_root)
    if snapshot_a != snapshot_b:
        diffs = sorted(
            k for k in set(snapshot_a) | set(snapshot_b)
            if snapshot_a.get(k) != snapshot_b.get(k)
        )
        raise AssertionError(
            f"Non-deterministic outputs detected in: {diffs}"
        )
    logger.info(
        "verify: %d JSON files reproduced bit-identically", len(snapshot_a)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Root directory containing per_case_dice/ subtree.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (defaults to --results-root).",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=list(TASKS),
        help="Restrict to a single task (repeatable).",
    )
    parser.add_argument(
        "--only-within-cross",
        action="store_true",
        help="Only compute within/cross JSONs (skip m_eff + nnunet).",
    )
    parser.add_argument(
        "--only-m-eff",
        action="store_true",
        help="Only compute m_eff JSONs.",
    )
    parser.add_argument(
        "--only-nnunet",
        action="store_true",
        help="Only compute nnunet baseline JSONs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the pipeline twice and assert bit-identical outputs.",
    )
    parser.add_argument(
        "--include-all-fms",
        action="store_true",
        help=(
            "Opt in to all FM traces under per_case_dice/. The default is the "
            "paper's 11-FM primary source pool so Table 1 is reproducible even "
            "when diagnostic cross-checkpoint traces are bundled."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG|INFO|WARNING|ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    results_root = args.results_root
    out_root = args.out or results_root
    tasks = tuple(args.task) if args.task else TASKS

    if args.verify:
        _verify_determinism(
            results_root,
            out_root,
            tasks,
            include_all_fms=args.include_all_fms,
        )
    else:
        run_pipeline(
            results_root,
            out_root,
            tasks=tasks,
            only_within_cross=args.only_within_cross,
            only_m_eff=args.only_m_eff,
            only_nnunet=args.only_nnunet,
            include_all_fms=args.include_all_fms,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
