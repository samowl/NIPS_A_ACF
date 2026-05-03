"""M12 — Bootstrap-bagging audit (96 cells = 8 FM × 4 tasks × 3 schemes).

Pure-analysis script: consumes per-case Dice JSONs already produced under
``results/per_case_dice/{task}/{fm}/seed_{seed}.json`` (SPEC §8) and emits
bootstrap-stability summaries to ``results/m12_bootstrap/{task}/{fm}/{scheme}.json``.

No GPU, no training. Idempotent: every bootstrap is seeded.

Three bagging schemes (per (task, FM) cell):

1. ``subject_bootstrap_B100``: resample case ids (with replacement) within the
   shared test set; per-bootstrap mean Dice (averaged first across seeds, then
   across resampled cases). Also reports Kendall's tau between the bootstrap
   FM-ranking and the point-estimate FM-ranking on the same task.

2. ``seed_bootstrap_B100``: resample seeds (with replacement) from the released
   seed list; per-bootstrap mean Dice = mean over resampled seeds.

3. ``pair_bootstrap_B100``: paired Δ Dice between FM_A and FM_B, resampled by
   shared case id (with replacement). All ordered pairs (A != B) are produced
   for every task and stored under the FM_A directory keyed by FM_B.

CLI
---
::

    python scripts/run_m12_bootstrap.py \\
        --tasks kvasir acdc_lv riga_cup brats_wt \\
        --fms dinov2_vitb14 dinov3_vitb16 sam2_hiera_b mae_vitb16 \\
              ijepa_vitb16 dam_vitb16 swag_regnet usat_vitb16 \\
        --B 100 --seed 0 --schemes subject seed pair

The script tolerates missing cells (logs and skips); the audit table only
covers cells that actually have per-case Dice JSONs on disk.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger("m12_bootstrap")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PER_CASE_ROOT = REPO_ROOT / "results" / "per_case_dice"
OUT_ROOT = REPO_ROOT / "results" / "m12_bootstrap"

SCHEME_SUBJECT = "subject_bootstrap_B100"
SCHEME_SEED = "seed_bootstrap_B100"
SCHEME_PAIR = "pair_bootstrap_B100"

VALID_SCHEMES: tuple[str, ...] = (SCHEME_SUBJECT, SCHEME_SEED, SCHEME_PAIR)
SCHEME_ALIASES: dict[str, str] = {
    "subject": SCHEME_SUBJECT,
    "seed": SCHEME_SEED,
    "pair": SCHEME_PAIR,
    SCHEME_SUBJECT: SCHEME_SUBJECT,
    SCHEME_SEED: SCHEME_SEED,
    SCHEME_PAIR: SCHEME_PAIR,
}

CI_LO_PCT = 2.5
CI_HI_PCT = 97.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellDice:
    """Per-case Dice aggregated across released seeds for a (task, FM) cell.

    ``case_ids`` is the canonical, sorted list of case ids common across all
    released seeds. ``per_seed_dice`` is shaped ``(n_seeds, n_cases)`` and rows
    are aligned to ``seeds``.
    """

    task: str
    fm: str
    seeds: tuple[int, ...]
    case_ids: tuple[str, ...]
    per_seed_dice: np.ndarray  # shape (n_seeds, n_cases)
    source_files: tuple[str, ...]


def _load_seed_json(path: Path) -> tuple[list[str], list[float], int]:
    """Parse a SPEC §8 per-case Dice JSON.

    Returns ``(test_ids, per_case_dice, seed)``.
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    test_ids = list(payload["test_ids"])
    per_case = list(payload["per_case_dice"])
    seed = int(payload["seed"])
    if len(test_ids) != len(per_case):
        raise ValueError(
            f"{path}: len(test_ids)={len(test_ids)} != "
            f"len(per_case_dice)={len(per_case)}"
        )
    return test_ids, per_case, seed


def load_cell(task: str, fm: str) -> CellDice | None:
    """Load all released seeds for a (task, FM) cell.

    Aligns per-case Dice across seeds on the intersection of ``test_ids``.
    Returns ``None`` if no seed JSONs are found.
    """
    cell_dir = PER_CASE_ROOT / task / fm
    if not cell_dir.is_dir():
        return None
    seed_files = sorted(cell_dir.glob("seed_*.json"))
    if not seed_files:
        return None

    per_seed: dict[int, dict[str, float]] = {}
    for fp in seed_files:
        try:
            test_ids, per_case, seed = _load_seed_json(fp)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("skip malformed %s: %s", fp, exc)
            continue
        per_seed[seed] = dict(zip(test_ids, per_case))

    if not per_seed:
        return None

    common_ids: set[str] | None = None
    for d in per_seed.values():
        ids = set(d.keys())
        common_ids = ids if common_ids is None else common_ids & ids
    assert common_ids is not None
    if not common_ids:
        logger.warning("(%s, %s): no shared case ids across seeds", task, fm)
        return None

    sorted_ids = tuple(sorted(common_ids))
    seeds_sorted = tuple(sorted(per_seed.keys()))
    arr = np.array(
        [[per_seed[s][cid] for cid in sorted_ids] for s in seeds_sorted],
        dtype=np.float64,
    )
    return CellDice(
        task=task,
        fm=fm,
        seeds=seeds_sorted,
        case_ids=sorted_ids,
        per_seed_dice=arr,
        source_files=tuple(str(fp.relative_to(REPO_ROOT)) for fp in seed_files),
    )


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(arr, CI_LO_PCT))
    hi = float(np.percentile(arr, CI_HI_PCT))
    return lo, hi


def subject_bootstrap(cell: CellDice, B: int, seed: int) -> dict[str, float | list[float]]:
    """Resample case ids (with replacement) and estimate mean Dice.

    For each bootstrap, draw ``n_cases`` indices with replacement, average the
    Dice across seeds first (per case), then across the resampled cases.
    """
    rng = np.random.default_rng(seed)
    per_case_mean = cell.per_seed_dice.mean(axis=0)  # (n_cases,)
    n_cases = per_case_mean.shape[0]

    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_cases, size=n_cases)
        boots[b] = float(per_case_mean[idx].mean())

    lo, hi = _percentile_ci(boots)
    return {
        "dice_mean": float(boots.mean()),
        "dice_lo": lo,
        "dice_hi": hi,
        "point_estimate": float(per_case_mean.mean()),
        "boot_means": boots.tolist(),
    }


def seed_bootstrap(cell: CellDice, B: int, seed: int) -> dict[str, float]:
    """Resample seeds (with replacement) and estimate mean Dice.

    Per bootstrap: draw ``n_seeds`` seeds with replacement, average the seed
    means.
    """
    per_seed_mean = cell.per_seed_dice.mean(axis=1)  # (n_seeds,)
    n_seeds = per_seed_mean.shape[0]
    if n_seeds < 1:
        raise ValueError(f"{cell.task}/{cell.fm}: no seeds available")

    rng = np.random.default_rng(seed + 1)
    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boots[b] = float(per_seed_mean[idx].mean())

    lo, hi = _percentile_ci(boots)
    return {
        "dice_mean": float(boots.mean()),
        "dice_lo": lo,
        "dice_hi": hi,
        "point_estimate": float(per_seed_mean.mean()),
    }


def pair_bootstrap(cell_a: CellDice, cell_b: CellDice, B: int, seed: int) -> dict:
    """Paired Δ Dice (FM_A - FM_B) bootstrapped over shared case ids.

    Aligns ``cell_a`` and ``cell_b`` on their intersection of ``case_ids``;
    resamples those shared indices with replacement; per bootstrap computes
    the mean per-case Dice (averaged across seeds within each cell, then
    across cases) for both cells and takes the difference.
    """
    if cell_a.task != cell_b.task:
        raise ValueError(
            f"task mismatch in pair_bootstrap: {cell_a.task} vs {cell_b.task}"
        )

    shared = sorted(set(cell_a.case_ids) & set(cell_b.case_ids))
    if not shared:
        return {
            "B": 0,
            "n_shared_cases": 0,
            "paired_delta_mean": float("nan"),
            "paired_delta_lo": float("nan"),
            "paired_delta_hi": float("nan"),
            "point_delta": float("nan"),
        }

    a_pos = {cid: i for i, cid in enumerate(cell_a.case_ids)}
    b_pos = {cid: i for i, cid in enumerate(cell_b.case_ids)}
    a_idx = np.array([a_pos[c] for c in shared], dtype=np.int64)
    b_idx = np.array([b_pos[c] for c in shared], dtype=np.int64)
    a_per_case = cell_a.per_seed_dice[:, a_idx].mean(axis=0)  # (n_shared,)
    b_per_case = cell_b.per_seed_dice[:, b_idx].mean(axis=0)
    diff = a_per_case - b_per_case  # paired by case id

    rng = np.random.default_rng(seed + 2)
    n = diff.shape[0]
    boots = np.empty(B, dtype=np.float64)
    for b_iter in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b_iter] = float(diff[idx].mean())

    lo, hi = _percentile_ci(boots)
    return {
        "B": int(B),
        "n_shared_cases": int(n),
        "paired_delta_mean": float(boots.mean()),
        "paired_delta_lo": lo,
        "paired_delta_hi": hi,
        "point_delta": float(diff.mean()),
    }


# ---------------------------------------------------------------------------
# Ranking stability (subject scheme only)
# ---------------------------------------------------------------------------


def kendall_tau_ranking(boot_per_fm: dict[str, np.ndarray],
                        point_per_fm: dict[str, float]) -> dict[str, float]:
    """Per-FM Kendall's tau between bootstrap rank-vectors and the point rank-vector.

    For each bootstrap iteration, ranks all FMs by bootstrap mean Dice
    (descending). Computes Kendall's tau between that rank-vector and the
    point-estimate rank-vector. Returns the mean tau across bootstraps,
    keyed by FM (the same scalar is assigned to every FM on the same task —
    this is the task-level ranking stability that the FM cell inherits).
    """
    fms = sorted(boot_per_fm.keys())
    if len(fms) < 2:
        return {fm: float("nan") for fm in fms}

    boot_matrix = np.stack([boot_per_fm[fm] for fm in fms], axis=0)  # (n_fms, B)
    point_vec = np.array([point_per_fm[fm] for fm in fms], dtype=np.float64)

    # Higher Dice = better; rank descending. ties → average ranks via scipy.
    point_ranks = stats.rankdata(-point_vec, method="average")
    taus: list[float] = []
    for b in range(boot_matrix.shape[1]):
        boot_ranks_b = stats.rankdata(-boot_matrix[:, b], method="average")
        tau, _p = stats.kendalltau(boot_ranks_b, point_ranks)
        if not np.isnan(tau):
            taus.append(float(tau))
    mean_tau = float(np.mean(taus)) if taus else float("nan")
    return {fm: mean_tau for fm in fms}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj):  # type: ignore[no-untyped-def]
    """Recursively replace ``NaN``/``Inf`` floats with ``None`` for strict JSON.

    ``json.dump`` defaults to ``allow_nan=True`` which writes the non-standard
    literal ``NaN``; downstream loaders (e.g. ``simplejson``, browsers) reject
    that. We canonicalise to ``null`` so the audit JSONs remain RFC 8259 valid.
    """
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(payload), fh, indent=2, sort_keys=True,
                  allow_nan=False)


def _shared_case_intersection(cells: dict[str, CellDice]) -> tuple[str, ...]:
    """Sorted intersection of case ids across all FMs on a task."""
    common: set[str] | None = None
    for cell in cells.values():
        ids = set(cell.case_ids)
        common = ids if common is None else common & ids
    return tuple(sorted(common)) if common else tuple()


def _subject_bootstrap_with_shared_idx(cell: CellDice,
                                       shared_ids: tuple[str, ...],
                                       boot_idx: np.ndarray) -> np.ndarray:
    """Re-evaluate per-bootstrap mean Dice for one cell using a shared index
    vector defined on the task-level ``shared_ids`` axis.

    Returns an array of shape ``(B,)`` of bootstrap means. ``boot_idx`` has
    shape ``(B, len(shared_ids))``; the same draws are used for every FM so
    Kendall's tau across FMs is per-bootstrap case-aligned.
    """
    pos = {cid: i for i, cid in enumerate(cell.case_ids)}
    proj = np.array([pos[c] for c in shared_ids], dtype=np.int64)
    per_case_mean = cell.per_seed_dice[:, proj].mean(axis=0)  # (len(shared_ids),)
    # Vectorised: per_case_mean[boot_idx] -> (B, n_shared); mean across axis=1.
    return per_case_mean[boot_idx].mean(axis=1)


def run_subject_for_task(task: str, cells: dict[str, CellDice], B: int,
                         seed: int) -> list[dict]:
    """Run subject_bootstrap for every FM cell on a task and emit per-cell JSONs.

    For ranking-stability tau, all FMs share one resampled case-id vector per
    bootstrap. The per-cell ``dice_mean``/CI is still computed on each cell's
    own ``case_ids`` (so single-FM cells with extra cases keep their full
    sample), while tau is computed on the shared intersection only.
    """
    boot_per_fm: dict[str, np.ndarray] = {}
    point_per_fm: dict[str, float] = {}
    interim: dict[str, dict] = {}

    for fm, cell in cells.items():
        res = subject_bootstrap(cell, B=B, seed=seed)
        interim[fm] = res

    # Shared-index pass for Kendall tau (case-aligned across FMs).
    shared_ids = _shared_case_intersection(cells)
    if len(shared_ids) >= 1 and len(cells) >= 2:
        rng_shared = np.random.default_rng(seed + 7919)  # distinct stream
        n_shared = len(shared_ids)
        boot_idx = rng_shared.integers(0, n_shared, size=(B, n_shared))
        for fm, cell in cells.items():
            boot_per_fm[fm] = _subject_bootstrap_with_shared_idx(
                cell, shared_ids, boot_idx
            )
            shared_proj = np.array(
                [cell.case_ids.index(c) for c in shared_ids], dtype=np.int64
            )
            point_per_fm[fm] = float(
                cell.per_seed_dice[:, shared_proj].mean(axis=0).mean()
            )
    else:
        # No usable shared pool — tau is undefined.
        for fm in cells:
            boot_per_fm[fm] = np.zeros(0, dtype=np.float64)
            point_per_fm[fm] = float("nan")

    tau_per_fm = (
        kendall_tau_ranking(boot_per_fm, point_per_fm)
        if (len(shared_ids) >= 1 and len(cells) >= 2)
        else {fm: float("nan") for fm in cells}
    )
    rows: list[dict] = []
    for fm, res in interim.items():
        out = {
            "task": task,
            "fm": fm,
            "scheme": SCHEME_SUBJECT,
            "B": int(B),
            "seed": int(seed),
            "n_cases": int(cells[fm].per_seed_dice.shape[1]),
            "n_seeds": int(cells[fm].per_seed_dice.shape[0]),
            "dice_mean": res["dice_mean"],
            "dice_lo": res["dice_lo"],
            "dice_hi": res["dice_hi"],
            "point_estimate": res["point_estimate"],
            "ranking_kendall_tau_vs_pointest": tau_per_fm[fm],
            "source_files": list(cells[fm].source_files),
        }
        _write_json(OUT_ROOT / task / fm / f"{SCHEME_SUBJECT}.json", out)
        rows.append(out)
    return rows


def run_seed_for_task(task: str, cells: dict[str, CellDice], B: int,
                      seed: int) -> list[dict]:
    rows: list[dict] = []
    for fm, cell in cells.items():
        res = seed_bootstrap(cell, B=B, seed=seed)
        out = {
            "task": task,
            "fm": fm,
            "scheme": SCHEME_SEED,
            "B": int(B),
            "seed": int(seed),
            "n_cases": int(cell.per_seed_dice.shape[1]),
            "n_seeds": int(cell.per_seed_dice.shape[0]),
            "dice_mean": res["dice_mean"],
            "dice_lo": res["dice_lo"],
            "dice_hi": res["dice_hi"],
            "point_estimate": res["point_estimate"],
            "source_files": list(cell.source_files),
        }
        _write_json(OUT_ROOT / task / fm / f"{SCHEME_SEED}.json", out)
        rows.append(out)
    return rows


def run_pair_for_task(task: str, cells: dict[str, CellDice], B: int,
                      seed: int) -> list[dict]:
    """All ordered FM pairs for a task; one JSON per FM_A holds all FM_B
    deltas, one summary row per pair.
    """
    rows: list[dict] = []
    fms = sorted(cells.keys())
    for fm_a in fms:
        per_b: dict[str, dict] = {}
        for fm_b in fms:
            if fm_a == fm_b:
                continue
            res = pair_bootstrap(cells[fm_a], cells[fm_b], B=B, seed=seed)
            per_b[fm_b] = res
            rows.append({
                "task": task,
                "fm_a": fm_a,
                "fm_b": fm_b,
                "scheme": SCHEME_PAIR,
                "B": res["B"],
                "n_shared_cases": res["n_shared_cases"],
                "paired_delta_mean": res["paired_delta_mean"],
                "paired_delta_lo": res["paired_delta_lo"],
                "paired_delta_hi": res["paired_delta_hi"],
                "point_delta": res["point_delta"],
            })
        out = {
            "task": task,
            "fm": fm_a,
            "scheme": SCHEME_PAIR,
            "B": int(B),
            "seed": int(seed),
            "deltas_vs": per_b,
            "source_files": list(cells[fm_a].source_files),
        }
        _write_json(OUT_ROOT / task / fm_a / f"{SCHEME_PAIR}.json", out)
    return rows


def write_summary(all_rows: dict[str, list[dict]]) -> None:
    """Render `results/m12_bootstrap/SUMMARY.md` and SUMMARY.csv from rows."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# M12 Bootstrap-Bagging Audit\n")
    lines.append(
        "96 cells = 8 FM × 4 tasks × 3 schemes "
        "(subject_bootstrap_B100 / seed_bootstrap_B100 / pair_bootstrap_B100).\n"
    )

    for scheme_label, rows in (
        ("subject_bootstrap_B100", all_rows.get(SCHEME_SUBJECT, [])),
        ("seed_bootstrap_B100", all_rows.get(SCHEME_SEED, [])),
    ):
        lines.append(f"\n## {scheme_label}\n")
        lines.append("| task | fm | n_seeds | n_cases | dice_mean | 95% CI | point | tau |")
        lines.append("|------|----|---------|---------|-----------|--------|-------|-----|")
        for r in sorted(rows, key=lambda x: (x["task"], x["fm"])):
            tau = r.get("ranking_kendall_tau_vs_pointest", float("nan"))
            tau_str = "—" if (isinstance(tau, float) and np.isnan(tau)) else f"{tau:.3f}"
            lines.append(
                f"| {r['task']} | {r['fm']} | {r['n_seeds']} | {r['n_cases']} | "
                f"{r['dice_mean']:.4f} | "
                f"[{r['dice_lo']:.4f}, {r['dice_hi']:.4f}] | "
                f"{r['point_estimate']:.4f} | {tau_str} |"
            )

    pair_rows = all_rows.get(SCHEME_PAIR, [])
    if pair_rows:
        lines.append("\n## pair_bootstrap_B100\n")
        lines.append("| task | fm_a | fm_b | n_shared | Δ mean | 95% CI | point Δ |")
        lines.append("|------|------|------|----------|--------|--------|---------|")
        for r in sorted(pair_rows, key=lambda x: (x["task"], x["fm_a"], x["fm_b"])):
            lines.append(
                f"| {r['task']} | {r['fm_a']} | {r['fm_b']} | {r['n_shared_cases']} | "
                f"{r['paired_delta_mean']:+.4f} | "
                f"[{r['paired_delta_lo']:+.4f}, {r['paired_delta_hi']:+.4f}] | "
                f"{r['point_delta']:+.4f} |"
            )

    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_lines = ["task,fm,scheme,fm_b,B,n_seeds,n_cases,dice_mean,dice_lo,dice_hi,"
                 "point,delta_mean,delta_lo,delta_hi,point_delta,kendall_tau"]
    for r in all_rows.get(SCHEME_SUBJECT, []):
        csv_lines.append(
            f"{r['task']},{r['fm']},{SCHEME_SUBJECT},,{r['B']},{r['n_seeds']},"
            f"{r['n_cases']},{r['dice_mean']:.6f},{r['dice_lo']:.6f},"
            f"{r['dice_hi']:.6f},{r['point_estimate']:.6f},,,,,"
            f"{r['ranking_kendall_tau_vs_pointest']:.6f}"
        )
    for r in all_rows.get(SCHEME_SEED, []):
        csv_lines.append(
            f"{r['task']},{r['fm']},{SCHEME_SEED},,{r['B']},{r['n_seeds']},"
            f"{r['n_cases']},{r['dice_mean']:.6f},{r['dice_lo']:.6f},"
            f"{r['dice_hi']:.6f},{r['point_estimate']:.6f},,,,,"
        )
    for r in all_rows.get(SCHEME_PAIR, []):
        csv_lines.append(
            f"{r['task']},{r['fm_a']},{SCHEME_PAIR},{r['fm_b']},{r['B']},,"
            f"{r['n_shared_cases']},,,,,"
            f"{r['paired_delta_mean']:.6f},{r['paired_delta_lo']:.6f},"
            f"{r['paired_delta_hi']:.6f},{r['point_delta']:.6f},"
        )
    (OUT_ROOT / "SUMMARY.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M12 bootstrap-bagging audit.")
    p.add_argument("--tasks", nargs="+", required=True,
                   help="task names (kvasir acdc_lv riga_cup brats_wt).")
    p.add_argument("--fms", nargs="+", required=True, help="FM names (8 FMs).")
    p.add_argument("--B", type=int, default=100, help="bootstrap iterations per cell.")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed.")
    p.add_argument("--schemes", nargs="+", default=["subject", "seed", "pair"],
                   help="any of: subject seed pair (or full names).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    schemes: list[str] = []
    for s in args.schemes:
        canon = SCHEME_ALIASES.get(s)
        if canon is None:
            raise SystemExit(f"unknown scheme {s!r}; expected one of {VALID_SCHEMES}")
        if canon not in schemes:
            schemes.append(canon)

    all_rows: dict[str, list[dict]] = {s: [] for s in VALID_SCHEMES}

    for task in args.tasks:
        cells: dict[str, CellDice] = {}
        for fm in args.fms:
            cell = load_cell(task, fm)
            if cell is None:
                logger.warning("missing cell: task=%s fm=%s", task, fm)
                continue
            cells[fm] = cell
        if not cells:
            logger.warning("task=%s: no cells loaded; skipping", task)
            continue

        if SCHEME_SUBJECT in schemes:
            all_rows[SCHEME_SUBJECT].extend(
                run_subject_for_task(task, cells, B=args.B, seed=args.seed)
            )
        if SCHEME_SEED in schemes:
            all_rows[SCHEME_SEED].extend(
                run_seed_for_task(task, cells, B=args.B, seed=args.seed)
            )
        if SCHEME_PAIR in schemes:
            all_rows[SCHEME_PAIR].extend(
                run_pair_for_task(task, cells, B=args.B, seed=args.seed)
            )

    write_summary(all_rows)
    logger.info("M12 wrote %d subject + %d seed + %d pair rows to %s",
                len(all_rows[SCHEME_SUBJECT]), len(all_rows[SCHEME_SEED]),
                len(all_rows[SCHEME_PAIR]), OUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
