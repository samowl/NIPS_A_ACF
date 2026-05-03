"""Correlation, filter, M_eff, and bootstrap estimators for FM-pool analysis.

Consolidates SPEC §6 (single source of truth for all estimators). This module
keeps the legacy scorer semantics where valid and makes the following changes:

* ``pearson_rbar`` returns NaN (not 0.0) when either input has std < eps. The
  NaN-exclude convention is specified in SPEC §6 for low-std pairs.
* ``hierarchical_bootstrap`` honours ``stat="icc"`` instead of falling back to
  Pearson.
* ``compute_m_eff`` returns NaN (not 1.0) when ``rho_fail`` is undefined
  (all-fail pool, zero variance in the failure indicator). SPEC §6 adopts the
  NaN-exclude convention for grand-mean aggregation.

References
----------
- McGraw & Wong (1996) "Forming inferences about some intraclass correlation
  coefficients." Psychological Methods 1(1):30-46.
- Kuncheva & Whitaker (2003) "Measures of diversity in classifier ensembles."
  Machine Learning 51(2):181-207.
- Davison & Hinkley (1997) "Bootstrap Methods and Their Application."
- Field & Welsh (2007) "Bootstrapping clustered data." JRSS-B 69(3):369-390.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

FamilyKey = tuple[str, int]
PerCaseDice = dict[FamilyKey, np.ndarray]


# ---------------------------------------------------------------------------
# Per-case Dice
# ---------------------------------------------------------------------------


def dice_per_case(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """SPEC §6: per-case Dice with the ``Dice(∅, ∅) = 1`` convention.

    ``pred`` and ``target`` are expected in shape ``[N, ...]`` where the leading
    axis indexes cases. Values must be binary (0/1) or boolean. Returns a 1-D
    array of length ``N`` with per-case Dice coefficients.

    The ``Dice(∅, ∅) = 1`` convention is applied *before* any empty-empty
    filter (SPEC §6).
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred.shape} vs {target.shape}"
        )
    if pred.ndim < 1:
        raise ValueError("expected leading case axis; got scalar input")

    n_cases = pred.shape[0]
    out = np.empty(n_cases, dtype=np.float64)
    flat_pred = pred.reshape(n_cases, -1)
    flat_target = target.reshape(n_cases, -1)
    for i in range(n_cases):
        p_sum = float(flat_pred[i].sum())
        t_sum = float(flat_target[i].sum())
        if p_sum == 0.0 and t_sum == 0.0:
            out[i] = 1.0
            continue
        inter = float(np.logical_and(flat_pred[i], flat_target[i]).sum())
        out[i] = (2.0 * inter) / (p_sum + t_sum + 1e-6)
    return out


# ---------------------------------------------------------------------------
# Correlation estimators
# ---------------------------------------------------------------------------


def pearson_rbar(x: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float],
                 eps: float = 1e-8) -> float:
    """SPEC §6: Pearson correlation with NaN on low-std inputs.

    Returns ``float('nan')`` when either input's standard deviation is below
    ``eps``. Callers filter out NaNs before averaging (NaN-exclude, SPEC §6).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: {x_arr.shape} vs {y_arr.shape}")
    if x_arr.size < 2:
        return float("nan")
    if np.std(x_arr) < eps or np.std(y_arr) < eps:
        return float("nan")
    r, _ = stats.pearsonr(x_arr, y_arr)
    return float(r)


def icc_a1(x: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> float:
    """SPEC §6: ICC(A,1) two-way absolute-agreement single-measurement.

    McGraw-Wong (1996) Case 2A, k=2 raters × n subjects. Returns NaN when the
    denominator is non-positive (very low between-subject variance).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(x_arr)
    if n < 2:
        return float("nan")
    subj_mean = (x_arr + y_arr) / 2.0
    grand = float(subj_mean.mean())
    rater_mean = np.array([x_arr.mean(), y_arr.mean()])
    ss_subject = 2.0 * float(np.sum((subj_mean - grand) ** 2))
    ss_rater = n * float(np.sum((rater_mean - grand) ** 2))
    ss_total = float(np.sum((x_arr - grand) ** 2) + np.sum((y_arr - grand) ** 2))
    ss_error = ss_total - ss_subject - ss_rater
    bms = ss_subject / max(n - 1, 1)
    rms = ss_rater  # df = k - 1 = 1
    ems = ss_error / max(n - 1, 1)
    denom = bms + ems + 2 * (rms - ems) / n
    if denom <= 0:
        return float("nan")
    return float((bms - ems) / denom)


# ---------------------------------------------------------------------------
# Pair-matrix and within/cross r-bar
# ---------------------------------------------------------------------------


def _stat_fn(stat: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if stat == "pearson":
        return pearson_rbar
    if stat == "icc":
        return icc_a1
    raise ValueError(f"stat must be 'pearson' or 'icc', got {stat!r}")


def _pearson_matrix(
    per_case_dice: PerCaseDice,
    keys: list[FamilyKey],
    eps: float = 1e-8,
) -> np.ndarray:
    """Vectorized Pearson matrix with the same NaN convention as pearson_rbar."""
    n = len(keys)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return matrix
    data = np.stack([np.asarray(per_case_dice[k], dtype=np.float64) for k in keys])
    if data.shape[1] < 2:
        return matrix
    valid = np.std(data, axis=1) >= eps
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return matrix
    corr = np.corrcoef(data[valid_idx])
    if corr.ndim == 0:
        corr = np.asarray([[float(corr)]], dtype=np.float64)
    matrix[np.ix_(valid_idx, valid_idx)] = corr
    return matrix


def family_pair_matrix(per_case_dice: PerCaseDice,
                       stat: str = "pearson") -> tuple[list[FamilyKey], np.ndarray]:
    """SPEC §6: pairwise correlation matrix over every (FM, seed) combination.

    Returns ``(keys, matrix)`` where ``keys`` is the sorted list of dict keys
    and ``matrix[i, j]`` is the pairwise statistic between ``keys[i]`` and
    ``keys[j]``.
    """
    keys = sorted(per_case_dice.keys())
    n = len(keys)
    if stat == "pearson":
        return keys, _pearson_matrix(per_case_dice, keys)
    fn = _stat_fn(stat)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            a = np.asarray(per_case_dice[keys[i]])
            b = np.asarray(per_case_dice[keys[j]])
            matrix[i, j] = matrix[j, i] = fn(a, b)
    return keys, matrix


def within_cross_rbar(per_case_dice: PerCaseDice,
                      family_pairs_within: set[frozenset[str]] | None = None,
                      stat: str = "pearson") -> dict[str, float | int | str]:
    """SPEC §6: symmetric multi-seed within- and cross-family r̄.

    Parameters
    ----------
    per_case_dice
        Mapping ``{(family, seed): 1-D array of per-case Dice}``.
    family_pairs_within
        Optional set of frozensets of family labels to count as within-family
        even if they appear as different family labels (SPEC §3 treats
        DINOv2-B × DINOv2-S as within).
    stat
        ``"pearson"`` or ``"icc"``.

    Returns
    -------
    dict
        Keys ``within``, ``cross``, ``gap``, ``n_within_pairs``,
        ``n_cross_pairs``, ``stat``. NaN-valued pairs are excluded from the
        mean (SPEC §6 NaN-exclude convention).
    """
    if family_pairs_within is None:
        family_pairs_within = set()

    families = sorted({k[0] for k in per_case_dice})
    seeds_by_fam = {
        f: sorted({k[1] for k in per_case_dice if k[0] == f}) for f in families
    }

    within: list[float] = []
    cross: list[float] = []

    if stat == "pearson":
        keys, matrix = family_pair_matrix(per_case_dice, stat="pearson")
        for i, (fa, _sa) in enumerate(keys):
            for j in range(i + 1, len(keys)):
                fb, _sb = keys[j]
                value = float(matrix[i, j])
                is_within_pair = (
                    fa == fb or frozenset({fa, fb}) in family_pairs_within
                )
                (within if is_within_pair else cross).append(value)
    else:
        fn = _stat_fn(stat)

        # same-family, different-seed pairs
        for fam in families:
            seeds = seeds_by_fam[fam]
            for i, s1 in enumerate(seeds):
                for s2 in seeds[i + 1:]:
                    a = per_case_dice[(fam, s1)]
                    b = per_case_dice[(fam, s2)]
                    within.append(fn(np.asarray(a), np.asarray(b)))

        # different-family pairs (all seed × seed combinations)
        for i, fa in enumerate(families):
            for fb in families[i + 1:]:
                is_within_pair = frozenset({fa, fb}) in family_pairs_within
                bucket = within if is_within_pair else cross
                for sa in seeds_by_fam[fa]:
                    for sb in seeds_by_fam[fb]:
                        a = per_case_dice[(fa, sa)]
                        b = per_case_dice[(fb, sb)]
                        bucket.append(fn(np.asarray(a), np.asarray(b)))

    within_clean = [v for v in within if not np.isnan(v)]
    cross_clean = [v for v in cross if not np.isnan(v)]

    w_mean = float(np.mean(within_clean)) if within_clean else float("nan")
    c_mean = float(np.mean(cross_clean)) if cross_clean else float("nan")
    gap = w_mean - c_mean if (within_clean and cross_clean) else float("nan")

    return {
        "within": w_mean,
        "cross": c_mean,
        "gap": gap,
        "n_within_pairs": len(within_clean),
        "n_cross_pairs": len(cross_clean),
        "stat": stat,
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def functional_floor_filter(per_case_dice: PerCaseDice,
                            min_mean_dice: float = 0.30,
                            scope: str = "per_fm") -> PerCaseDice:
    """SPEC §6: drop pool members whose mean Dice < ``min_mean_dice``.

    Parameters
    ----------
    per_case_dice
        Mapping ``{(fm, seed): 1-D array of per-case Dice}``.
    min_mean_dice
        Functional-floor threshold. SPEC §6 default is 0.30.
    scope
        ``"per_fm"`` (paper convention): drop an entire FM across all seeds
        when its cross-seed mean falls below the floor. ``"per_fm_seed"``:
        drop only the individual (FM, seed) pairs that fail.
    """
    typed = {k: np.asarray(v, dtype=np.float64) for k, v in per_case_dice.items()}

    if scope == "per_fm_seed":
        return {
            k: v for k, v in typed.items()
            if float(np.mean(v)) >= float(min_mean_dice)
        }
    if scope != "per_fm":
        raise ValueError(f"scope must be 'per_fm' or 'per_fm_seed', got {scope!r}")

    by_fm: dict[str, list[np.ndarray]] = {}
    for (fm, _seed), arr in typed.items():
        by_fm.setdefault(fm, []).append(arr)

    functional_fms: set[str] = set()
    for fm, seed_arrs in by_fm.items():
        fm_mean = float(np.mean([float(np.mean(a)) for a in seed_arrs]))
        if fm_mean >= float(min_mean_dice):
            functional_fms.add(fm)

    return {k: v for k, v in typed.items() if k[0] in functional_fms}


def empty_empty_filter(per_case_dice: PerCaseDice,
                       criterion: str = "unanimous",
                       thresh: float = 0.995,
                       max_thresh: float = 0.99) -> np.ndarray:
    """SPEC §6: boolean mask of cases to KEEP (True = not trivial).

    Criteria:
    * ``unanimous``: drop cases where MIN over all (FM, seed) >= thresh.
    * ``majority``: drop cases where >=90% of (FM, seed) > thresh.
    * ``all_ceiling``: drop cases where MAX over all (FM, seed) >= max_thresh.
    """
    stacked = np.stack([np.asarray(v) for v in per_case_dice.values()], axis=1)
    if criterion == "unanimous":
        return stacked.min(axis=1) < thresh
    if criterion == "majority":
        return (stacked > thresh).mean(axis=1) < 0.9
    if criterion == "all_ceiling":
        return stacked.max(axis=1) < max_thresh
    raise ValueError(f"unknown criterion {criterion!r}")


def sparse_class_sensitivity_sweep(per_case_dice: PerCaseDice,
                                   statistic_fn: Callable[[PerCaseDice], object]) -> list[dict]:
    """SPEC §6: apply canonical empty-empty filters and evaluate a statistic.

    Returns a list of ``{filter, n_kept, value}`` dicts, one per filter,
    including an ``all_cases`` no-filter baseline.
    """
    out: list[dict] = [{
        "filter": "all_cases",
        "n_kept": len(next(iter(per_case_dice.values()))),
        "value": statistic_fn(per_case_dice),
    }]
    for criterion, kwargs in [
        ("unanimous", {"thresh": 0.995}),
        ("majority", {"thresh": 0.99}),
        ("all_ceiling", {"max_thresh": 0.99}),
    ]:
        mask = empty_empty_filter(per_case_dice, criterion=criterion, **kwargs)
        kept = {k: np.asarray(v)[mask] for k, v in per_case_dice.items()}
        out.append({
            "filter": criterion,
            "n_kept": int(mask.sum()),
            "value": statistic_fn(kept),
        })
    return out


# ---------------------------------------------------------------------------
# M_eff
# ---------------------------------------------------------------------------


def compute_m_eff(per_case_dice: PerCaseDice, tau: float = 0.85) -> dict[str, float | int]:
    """SPEC §6: effective number of independent FM-seeds.

    ``M_eff = M / (1 + (M-1) * rho_fail)`` where ``rho_fail`` is the mean
    pairwise Pearson correlation of binary failure indicators
    ``(D_i < tau)`` at threshold ``tau`` (default 0.85).

    Returns ``M_eff = NaN`` when ``rho_fail`` is undefined (all-fail pool with
    zero variance in every failure indicator). Callers should exclude NaN
    values from the grand-mean (SPEC §6 NaN-exclude convention).
    """
    keys = sorted(per_case_dice.keys())
    m = len(keys)
    if m < 2:
        return {
            "rho_fail": float("nan"),
            "M_eff": float("nan"),
            "n_fm_seed": int(m),
        }

    # Binary failure indicators, shape [M, N]. Reuse the vectorized Pearson
    # convention so scorer-level 4-FM composition sweeps remain tractable.
    fail_map = {
        k: (np.asarray(per_case_dice[k]) < tau).astype(np.float64)
        for k in keys
    }
    matrix = _pearson_matrix(fail_map, keys)
    pairwise = matrix[np.triu_indices(m, k=1)]
    pairwise = pairwise[~np.isnan(pairwise)]

    if pairwise.size == 0:
        logger.debug("compute_m_eff: all pairwise failure correlations NaN (M=%d)", m)
        return {
            "rho_fail": float("nan"),
            "M_eff": float("nan"),
            "n_fm_seed": int(m),
        }

    rho_fail = float(np.mean(pairwise))
    denom = 1.0 + (m - 1) * rho_fail
    m_eff = float(m) / denom if denom != 0 else float("nan")
    return {
        "rho_fail": rho_fail,
        "M_eff": float(m_eff),
        "n_fm_seed": int(m),
    }


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _ci95(arr: list[float]) -> tuple[float, float]:
    if not arr:
        return (float("nan"), float("nan"))
    a = np.asarray(arr)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def case_bootstrap(per_case_dice: PerCaseDice,
                   family_pairs_within: set[frozenset[str]] | None = None,
                   n_boot: int = 2000,
                   seed: int = 0,
                   stat: str = "pearson") -> dict:
    """SPEC §6: case-level bootstrap — resample test-case indices with replacement.

    Returns the point estimate from :func:`within_cross_rbar` and 95% percentile
    CIs for ``within``, ``cross``, ``gap``.
    """
    rng = np.random.default_rng(seed)
    keys = list(per_case_dice.keys())
    n = len(next(iter(per_case_dice.values())))
    boots: dict[str, list[float]] = {"within": [], "cross": [], "gap": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sampled = {k: np.asarray(per_case_dice[k])[idx] for k in keys}
        r = within_cross_rbar(sampled, family_pairs_within=family_pairs_within, stat=stat)
        for key in boots:
            val = r[key]
            if isinstance(val, (int, float)) and not np.isnan(val):
                boots[key].append(float(val))

    point = within_cross_rbar(per_case_dice,
                              family_pairs_within=family_pairs_within,
                              stat=stat)
    return {
        "point": point,
        "ci95": {k: _ci95(v) for k, v in boots.items()},
        "n_boot": int(n_boot),
        "stat": stat,
    }


def subject_cluster_bootstrap(per_case_dice: PerCaseDice,
                              case_to_subject: Iterable[str],
                              family_pairs_within: set[frozenset[str]] | None = None,
                              n_boot: int = 2000,
                              seed: int = 0,
                              stat: str = "pearson") -> dict:
    """SPEC §6: cluster bootstrap over higher-level case units.

    Resamples subjects/patients with replacement, then takes all 2-D cases
    belonging to each sampled subject. Use this when the released test unit is a
    slice but inference should respect patient/volume clustering (ACDC, BraTS).
    """
    case_to_subject_arr = np.asarray(list(case_to_subject))
    n_cases = len(next(iter(per_case_dice.values())))
    if len(case_to_subject_arr) != n_cases:
        raise ValueError(
            "case_to_subject length mismatch: "
            f"{len(case_to_subject_arr)} vs {n_cases}"
        )

    unique_subjects = np.array(sorted(set(case_to_subject_arr.tolist())))
    subject_to_cases = {
        subject: np.where(case_to_subject_arr == subject)[0]
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {"within": [], "cross": [], "gap": []}

    for _ in range(n_boot):
        sampled_subjects = rng.choice(
            unique_subjects, size=len(unique_subjects), replace=True
        )
        case_idx = np.concatenate(
            [subject_to_cases[subject] for subject in sampled_subjects]
        )
        sampled = {k: np.asarray(v)[case_idx] for k, v in per_case_dice.items()}
        r = within_cross_rbar(
            sampled, family_pairs_within=family_pairs_within, stat=stat
        )
        for key in boots:
            val = r[key]
            if isinstance(val, (int, float)) and not np.isnan(val):
                boots[key].append(float(val))

    point = within_cross_rbar(
        per_case_dice, family_pairs_within=family_pairs_within, stat=stat
    )
    return {
        "point": point,
        "ci95": {k: _ci95(v) for k, v in boots.items()},
        "n_boot": int(n_boot),
        "stat": stat,
        "n_subjects": int(len(unique_subjects)),
    }


def hierarchical_bootstrap(per_case_dice: PerCaseDice,
                           families: list[str] | None = None,
                           n_boot: int = 2000,
                           seed: int = 0,
                           stat: str = "pearson") -> dict:
    """SPEC §6: 3-level nested bootstrap (families → seeds → cases).

    Implementation note: this path honours ``stat="icc"`` directly instead of
    reverting to Pearson. Both paths are selected via :func:`_stat_fn`.

    ``families`` overrides the family list discovered in ``per_case_dice``; if
    ``None``, it defaults to ``sorted({k[0] for k in per_case_dice})``.
    """
    rng = np.random.default_rng(seed)
    if families is None:
        families = sorted({k[0] for k in per_case_dice})
    else:
        families = list(families)

    seeds_by_fam = {
        f: sorted({k[1] for k in per_case_dice if k[0] == f}) for f in families
    }
    n = len(next(iter(per_case_dice.values())))

    boots: dict[str, list[float]] = {"within": [], "cross": [], "gap": []}
    for _ in range(n_boot):
        case_idx = rng.integers(0, n, n)
        sampled_fams = rng.choice(families, size=len(families), replace=True)
        sampled: dict[tuple[str, int], np.ndarray] = {}
        draw_id = 0
        for fam in sampled_fams:
            fam_seeds = seeds_by_fam[fam]
            if not fam_seeds:
                continue
            sampled_seeds = rng.choice(fam_seeds, size=len(fam_seeds), replace=True)
            for s in sampled_seeds:
                # Keep the family label intact for within/cross bucketing, but
                # use a draw-local seed ID so duplicate bootstrap draws do not
                # collide in the dict.
                sampled[(str(fam), draw_id)] = (
                    np.asarray(per_case_dice[(fam, int(s))])[case_idx]
                )
                draw_id += 1

        r = within_cross_rbar(sampled, stat=stat)
        if not np.isnan(r["within"]) and not np.isnan(r["cross"]):
            w = float(r["within"])
            c = float(r["cross"])
            boots["within"].append(w)
            boots["cross"].append(c)
            boots["gap"].append(w - c)

    point = within_cross_rbar(per_case_dice, stat=stat)
    return {
        "point": point,
        "ci95": {k: _ci95(v) for k, v in boots.items()},
        "n_boot": int(n_boot),
        "stat": stat,
    }


def paired_delta_bootstrap(per_case_a: PerCaseDice,
                           per_case_b: PerCaseDice,
                           family_pairs_within: set[frozenset[str]] | None = None,
                           n_boot: int = 2000,
                           seed: int = 0,
                           stat: str = "pearson") -> dict:
    """SPEC §6: paired-Δ case bootstrap on two pools sharing case indices.

    Resamples case indices with replacement and computes
    ``Δ_A - Δ_B = (within_A - cross_A) - (within_B - cross_B)`` each iteration.
    Returns point estimates and a 95% percentile CI for the delta-of-deltas.
    """
    rng = np.random.default_rng(seed + 1)
    n_a = len(next(iter(per_case_a.values())))
    n_b = len(next(iter(per_case_b.values())))
    if n_a != n_b:
        raise ValueError(
            f"paired_delta_bootstrap expects identical case counts, got {n_a} vs {n_b}"
        )

    deltas_ab: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_a, n_a)
        sampled_a = {k: np.asarray(v)[idx] for k, v in per_case_a.items()}
        sampled_b = {k: np.asarray(v)[idx] for k, v in per_case_b.items()}
        ra = within_cross_rbar(sampled_a, family_pairs_within=family_pairs_within, stat=stat)
        rb = within_cross_rbar(sampled_b, family_pairs_within=family_pairs_within, stat=stat)
        gap_a = ra["gap"]
        gap_b = rb["gap"]
        if (isinstance(gap_a, float) and not np.isnan(gap_a)
                and isinstance(gap_b, float) and not np.isnan(gap_b)):
            deltas_ab.append(float(gap_a) - float(gap_b))

    point_a = within_cross_rbar(per_case_a,
                                family_pairs_within=family_pairs_within,
                                stat=stat)
    point_b = within_cross_rbar(per_case_b,
                                family_pairs_within=family_pairs_within,
                                stat=stat)
    gap_a = point_a["gap"]
    gap_b = point_b["gap"]
    if (isinstance(gap_a, float) and not np.isnan(gap_a)
            and isinstance(gap_b, float) and not np.isnan(gap_b)):
        delta_of_deltas = float(gap_a) - float(gap_b)
    else:
        delta_of_deltas = float("nan")
    return {
        "point_a": point_a,
        "point_b": point_b,
        "delta_of_deltas": delta_of_deltas,
        "delta_ci95": _ci95(deltas_ab),
        "n_boot": int(n_boot),
        "stat": stat,
    }
