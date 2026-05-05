#!/usr/bin/env python
"""Summarise case-identical RIGA nnU-Net held-out traces.

The input is the output directory produced by ``run_nnunet.sh
case_riga_100ep`` or by ``run_nnunet_case_worker.sh``:

    results/nnunet/nnunet_case_riga_2d_100ep/riga_cup_fold{F}_seed{S}.json
    results/nnunet/nnunet_case_riga_2d_1000ep/riga_cup_fold{F}_seed{S}.json

Each file contains per-case Dice on the same Magrabia held-out cases used by
the primary frozen-encoder RIGA rows. The summary computes same-fold
different-seed and cross-fold pairwise Pearson correlations, with a fixed
case-bootstrap interval for the gap.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("fmpool.compute_nnunet_case_identical")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SEEDS = (13, 37)
FILE_RE = re.compile(r"^riga_cup_fold(?P<fold>[0-4])_seed(?P<seed>13|37)\.json$")


@dataclass(frozen=True)
class Trace:
    path: Path
    fm: str
    fold: int
    seed: int
    ids: tuple[str, ...]
    dice: np.ndarray
    mean_dice: float
    elapsed_s: float | None


def _git_commit_sha(cwd: Path) -> str:
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
    except Exception:  # noqa: BLE001
        return "not_recorded_for_artifact"


def _iso_utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strict(value):
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _load(path: Path) -> Trace:
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: expected canonical filename "
            "riga_cup_fold{0..4}_seed{13|37}.json"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"task", "fm", "fold", "seed", "n_test", "test_ids", "per_case_dice"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if payload["task"] != "riga_cup":
        raise ValueError(f"{path}: expected task=riga_cup, got {payload['task']!r}")
    fold = int(payload["fold"])
    seed = int(payload["seed"])
    if fold != int(match.group("fold")) or seed != int(match.group("seed")):
        raise ValueError(
            f"{path}: filename fold/seed disagree with payload "
            f"fold={fold} seed={seed}"
        )
    ids = tuple(str(x) for x in payload["test_ids"])
    dice = np.asarray(payload["per_case_dice"], dtype=np.float64)
    if int(payload["n_test"]) != len(ids) or len(ids) != len(dice):
        raise ValueError(
            f"{path}: inconsistent lengths n_test={payload['n_test']} "
            f"len(test_ids)={len(ids)} len(per_case_dice)={len(dice)}"
        )
    if len(dice) == 0 or not np.all(np.isfinite(dice)):
        raise ValueError(f"{path}: per_case_dice must be nonempty and finite")
    return Trace(
        path=path,
        fm=str(payload["fm"]),
        fold=fold,
        seed=seed,
        ids=ids,
        dice=dice,
        mean_dice=float(payload.get("mean_dice", np.mean(dice))),
        elapsed_s=(
            float(payload["training_elapsed_s"])
            if payload.get("training_elapsed_s") is not None
            else None
        ),
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _point(traces: list[Trace], idx: np.ndarray | None = None) -> dict:
    within: list[float] = []
    cross: list[float] = []
    for i, a in enumerate(traces):
        av = a.dice if idx is None else a.dice[idx]
        for b in traces[i + 1:]:
            bv = b.dice if idx is None else b.dice[idx]
            r = _pearson(av, bv)
            if math.isnan(r):
                continue
            (within if a.fold == b.fold else cross).append(r)
    return {
        "within_rbar": float(np.mean(within)) if within else float("nan"),
        "cross_rbar": float(np.mean(cross)) if cross else float("nan"),
        "gap": (
            float(np.mean(within) - np.mean(cross))
            if within and cross else float("nan")
        ),
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
        "within_pairs": within,
        "cross_pairs": cross,
    }


def _case_bootstrap_gap(traces: list[Trace]) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(traces[0].dice)
    gaps: list[float] = []
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        gap = _point(traces, idx=idx)["gap"]
        if math.isfinite(gap):
            gaps.append(float(gap))
    if not gaps:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(np.asarray(gaps), [2.5, 97.5])
    return [float(lo), float(hi)]


def _protocol_label(input_dir: Path) -> str:
    name = input_dir.name
    if "1000ep" in name:
        return "1000-epoch"
    if "100ep" in name:
        return "100-epoch"
    return "seeded"


def _expected_fm(input_dir: Path) -> str:
    name = input_dir.name
    if "1000ep" in name:
        return "nnunet_2d_1000ep"
    if "100ep" in name:
        return "nnunet_2d_100ep"
    raise ValueError(
        f"cannot infer expected nnU-Net fm label from input directory {input_dir}"
    )


def summarise(input_dir: Path, output: Path, repo_root: Path) -> dict:
    traces = [_load(p) for p in sorted(input_dir.glob("riga_cup_fold*_seed*.json"))]
    if not traces:
        raise FileNotFoundError(f"no riga_cup_fold*_seed*.json under {input_dir}")
    expected_cells = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    counts = Counter((tr.fold, tr.seed) for tr in traces)
    duplicates = sorted(cell for cell, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(
            "duplicate RIGA case-identical nnU-Net cells are not allowed: "
            f"{duplicates}"
        )
    observed_cells = set(counts)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(
            "incomplete or unexpected RIGA case-identical nnU-Net grid: "
            f"missing={missing}, extra={extra}, observed={sorted(observed_cells)}"
        )
    expected_fm = _expected_fm(input_dir)
    bad_fm = sorted((str(tr.path), tr.fm) for tr in traces if tr.fm != expected_fm)
    if bad_fm:
        raise ValueError(f"unexpected fm labels for {input_dir}: {bad_fm}")
    first_ids = traces[0].ids
    for tr in traces:
        if tr.ids != first_ids:
            raise ValueError(f"test_ids mismatch: {tr.path} disagrees with {traces[0].path}")
    point = _point(traces)
    protocol_label = _protocol_label(input_dir)
    payload = {
        "schema_version": "fmpool_clean_v1",
        "generated_at": _iso_utc_now(),
        "commit_sha": _git_commit_sha(repo_root),
        "protocol": (
            f"Case-identical RIGA Cup nnU-Net v2 2D {protocol_label} scope check; "
            "train folds use BinRushed+MESSIDOR and held-out evaluation uses "
            "the primary Magrabia cases."
        ),
        "dataset": "RIGA_cup",
        "task": "riga_cup",
        "fm": expected_fm,
        "case_unit": "fundus image",
        "n_test": len(first_ids),
        "test_source": "Magrabia",
        "n_models": len(traces),
        "n_models_trained": len(traces),
        "folds": sorted({tr.fold for tr in traces}),
        "seeds": sorted({tr.seed for tr in traces}),
        "within_fold_rbar": point["within_rbar"],
        "cross_fold_rbar": point["cross_rbar"],
        "delta_within_minus_cross": point["gap"],
        "delta_ci95_case_bootstrap": _case_bootstrap_gap(traces),
        "delta_ci95_method": "case bootstrap over held-out cases",
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_within_pairs": point["n_within_pairs"],
        "n_cross_pairs": point["n_cross_pairs"],
        "mean_dice_all": float(np.mean([tr.mean_dice for tr in traces])),
        "mean_dice_std": float(np.std([tr.mean_dice for tr in traces], ddof=1))
        if len(traces) > 1 else 0.0,
        "mean_dice_per_model": {
            f"fold{tr.fold}_seed{tr.seed}": tr.mean_dice for tr in traces
        },
        "per_model_mean_dice": {
            f"fold{tr.fold}_seed{tr.seed}": tr.mean_dice for tr in traces
        },
        "case_ids": list(first_ids),
        "per_case_dice": {
            f"fold{tr.fold}_seed{tr.seed}": [float(x) for x in tr.dice]
            for tr in traces
        },
        "training_elapsed_s_per_model": {
            f"fold{tr.fold}_seed{tr.seed}": tr.elapsed_s for tr in traces
        },
        "within_pairs": point["within_pairs"],
        "cross_pairs": point["cross_pairs"],
        "note": (
            "Case-identical full-pipeline scope check on the primary RIGA "
            "Magrabia held-out cases. This is a single-task 2D nnU-Net "
            "diagnostic, not an official nnU-Net benchmark or a replacement "
            "for the frozen-encoder primary evidence."
        ),
        "source_files": [str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)
                         for p in sorted(tr.path for tr in traces)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/nnunet/nnunet_case_riga_2d_100ep"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/_merged/nnunet/riga_case_identical_2d_100ep.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    payload = summarise(args.input_dir, args.out, args.repo_root)
    LOGGER.info(
        "wrote %s: n=%d models=%d gap=%.4f",
        args.out,
        payload["n_test"],
        payload["n_models"],
        payload["delta_within_minus_cross"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
