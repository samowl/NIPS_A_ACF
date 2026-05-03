#!/usr/bin/env python3
"""One-command artifact smoke test.

Runs the released CPU-only summary scorers into a scratch directory and
compares their outputs with the bundled JSON summaries. The test intentionally
ignores timestamp/commit fields and tolerates roundoff at 1e-12. It does not
require raw medical images, checkpoints, feature caches, GPUs, or Torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


IGNORE_KEYS = {"generated_at", "commit_sha", "results_root", "per_case_root"}
FLOAT_TOL = 1e-12
M14_TASKS = ("riga_cup", "riga_disc", "acdc_lv", "isic2018")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _compare(expected: Any, actual: Any, where: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        errors: list[str] = []
        if set(expected) != set(actual):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            if missing:
                errors.append(f"{where}: missing keys {missing}")
            if extra:
                errors.append(f"{where}: extra keys {extra}")
        for key in sorted(set(expected) & set(actual)):
            if key in IGNORE_KEYS:
                continue
            errors.extend(_compare(expected[key], actual[key], f"{where}.{key}"))
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: length {len(expected)} != {len(actual)}"]
        errors = []
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual, strict=True)):
            errors.extend(_compare(exp_item, act_item, f"{where}[{idx}]"))
        return errors
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        exp = float(expected)
        act = float(actual)
        if math.isnan(exp) and math.isnan(act):
            return []
        if abs(exp - act) <= FLOAT_TOL:
            return []
        return [f"{where}: {exp!r} != {act!r}"]
    if expected != actual:
        return [f"{where}: {expected!r} != {actual!r}"]
    return []


def _run(cmd: list[str], root: Path, tmp: Path) -> None:
    env = os.environ.copy()
    src_path = str(root / "code" / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing else os.pathsep.join([src_path, existing])
    )
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=root, env=env, check=True)


def _compare_file(root: Path, expected_rel: str, actual: Path) -> None:
    expected_path = root / expected_rel
    errors = _compare(_load_json(expected_path), _load_json(actual), expected_rel)
    if errors:
        joined = "\n".join(errors[:40])
        more = "" if len(errors) <= 40 else f"\n... {len(errors) - 40} more"
        raise AssertionError(f"{expected_rel} mismatch:\n{joined}{more}")
    print(f"OK {expected_rel}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Bundle root, default inferred from this script.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the scratch output directory for inspection.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    results_root = root / "results" / "_merged"
    if not results_root.is_dir():
        raise SystemExit(f"missing results root: {results_root}")

    with tempfile.TemporaryDirectory(prefix="fmpool_artifact_smoke_") as tmp_name:
        tmp = Path(tmp_name)
        _run(
            [
                sys.executable,
                "code/scripts/compute_all.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "compute_all"),
                "--verify",
            ],
            root,
            tmp,
        )
        compute_all_out = tmp / "compute_all"
        for rel in [
            "paper_table.json",
            "provenance_summary.json",
            "within_cross/kvasir.json",
            "within_cross/acdc_lv.json",
            "within_cross/brats_wt.json",
            "within_cross/riga_cup.json",
            "within_cross/riga_disc.json",
            "m_eff/kvasir.json",
            "m_eff/acdc_lv.json",
            "m_eff/brats_wt.json",
            "m_eff/riga_cup.json",
            "m_eff/riga_disc.json",
            "subject_level/acdc_lv.json",
            "subject_level/brats_wt.json",
            "calibration_split/kvasir.json",
            "calibration_split/acdc_lv.json",
            "calibration_split/brats_wt.json",
            "calibration_split/riga_cup.json",
            "calibration_split/riga_disc.json",
        ]:
            _compare_file(
                root,
                f"results/_merged/{rel}",
                compute_all_out / rel,
            )
        _run(
            [
                sys.executable,
                "code/scripts/compute_m4_table.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "m4_unet_summary.json"),
            ],
            root,
            tmp,
        )
        _compare_file(root, "results/_merged/m4_unet_summary.json", tmp / "m4_unet_summary.json")

        _run(
            [
                sys.executable,
                "code/scripts/compute_late_brats_summaries.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "late_brats_summaries.json"),
            ],
            root,
            tmp,
        )
        _compare_file(root, "results/_merged/late_brats_summaries.json", tmp / "late_brats_summaries.json")

        _run(
            [
                sys.executable,
                "code/scripts/compute_m14_summary.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "m14_clinical_summary.json"),
            ],
            root,
            tmp,
        )
        _compare_file(root, "results/_merged/m14_clinical_summary.json", tmp / "m14_clinical_summary.json")

        m14_task_outputs: dict[str, Path] = {}
        for task in M14_TASKS:
            out = tmp / f"m14_clinical_summary_{task}.json"
            _run(
                [
                    sys.executable,
                    "code/scripts/compute_m14_summary.py",
                    "--results-root",
                    str(results_root),
                    "--task",
                    task,
                    "--out",
                    str(out),
                ],
                root,
                tmp,
            )
            m14_task_outputs[task] = out
            _compare_file(
                root,
                f"results/_merged/m14_clinical_summary_{task}.json",
                out,
            )

        extension = {
            "schema_version": "m14_clinical_extension_summary_v1",
            "tasks": {task: _load_json(m14_task_outputs[task]) for task in M14_TASKS},
        }
        extension_out = tmp / "m14_clinical_extension_summary.json"
        extension_out.write_text(json.dumps(extension, indent=2, sort_keys=True) + "\n")
        _compare_file(
            root,
            "results/_merged/m14_clinical_extension_summary.json",
            extension_out,
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_cross_checkpoint_family.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "cross_checkpoint_family.json"),
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/cross_checkpoint_family.json",
            tmp / "cross_checkpoint_family.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_audit_sensitivity.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "audit_sensitivity.json"),
                "--n-perm",
                "1000",
                "--seed",
                "0",
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/audit_sensitivity.json",
            tmp / "audit_sensitivity.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_quality_controlled_pairs.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "quality_controlled_pairs.json"),
                "--n-boot",
                "1000",
                "--seed",
                "0",
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/quality_controlled_pairs.json",
            tmp / "quality_controlled_pairs.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_item_difficulty_robustness.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "item_difficulty_robustness.json"),
                "--n-boot",
                "2000",
                "--seed",
                "0",
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/item_difficulty_robustness.json",
            tmp / "item_difficulty_robustness.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_distshift.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "distshift_riga_messidor.json"),
                "--n-boot",
                "500",
                "--seed",
                "0",
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/distshift_riga_messidor.json",
            tmp / "distshift_riga_messidor.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_tau_sweep.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "tau_sweep.json"),
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/tau_sweep.json",
            tmp / "tau_sweep.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_threshold_table.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "threshold_table_riga_cup.json"),
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/threshold_table_riga_cup.json",
            tmp / "threshold_table_riga_cup.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_m15_cv_summary.py",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp / "m15_cv_summary.json"),
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/m15_cv_summary.json",
            tmp / "m15_cv_summary.json",
        )

        _run(
            [
                sys.executable,
                "code/scripts/compute_heldout_11fm_summary.py",
                "--per-case-root",
                "results/_merged/per_case_dice_heldout_11fm",
                "--out",
                str(tmp / "heldout_11fm_summary.json"),
            ],
            root,
            tmp,
        )
        _compare_file(
            root,
            "results/_merged/diagnostics/heldout_11fm_summary.json",
            tmp / "heldout_11fm_summary.json",
        )

        if args.keep_tmp:
            keep = root / "_artifact_smoke_tmp"
            if keep.exists():
                raise SystemExit(f"refusing to overwrite existing {keep}")
            tmp.rename(keep)
            print(f"kept scratch outputs at {keep}")

    print("FMPOOL_ARTIFACT_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
