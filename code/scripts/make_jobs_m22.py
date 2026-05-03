"""Emit M22 job manifest (MC-dropout DINOv2-B / riga_cup).

Format
------
One line per (task, fm, seed, dropout_rate) cell:

    HEAD_MC {task} {fm} {seed} {dropout_rate}

The matching feature cache (PASS-A) is reused from the v2 pipeline; this
matrix only varies PASS-B (dropout-rate x seed). ``run_worker_m22.sh`` skips
cells whose JSON output already exists (resume-safe).

Matrix: 1 FM x 1 task x 4 seeds x 3 dropout rates = 12 cells.
"""
from __future__ import annotations

from pathlib import Path

TASKS = ["riga_cup"]
FMS = ["dinov2_vitb14"]
SEEDS = [42, 43, 44, 45]
DROPOUT_RATES = [0.1, 0.2, 0.4]


def main() -> None:
    lines: list[str] = []
    for task in TASKS:
        for fm in FMS:
            for rate in DROPOUT_RATES:
                for seed in SEEDS:
                    lines.append(f"HEAD_MC {task} {fm} {seed} {rate}")
    out = Path("jobs_m22.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
