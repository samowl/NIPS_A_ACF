#!/usr/bin/env python3
"""Emit PASS-A/PASS-B jobs for the RETFound/MedSAM encoder-only boundary probe.

This is not an intended-use promptable MedSAM or clinical RETFound benchmark.
It runs the same frozen encoder + 1x1 decoder protocol as the primary audit,
conditioned on local checkpoint availability.
"""
from __future__ import annotations

from pathlib import Path


TASKS = ["riga_cup", "riga_disc", "acdc_lv", "kvasir"]
FMS = ["retfound_vitl16", "medsam_vitb16"]
SEEDS = [42, 43, 44, 45]


def main() -> None:
    lines: list[str] = []
    for task in TASKS:
        for fm in FMS:
            lines.append(f"EXTRACT {task} {fm}")
    for seed in SEEDS:
        for task in TASKS:
            for fm in FMS:
                lines.append(f"HEAD {task} {fm} {seed}")
    out = Path(__file__).resolve().parents[1] / "jobs_medical_fm.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_extract = len(TASKS) * len(FMS)
    n_head = n_extract * len(SEEDS)
    print(f"wrote {len(lines)} jobs ({n_extract} EXTRACT + {n_head} HEAD) to {out}")


if __name__ == "__main__":
    main()
