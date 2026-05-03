"""Emit M3 head-sweep job manifest for run_worker_m3.sh.

M3 reuses the PASS-A feature cache produced by run_worker_v2 / make_jobs_v2.
This script does NOT emit EXTRACT lines: it assumes the cache for
``riga_cup`` x {dinov2_vitb14, biomedclip} is already present.

Format
------
One job per line:

    M3 {head_design} {task} {fm} {seed}

Grid
----
- 4 head designs : linear_1x1, mlp_2layer, unet_lite, transformer_decoder
- 1 task         : riga_cup
- 2 FMs          : dinov2_vitb14, biomedclip
- 2 seeds        : 42, 43

Total: 4 x 1 x 2 x 2 = 16 lines (8 design-FM cells x 2 seeds).
"""
from __future__ import annotations

from pathlib import Path

HEAD_DESIGNS = ("linear_1x1", "mlp_2layer", "unet_lite", "transformer_decoder")
TASKS = ("riga_cup",)
FMS = ("dinov2_vitb14", "biomedclip")
SEEDS = (42, 43)


def main() -> None:
    lines: list[str] = []
    for head in HEAD_DESIGNS:
        for task in TASKS:
            for fm in FMS:
                for seed in SEEDS:
                    lines.append(f"M3 {head} {task} {fm} {seed}")
    out = Path("jobs_m3.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} M3 jobs to {out}")


if __name__ == "__main__":
    main()
