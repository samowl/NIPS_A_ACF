"""Generate jobs.txt listing all (task, fm, seed) combinations per SPEC §7.

Ordering: round-robin across FMs so that each worker gets a mix of small/large FMs
(load balancing). Output format: one "task fm seed" per line.
"""
from pathlib import Path
import itertools

TASKS = ["kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc"]
FMS = [
    "dinov2_vitb14",
    "dinov2_vits14",
    "biomedclip",
    "clip_vitl14",
    "clip_vitb16",
    "mae_vitb16",
    "deit_vitb16",
    "convnext_tiny",
    "efficientnet_b0",
    "resnet50",
    "resnet18",
]
SEEDS = [42, 43, 44, 45]

def main():
    jobs = []
    # Ordering: (seed, fm, task) — each seed block has all tasks x all FMs
    for seed in SEEDS:
        for fm in FMS:
            for task in TASKS:
                jobs.append((task, fm, seed))
    out = Path("jobs.txt")
    out.write_text("\n".join(f"{t} {f} {s}" for t, f, s in jobs) + "\n")
    print(f"wrote {len(jobs)} jobs to {out}")

if __name__ == "__main__":
    main()
