"""Build C1 figure: 3-procedure comparison for item-difficulty residualisation.

CPU-only. Reads released JSON files; no GPU, no model recomputation.

Procedures compared:
  Raw   : per-case Dice Pearson without residualisation (the Table-1 gap).
  Naive : per-case Dice Pearson on residuals where the difficulty regressor
          uses the target trace itself (mechanically leaked, illustrative).
  Held-out family-split : two halves of the FM panel are split into A and B
          subsets; difficulty fitted on A, residualised B's same/cross gap.
          Available only on Cup and ACDC.
  Cross-fitted          : each trace's difficulty regressor is estimated using
          only traces from broad upstream families *other than* the target
          trace's own family. Available on all 5 primary task rows.

Source files (read-only):
  results/_merged/diagnostics/item_difficulty_robustness.json
    -> task_rows[task].cross_fitted_item_difficulty_floor_0.30.bootstrap
       .residualized_gap_95ci, .raw_gap_95ci
  results/_merged/diagnostics/r7_item_difficulty_holdout.json
    -> [cup|acdc].holdout_mean_family
  Naive null values from appendix prose (Cup 0.886, ACDC 0.772) are taken from
  appendix tab:diffleak as released illustrative leaked-null estimates.

Outputs: figures/fig_diffleak_procedures.{pdf,png}
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "results" / "_merged" / "diagnostics"
OUT = REPO / "figures" / "fig_diffleak_procedures"

with open(DIAG / "item_difficulty_robustness.json") as f:
    cf = json.load(f)
with open(DIAG / "r7_item_difficulty_holdout.json") as f:
    ho = json.load(f)

TASKS = ["kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc"]
LABELS = ["Kvasir", "ACDC LV", "BraTS fg", "RIGA Cup", "RIGA Disc"]

raw_ci, cf_ci = [], []
for t in TASKS:
    bs = cf["task_rows"][t]["cross_fitted_item_difficulty_floor_0.30"]["bootstrap"]
    raw_ci.append(bs["raw_gap_95ci"])
    cf_ci.append(bs["residualized_gap_95ci"])

raw_mid = [(lo + hi) / 2 for lo, hi in raw_ci]
raw_lo = [m - lo for m, (lo, _) in zip(raw_mid, raw_ci)]
raw_hi = [hi - m for m, (_, hi) in zip(raw_mid, raw_ci)]

cf_mid = [(lo + hi) / 2 for lo, hi in cf_ci]
cf_lo = [m - lo for m, (lo, _) in zip(cf_mid, cf_ci)]
cf_hi = [hi - m for m, (_, hi) in zip(cf_mid, cf_ci)]

ho_cup_family = ho["cup"]["holdout_mean_family"]
ho_acdc_family = ho["acdc"]["holdout_mean_family"]
ho_cup_strict = ho["cup"]["holdout_mean_strict"]
ho_acdc_strict = ho["acdc"]["holdout_mean_strict"]

NAIVE_CUP, NAIVE_ACDC = 0.886, 0.772

x = np.arange(len(TASKS))
fig, ax = plt.subplots(figsize=(9.6, 4.4))
W = 0.20

ax.bar(
    x - 1.5 * W, raw_mid, W,
    yerr=[raw_lo, raw_hi],
    color="#3a3a3a", capsize=2.5,
    label=r"Raw $\Delta$ (no residualisation)",
)

ax.bar(
    x - 0.5 * W, cf_mid, W,
    yerr=[cf_lo, cf_hi],
    color="#3a7ec0", capsize=2.5,
    label="Cross-fitted broad-family-leaveout",
)

ho_y = [None, ho_acdc_family, None, ho_cup_family, None]
for i, v in enumerate(ho_y):
    if v is not None:
        ax.bar(
            [x[i] + 0.5 * W], [v], W,
            color="#e58a3c",
            label="Held-out family-split (Cup, ACDC only)" if i == 1 else None,
        )

naive_y = [None, NAIVE_ACDC, None, NAIVE_CUP, None]
for i, v in enumerate(naive_y):
    if v is not None:
        ax.bar(
            [x[i] + 1.5 * W], [v], W,
            color="white", edgecolor="#c0392b", linewidth=1.4, hatch="//",
            label="Naive (leaked, illustrative)" if i == 1 else None,
        )

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel(r"$\Delta = \rho_{\mathrm{same/release}} - \rho_{\mathrm{cross}}$")
ax.set_title("Item-difficulty residualisation: three procedures on aligned per-case Dice")
ax.set_ylim(0, 1.05)
ax.axhline(0, color="gray", linewidth=0.4)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=1)

plt.tight_layout()
os.makedirs(REPO / "figures", exist_ok=True)
plt.savefig(f"{OUT}.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}.png", dpi=180, bbox_inches="tight")

print(f"saved {OUT}.pdf")
print(f"saved {OUT}.png")
print()
print("numeric summary:")
print(f"  raw_mid     : {[f'{v:.3f}' for v in raw_mid]}")
print(f"  crossfit_mid: {[f'{v:.3f}' for v in cf_mid]}")
print(f"  holdout-fam : Cup={ho_cup_family:.3f}, ACDC={ho_acdc_family:.3f}")
print(f"  holdout-str : Cup={ho_cup_strict:.3f}, ACDC={ho_acdc_strict:.3f}")
print(f"  naive (ill.): Cup={NAIVE_CUP}, ACDC={NAIVE_ACDC}")
