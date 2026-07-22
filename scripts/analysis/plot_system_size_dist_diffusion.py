"""System-size distribution (truth vs diffusion prediction), all 6 leads
combined. X axis: characteristic scale L = sqrt(area) [km, log scale].
Y axis: probability density (KDE, normalized to integrate to 1).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from latentsky.registry import REPO_ROOT as REPO

L_MIN, L_MAX = 30, 2000
L_GRID = np.geomspace(L_MIN, L_MAX, 300)


def kde(values):
    v = values[(values >= L_MIN) & (values <= L_MAX)]
    if len(v) < 10:
        return np.zeros_like(L_GRID)
    lv = np.log(v)
    bw = v.std() / v.mean() * 0.4
    return gaussian_kde(lv, bw_method=bw)(np.log(L_GRID))


df = pd.read_csv(REPO / "docs/scale_systems_diffusion.csv.gz")
truth_L = df[df["role"].isin(["match", "miss"])]["L_t"].dropna().to_numpy()
pred_L = df[df["role"].isin(["match", "false_alarm"])]["L_p"].dropna().to_numpy()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(L_GRID, kde(truth_L), color="k", lw=2, label=f"truth (n={len(truth_L)})")
ax.plot(L_GRID, kde(pred_L), color="#d62728", lw=2, label=f"diffusion pred (n={len(pred_L)})")
ax.set_xscale("log")
ax.set_xlabel("system scale L = sqrt(area)  (km)")
ax.set_ylabel("probability density")
ax.set_title("System-size distribution -- diffusion (all 6 leads combined)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

out = REPO / "figures/main/final_12_diffusion/size_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
