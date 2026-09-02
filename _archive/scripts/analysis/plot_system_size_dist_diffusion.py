"""System-size distribution (truth vs diffusion prediction), one curve per
lead so the distribution's evolution over time is visible (not collapsed
into a single combined curve). X axis: characteristic scale L = sqrt(area)
[km, log scale]. Y axis: probability density (KDE, normalized to 1).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
leads = sorted(df["lead_h"].unique())
cmap = cm.plasma
colors = [cmap(i / (len(leads) - 1)) for i in range(len(leads))]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
ax_t, ax_p = axes

for lead, color in zip(leads, colors):
    sub = df[df["lead_h"] == lead]
    truth_L = sub[sub["role"].isin(["match", "miss"])]["L_t"].dropna().to_numpy()
    pred_L = sub[sub["role"].isin(["match", "false_alarm"])]["L_p"].dropna().to_numpy()
    ax_t.plot(L_GRID, kde(truth_L), color=color, lw=2, label=f"{lead:g}h  (n={len(truth_L)})")
    ax_p.plot(L_GRID, kde(pred_L), color=color, lw=2, label=f"{lead:g}h  (n={len(pred_L)})")

for ax, title in zip(axes, ["(a) truth", "(b) diffusion prediction"]):
    ax.set_xscale("log")
    ax.set_xlabel("system scale L = sqrt(area)  (km)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.3)
ax_t.set_ylabel("probability density")
ax_p.legend(title="lead time", fontsize=8, title_fontsize=8, frameon=False, loc="upper right")

fig.suptitle("System-size distribution vs lead time -- diffusion")

out = REPO / "figures/main/final_12_diffusion/size_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
