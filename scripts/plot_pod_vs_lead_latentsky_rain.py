"""POD(t) vs forecast lead, one curve per system scale — LatentSky
(mseuv + rain_log1p input sweep).

Same style as plot_pod_vs_lead_graphcast.py, reading the multi-lead
scale_summary_rain.csv (one file, not per-lead files like the AI models).
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from latentsky.registry import REPO_ROOT as REPO


def free_exp(t, A, tau):
    return A * np.exp(-t / tau)


df = pd.read_csv(REPO / "docs/scale_summary_rain.csv")
df = df[df["name"].str.startswith("mseuv_rain_lead")]
scales = sorted(df["scale_mid"].unique())
leads = sorted(df["lead_h"].unique())

CMAP = plt.cm.viridis
NORM = mcolors.LogNorm(vmin=min(scales), vmax=max(scales))

fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

t_fit = np.linspace(min(leads), max(leads), 100)
for s in scales:
    sub = df[df["scale_mid"] == s].sort_values("lead_h")
    t = sub["lead_h"].values.astype(float)
    pod = sub["POD"].values
    color = CMAP(NORM(s))
    ax.plot(t, pod, "o-", color=color, ms=6, lw=1.3,
            markeredgecolor="k", markeredgewidth=0.4, zorder=5)
    if (pod > 0).any():
        try:
            popt, _ = curve_fit(free_exp, t, pod, p0=[pod[0], 2.0],
                                bounds=([1e-4, 0.05], [1.0, 500.0]))
            ax.plot(t_fit, free_exp(t_fit, *popt), "--", color=color, lw=0.9, alpha=0.6)
        except Exception:
            pass

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("system scale  L = √area  [km]", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xlabel("forecast lead  t  [h]", fontsize=11)
ax.set_ylabel("POD", fontsize=11)
ax.set_xticks(leads)
ax.set_title("LatentSky (9var + rain_log1p input): POD vs lead, by system scale\n"
             "40-day held-out test pieces, threshold=0.1 mm  "
             "(dashed: free-amplitude exp fit)", fontsize=10)
ax.grid(alpha=0.25)

out = REPO / "docs/figures/pod_vs_lead_latentsky_rain.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)
