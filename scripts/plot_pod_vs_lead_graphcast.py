"""POD(t) vs forecast lead, one curve per system scale — GraphCast.

Raw decay curves underlying the τ-vs-scale fit in
plot_tau_vs_scale_ai_models.py: POD at leads {6,12,18,24}h, colored by
scale so the scale-dependence of the decay is visible directly (not just
the fitted τ summary).
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
LEADS = [6, 12, 18, 24]


def load_multilead(model):
    frames = []
    for lead in LEADS:
        f = REPO / f"docs/scale_summary_{model}_2020-08_lead{lead}h_thr2p4.csv"
        df = pd.read_csv(f)
        df["lead_h"] = lead  # override buggy constant column with true forecast lead
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def free_exp(t, A, tau):
    return A * np.exp(-t / tau)


df = load_multilead("graphcast")
scales = sorted(df["scale_mid"].unique())

CMAP = plt.cm.viridis
NORM = mcolors.LogNorm(vmin=min(scales), vmax=max(scales))

fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

t_fit = np.linspace(6, 24, 100)
for s in scales:
    sub = df[df["scale_mid"] == s].sort_values("lead_h")
    t = sub["lead_h"].values.astype(float)
    pod = sub["POD"].values
    color = CMAP(NORM(s))
    ax.plot(t, pod, "o-", color=color, ms=6, lw=1.3,
            markeredgecolor="k", markeredgewidth=0.4, zorder=5)
    try:
        popt, _ = curve_fit(free_exp, t, pod, p0=[pod[0], 10.0],
                            bounds=([1e-4, 0.5], [1.0, 500.0]))
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
ax.set_xticks(LEADS)
ax.set_title("GraphCast: POD vs lead, by system scale\n"
             "August 2020, tropics ±40°, threshold=2.4 mm/6h  "
             "(dashed: free-amplitude exp fit)", fontsize=10)
ax.grid(alpha=0.25)

out = REPO / "docs/figures/pod_vs_lead_graphcast.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)
