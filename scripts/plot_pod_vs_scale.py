"""POD vs scale for LatentSky, persistence, and random null.

One line per lead time, three line styles (solid/dashed/dotted).
Random null shown as a shaded band across leads.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from latentsky.registry import REPO_ROOT as REPO

model_df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
model_df  = model_df[model_df["predictor_set"] == "9var (mse+uv)"]
base_df   = pd.read_csv(REPO / "docs/scale_summary_baselines.csv")
pers_df   = base_df[base_df["predictor_set"] == "persistence"]
clim_df   = base_df[base_df["predictor_set"] == "climatology"]
rand_df   = base_df[base_df["predictor_set"] == "random"]

LEADS = [0, 1, 3, 6]

# colormap: red (0h) → blue (6h)
CMAP  = plt.cm.rainbow
NORM  = mcolors.Normalize(vmin=0, vmax=6)

fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)

# --- random null: shaded band across all leads ---
rand_pod_all = []
for lead in LEADS:
    sub = rand_df[rand_df["lead_h"] == lead].sort_values("scale_mid")
    if not sub.empty:
        rand_pod_all.append(sub["POD"].values)
scales_rand = rand_df[rand_df["lead_h"] == LEADS[0]].sort_values("scale_mid")["scale_mid"].values
if rand_pod_all:
    rand_arr = np.array(rand_pod_all)
    ax.fill_between(scales_rand, rand_arr.min(axis=0), rand_arr.max(axis=0),
                    color="0.75", alpha=0.4, zorder=1, label="random null (band across leads)")
    ax.plot(scales_rand, rand_arr.mean(axis=0),
            color="0.5", lw=1.2, ls=":", zorder=2)

# --- climatology: single line (prediction is always August mean, lead-invariant) ---
clim_sub = clim_df[clim_df["lead_h"] == 0].sort_values("scale_mid")
ax.plot(clim_sub["scale_mid"], clim_sub["POD"],
        color="k", lw=1.5, ls="-.", zorder=3, label="climatology")

# --- LatentSky and persistence: one line per lead ---
for lead in LEADS:
    color = CMAP(NORM(lead))

    ls_sub   = model_df[model_df["lead_h"] == lead].sort_values("scale_mid")
    pers_sub = pers_df[pers_df["lead_h"] == lead].sort_values("scale_mid")

    label_ls   = f"{int(lead)}h" if lead != LEADS[-1] else f"{int(lead)}h lead"
    label_pers = None   # avoid legend clutter; distinguish by linestyle

    ax.plot(ls_sub["scale_mid"], ls_sub["POD"],
            color=color, lw=2.0, ls="-", zorder=4,
            label=f"LatentSky  {int(lead)}h" if lead == LEADS[0] else f"  {int(lead)}h")
    ax.plot(pers_sub["scale_mid"], pers_sub["POD"],
            color=color, lw=1.5, ls="--", zorder=3)

# dummy handles for linestyle legend
from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color="k", lw=2.0, ls="-",   label="LatentSky"),
    Line2D([0], [0], color="k", lw=1.5, ls="--",  label="persistence"),
    Line2D([0], [0], color="k", lw=1.5, ls="-.",  label="climatology"),
    Line2D([0], [0], color="0.5", lw=1.2, ls=":", label="random null"),
]

# lead-time colorbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, ticks=LEADS,
                    boundaries=np.linspace(0, 6, 256))
cbar.set_label("lead time [h]", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# L* reference (chaotic zone boundary from random null)
ax.axvline(146, color="#d62728", lw=1.2, ls=":", zorder=5)
ax.text(150, 0.02, "L* ≈ 146 km", fontsize=8, color="#d62728", va="bottom")

# intersection of LatentSky 3h and climatology — scale where model beats clim at 3h
ls3   = model_df[model_df["lead_h"] == 3.0].sort_values("scale_mid")
clim0 = clim_df[clim_df["lead_h"] == 0.0].sort_values("scale_mid")
common_scales = np.intersect1d(ls3["scale_mid"].values, clim0["scale_mid"].values)
ls3_pod   = ls3.set_index("scale_mid").loc[common_scales, "POD"].values
clim_pod  = clim0.set_index("scale_mid").loc[common_scales, "POD"].values
diff = ls3_pod - clim_pod
# find first crossing (diff changes sign from negative to positive)
cross_idx = np.where(np.diff(np.sign(diff)) > 0)[0]
if len(cross_idx) > 0:
    i = cross_idx[0]
    # linear interpolation between the two bracketing scales
    s0, s1 = common_scales[i], common_scales[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    L_clim_cross = s0 + (s1 - s0) * (-d0) / (d1 - d0)
    color_3h = CMAP(NORM(3))
    ax.axvline(L_clim_cross, color=color_3h, lw=1.4, ls="--", zorder=5)
    ax.text(L_clim_cross * 1.04, 0.55,
            f"L_clim ≈ {L_clim_cross:.0f} km\n(LatentSky 3h > clim)",
            fontsize=8, color=color_3h, va="center")

ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, 0.92, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55", va="top")

ax.set_xscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("POD", fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("POD vs scale — LatentSky vs persistence vs random null\n"
             "August 2020, tropics ±40°.  Color = lead time",
             fontsize=10)
ax.legend(handles=style_handles, frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/pod_vs_scale_baselines.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)
