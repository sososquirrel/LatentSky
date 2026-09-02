"""POD(t) per scale bin, side by side: persistence vs two-head vs
two-head+diffusion.

Row 1: raw POD(t) -- the actual detection rate at each sampled lead, one
line per scale bin, colored by scale.
Row 2: the same data normalized by each scale bin's own POD at its
earliest available lead (POD(t)/POD(anchor); anchor=0h for persistence,
0.25h for two-head/diffusion, matching fit_pod0_normalized in
scripts/wb2/plot_tau_vs_scale_ai_models.py exactly) -- this is the curve
the tau(L) fits are actually applied to, so row 2 is what makes those tau
numbers checkable by eye.
Row 3: ONLY the fitted exponential exp(-t/tau) for each scale bin (no data
points) -- what the fit actually claims the decay looks like, with tau
labeled at each curve's endpoint. Dashed where tau exceeds the sampled
lead span (extrapolated, not interpolated -- same convention used
elsewhere in this project for flagging unreliable fits).
Row 4: the 2-parameter alternative B + (1-B)*exp(-t/tau) -- allows a
nonzero floor B (skill that never fully vanishes) instead of forcing decay
all the way to 0. Added because row 3's B=0-forced fit was shown to
undershoot the real 6h data point by ~0.10 for small-scale persistence
bins (tau small -> 6h is many e-folding times out -> forced-zero fit
crashes below the real, non-vanishing residual skill).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from latentsky.paths import FIGURES_DIR
from latentsky.registry import REPO_ROOT as REPO

pr = pd.read_csv(REPO / "docs/scale_summary_baselines.csv")
pr = pr[pr["predictor_set"] == "persistence"]

th = pd.read_csv(REPO / "docs/scale_summary_modelcompare_rebinned16.csv")
th_runs = [f"patchsplit_twohead_olr_fss_lead{L}" for L in
           ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]]
th = th[th["name"].isin(th_runs)]

df_ = pd.read_csv(REPO / "docs/scale_summary_diffusion_rebinned16.csv")
df_runs = [f"diffusion_lead{L}" for L in ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]]
df_ = df_[df_["name"].isin(df_runs)]

datasets = [(pr, "persistence", 0.0), (th, "two-head", 0.25), (df_, "two-head+diffusion", 0.25)]

all_scales = sorted(set(pr["scale_mid"]) | set(th["scale_mid"]) | set(df_["scale_mid"]))
cmap = plt.get_cmap("viridis")
cnorm = LogNorm(vmin=min(all_scales), vmax=max(all_scales))

fig, axes = plt.subplots(4, 3, figsize=(18.5, 20.5), sharex="col")

for col, (df, title, anchor) in enumerate(datasets):
    pod_anchor = df[df["lead_h"] == anchor].set_index("scale_mid")["POD"]
    for s in sorted(df["scale_mid"].unique()):
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        color = cmap(cnorm(s))
        axes[0, col].plot(sub["lead_h"], sub["POD"], "o-", color=color, lw=1.8, ms=6,
                          markeredgecolor="k", markeredgewidth=0.4, zorder=3)

        p0 = pod_anchor.get(s, np.nan)
        if not np.isfinite(p0) or p0 <= 0:
            continue
        t = sub["lead_h"].values.astype(float)
        norm_pod = (sub["POD"] / p0).values
        axes[1, col].plot(t, norm_pod, "o-", color=color, lw=1.8, ms=6,
                          markeredgecolor="k", markeredgewidth=0.4, zorder=3)

        # row 3: fit exp(-(t-anchor)/tau) to the normalized curve, plot the
        # fit only (no data points), dashed if tau extrapolates past the
        # sampled lead span, labeled with tau at the curve's right end.
        t_rel = t - anchor
        try:
            popt, _ = curve_fit(lambda tt, tau: np.exp(-tt / tau), t_rel, norm_pod,
                                p0=[2.0], bounds=([0.05], [500.0]))
            tau = popt[0]
        except Exception:
            continue
        span = t_rel.max()
        extrap = tau > span
        t_fit = np.linspace(0, span, 100)
        y_fit = np.exp(-t_fit / tau)
        axes[2, col].plot(t_fit + anchor, y_fit, ls="--" if extrap else "-",
                          color=color, lw=1.8, alpha=0.55 if extrap else 0.9, zorder=3)
        if not extrap:
            # vertical dropline at t=tau (the e-folding time), up to the
            # curve itself -- easier to read at a glance than a text label,
            # especially with many overlapping scale bins per panel.
            axes[2, col].plot([tau + anchor, tau + anchor], [0, np.exp(-1)],
                              ":", color=color, lw=1.3, alpha=0.8, zorder=2)

        # row 4: 2-parameter fit with a free floor B, instead of forcing B=0
        try:
            popt2, _ = curve_fit(
                lambda tt, B, tau2: B + (1 - B) * np.exp(-tt / tau2),
                t_rel, norm_pod, p0=[0.1, 2.0],
                bounds=([0.0, 0.05], [0.95, 500.0]))
            B, tau2 = popt2
        except Exception:
            continue
        y_fit2 = B + (1 - B) * np.exp(-t_fit / tau2)
        axes[3, col].plot(t_fit + anchor, y_fit2, "-", color=color, lw=1.8,
                          alpha=0.9, zorder=3)
        if tau2 <= span:
            # same dropline convention, but to B + (1-B)/e -- the
            # e-folding point of the DECAYING part of this curve, not the
            # floor itself.
            y_at_tau2 = B + (1 - B) * np.exp(-1)
            axes[3, col].plot([tau2 + anchor, tau2 + anchor], [0, y_at_tau2],
                              ":", color=color, lw=1.3, alpha=0.8, zorder=2)

    axes[0, col].set_title(title, fontsize=12)
    axes[0, col].set_ylim(0, 1.05)
    axes[1, col].set_ylim(0, 1.05)
    axes[2, col].set_ylim(0, 1.05)
    axes[3, col].set_ylim(0, 1.05)
    axes[3, col].set_xlabel("lead (h)")
    axes[1, col].axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
    for row in (0, 1, 2, 3):
        axes[row, col].grid(alpha=0.25)

axes[0, 0].set_ylabel("POD")
axes[1, 0].set_ylabel("POD(t) / POD(anchor)\n(anchor: persistence=0h, two-head/diffusion=0.25h)")
axes[2, 0].set_ylabel("fitted exp(-t/tau), B=0 forced\n(dashed = tau extrapolates past sampled lead span)")
axes[3, 0].set_ylabel("fitted B + (1-B)*exp(-t/tau)\n(free floor B, no extrapolation flag needed)")

sm = ScalarMappable(norm=cnorm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, pad=0.02)
cbar.set_label("system scale  L = √A  (km)")

fig.suptitle("POD(t) -- persistence vs two-head vs two-head+diffusion, one line per scale bin\n"
             "top: raw POD   bottom: normalized by POD(anchor) -- the curve the tau(L) fits are applied to")

out = FIGURES_DIR / "pod_vs_lead_persistence_vs_twohead_vs_diffusion.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
