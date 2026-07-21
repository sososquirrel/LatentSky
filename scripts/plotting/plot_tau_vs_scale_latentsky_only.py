"""τ vs scale: LatentSky only (no persistence/random baselines, no chaotic zone / L*).

Fits POD(t)/POD(0h) = exp(-t/τ), B=0 forced (1-month training run,
no climatological floor expected).
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
from latentsky.paths import FIGURES_DIR

model_df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
model_df = model_df[model_df["predictor_set"] == "9var (mse+uv)"]

CMAP = plt.cm.RdYlGn
NORM = mcolors.Normalize(vmin=0, vmax=1)


def exp_decay(t, tau):
    return np.exp(-t / tau)


def fit_series(df):
    scales = sorted(df["scale_mid"].unique())
    pod0   = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    sc_out, tau_out, pod_out = [], [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t   = sub["lead_h"].values.astype(float)
        p0  = pod0.get(s, np.nan)
        if np.isnan(p0) or p0 <= 0:
            continue
        norm_pod = (sub["POD"] / p0).values
        try:
            popt, _ = curve_fit(exp_decay, t, norm_pod, p0=[2.0],
                                bounds=([0.05], [500.0]))
            sc_out.append(s)
            tau_out.append(popt[0])
            pod_out.append(p0)
        except Exception:
            pass
    return np.array(sc_out), np.array(tau_out), np.array(pod_out)


def powerlaw_fit(scales, taus):
    valid = np.isfinite(taus) & (taus > 0)
    if valid.sum() < 2:
        return None, None
    b, lna = np.polyfit(np.log(scales[valid]), np.log(taus[valid]), 1)
    return np.exp(lna), b


ls_sc, tau_ls, pod_ls = fit_series(model_df)
L_fit = np.geomspace(ls_sc.min(), ls_sc.max(), 200)

fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

a_l, b_l = powerlaw_fit(ls_sc, tau_ls)
ax.scatter(ls_sc, tau_ls, c=pod_ls, cmap=CMAP, norm=NORM,
           marker="o", s=80, zorder=6,
           edgecolors="k", linewidths=0.5, label="LatentSky (B=0)")
if a_l:
    ax.plot(L_fit, a_l * L_fit ** b_l, color="#1f77b4", lw=1.8, ls="--",
            label=f"  τ_LS = {a_l:.3f}·L^{b_l:.2f}")

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("POD(0h)  — initial detection skill", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(28, 6.35, "6 h lead", fontsize=8, color="0.4")
ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, 0.07, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent predictability — LatentSky\n"
             "August 2020, tropics ±40°", fontsize=10)

handles, labels = ax.get_legend_handles_labels()
shape_h = [h for h, l in zip(handles, labels) if not l.startswith("  τ")]
shape_l = [l for l in labels if not l.startswith("  τ")]
ax.legend(shape_h, shape_l, frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, which="both")

out = FIGURES_DIR / "tau_vs_scale_latentsky_only.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)
