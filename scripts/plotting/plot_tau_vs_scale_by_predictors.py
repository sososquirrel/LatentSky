"""Does the choice of input variables change the scale-dependent
predictability law tau(L), or just shift its overall level?

Three predictor sets, all with a real lead=0h anchor, all fit the same
POD(0h)-normalized way (exp(-t/tau), B=0 forced) — directly comparable,
no free-amplitude extrapolation issues like the AI-model comparison.

  5var (mse)     : precipw, t2m, olrtoa, mse_500hPa, mse_850hPa
  5var (relhum)  : precipw, t2m, olrtoa, relhum_500hPa, relhum_850hPa
  9var (mse+uv)  : the above mse set + zonal/meridional wind @500/850hPa
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
CMAP = plt.cm.RdYlGn
NORM = mcolors.Normalize(vmin=0, vmax=1)


def fit_pod0_normalized(df):
    scales = sorted(df["scale_mid"].unique())
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    sc_out, tau_out, pod0_out = [], [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(s, np.nan)
        if np.isnan(p0) or p0 <= 0:
            continue
        norm_pod = (sub["POD"] / p0).values
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), t, norm_pod,
                                p0=[2.0], bounds=([0.05], [500.0]))
            sc_out.append(s)
            tau_out.append(popt[0])
            pod0_out.append(p0)
        except Exception:
            pass
    return np.array(sc_out), np.array(tau_out), np.array(pod0_out)


def powerlaw_fit(scales, taus):
    valid = np.isfinite(taus) & (taus > 0)
    if valid.sum() < 2:
        return None, None
    b, lna = np.polyfit(np.log(scales[valid]), np.log(taus[valid]), 1)
    return np.exp(lna), b


df_all = pd.read_csv(REPO / "docs/scale_summary.csv")

series = [
    ("5var (mse)",     "#ff7f0e", "s"),
    ("5var (relhum)",  "#9467bd", "D"),
    ("9var (mse+uv)",  "#1f77b4", "o"),
]

fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
results = {}
for label, color, marker in series:
    df = df_all[df_all["predictor_set"] == label]
    sc, tau, pod0 = fit_pod0_normalized(df)
    results[label] = (sc, tau, pod0)
    # Marker shape = series identity; marker fill = POD(0h) on a shared
    # colorbar across all three series (so absolute detection skill at the
    # anchor lead is directly comparable, not just the decay timescale).
    ax.scatter(sc, tau, c=pod0, cmap=CMAP, norm=NORM, marker=marker, s=90,
               edgecolors=color, linewidths=1.8, zorder=5, label=label)
    a, b = powerlaw_fit(sc, tau)
    if a:
        L_fit = np.geomspace(sc.min(), sc.max(), 200)
        ax.plot(L_fit, a * L_fit ** b, color=color, lw=1.5, ls="--", alpha=0.7,
                label=f"  τ = {a:.4f}·L^{b:.2f}")

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("POD(0h) — initial detection skill", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=11)
ax.set_title("Does the input variable set change the scaling law?\n"
             "Same fit method (POD(0h)-normalized) across all three — directly comparable\n"
             "marker shape = predictor set (colored edge)  |  marker fill = POD(0h)",
             fontsize=9.5)

handles, labels = ax.get_legend_handles_labels()
shape_h = [h for h, l in zip(handles, labels) if not l.startswith("  τ")]
shape_l = [l for l in labels if not l.startswith("  τ")]
ax.legend(shape_h, shape_l, frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_vs_scale_by_predictors.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== power-law fits (tau = a * L^b) by predictor set ===")
for label, _, _ in series:
    sc, tau, _ = results[label]
    a, b = powerlaw_fit(sc, tau)
    if a:
        print(f"{label:16s}  a={a:.4f}  b={b:.3f}  n_scales={len(sc)}")
