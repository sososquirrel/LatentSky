"""τ vs scale: LatentSky vs GraphCast and FuXi (multi-lead).

LatentSky: leads 0,1,2,3,6h — normalised to 0h, B=0 forced.
WB2 models: leads 6,12,18,24h — normalised to 6h, free asymptote B.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from latentsky.registry import REPO_ROOT as REPO
DOCS = REPO / "docs"

# --- load LatentSky ---
ls_df = pd.read_csv(DOCS / "scale_summary_v3.csv")
ls_df = ls_df[ls_df["predictor_set"] == "9var (mse+uv)"]
scales = sorted(ls_df["scale_mid"].unique())
scale_arr = np.array(scales)

# --- load WB2 multi-lead ---
def load_wb2(model):
    parts = []
    for lead in [6, 12, 18, 24]:
        f = DOCS / f"scale_summary_{model}_2020-08_lead{lead}h_thr2p4.csv"
        df = pd.read_csv(f)
        df["lead_h"] = float(lead)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

gc_df = load_wb2("graphcast")
try:
    fx_df = load_wb2("fuxi")
except FileNotFoundError:
    fx_df = None

# random null: ERA5 frames shifted ~15 days within August 2020
try:
    rand_df = load_wb2("random_era5")
except FileNotFoundError:
    rand_df = None


# ---------------------------------------------------------------
def exp_decay_b0(t, tau):
    """B=0: skill goes to zero (LatentSky — 1 month training)."""
    return np.exp(-t / tau)

def exp_decay_free(t_rel, tau, B):
    """Free asymptote B (WB2 — learned climatology from ERA5)."""
    return (1 - B) * np.exp(-t_rel / tau) + B


def fit_tau_ls(df):
    """Fit τ per scale from LatentSky leads 0..6h, B=0 forced."""
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    taus = []
    for scale in scales:
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(scale, np.nan)
        if np.isnan(p0) or p0 <= 0:
            taus.append(np.nan); continue
        norm = (sub["POD"] / p0).values
        try:
            popt, _ = curve_fit(exp_decay_b0, t, norm, p0=[2.0],
                                bounds=([0.05], [500.0]))
            taus.append(popt[0])
        except Exception:
            taus.append(np.nan)
    return np.array(taus)


def fit_tau_wb2(df, rand_pod6=None):
    """Fit τ per scale from WB2 leads 6..24h, normalised to 6h.

    If rand_pod6 is given (dict scale→POD_random at lead 6h), B is fixed to
    POD_random / POD_model(6h) — decay time above the noise floor.
    Otherwise B is fitted freely.
    """
    pod6 = df[df["lead_h"] == 6.0].set_index("scale_mid")["POD"]
    taus, Bs = [], []
    for scale in scales:
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        if sub.empty:
            taus.append(np.nan); Bs.append(np.nan); continue
        t = sub["lead_h"].values.astype(float)
        t_rel = t - 6.0          # time elapsed since first forecast
        p6 = pod6.get(scale, np.nan)
        if np.isnan(p6) or p6 <= 0:
            taus.append(np.nan); Bs.append(np.nan); continue
        norm = (sub["POD"] / p6).values
        if rand_pod6 is not None:
            B = min(rand_pod6.get(scale, 0.0) / p6, 0.95)
            try:
                popt, _ = curve_fit(
                    lambda t, tau: exp_decay_free(t, tau, B),
                    t_rel, norm, p0=[10.0], bounds=([0.5], [500.0]),
                )
                taus.append(popt[0]); Bs.append(B)
            except Exception:
                taus.append(np.nan); Bs.append(np.nan)
        else:
            try:
                popt, _ = curve_fit(
                    exp_decay_free, t_rel, norm,
                    p0=[10.0, 0.3],
                    bounds=([0.5, 0.0], [500.0, 1.0]),
                )
                taus.append(popt[0]); Bs.append(popt[1])
            except Exception:
                taus.append(np.nan); Bs.append(np.nan)
    return np.array(taus), np.array(Bs)


def powerlaw_fit(scale_arr, tau_arr):
    valid = np.isfinite(tau_arr) & (tau_arr > 0)
    if valid.sum() < 2:
        return None, None
    b, lna = np.polyfit(np.log(scale_arr[valid]), np.log(tau_arr[valid]), 1)
    return np.exp(lna), b


tau_ls = fit_tau_ls(ls_df)

# build per-scale POD_random at lead 6h as the noise-floor asymptote for WB2 fits
rand_pod6 = None
if rand_df is not None:
    rand_pod6 = rand_df[rand_df["lead_h"] == 6.0].set_index("scale_mid")["POD"].to_dict()

tau_gc, B_gc = fit_tau_wb2(gc_df, rand_pod6=rand_pod6)
if fx_df is not None:
    tau_fx, B_fx = fit_tau_wb2(fx_df, rand_pod6=rand_pod6)
else:
    tau_fx, B_fx = None, None

# print summary
print("=== τ summary (h) — B_gc fixed at POD_random/POD_gc(6h) ===")
print(f"{'Scale':>8}  {'LatentSky':>10}  {'GraphCast':>10}  {'B_gc':>6}")
for i, s in enumerate(scales):
    print(f"{s:8.0f}  {tau_ls[i]:10.2f}  {tau_gc[i]:10.2f}  {B_gc[i]:6.2f}")

# ---------------------------------------------------------------
L_fit = np.geomspace(scale_arr[0], scale_arr[-1], 200)

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

a_r, b_r = None, None

# GraphCast
a_gc, b_gc = powerlaw_fit(scale_arr, tau_gc)
ax.scatter(scale_arr, tau_gc, color="#d62728", s=65, marker="^", zorder=5,
           label="GraphCast (leads 6–24h, B=random null)")
if a_gc:
    ax.plot(L_fit, a_gc * L_fit ** b_gc, color="#d62728", lw=1.8, ls="--",
            label=f"  τ = {a_gc:.3f}·L^{b_gc:.2f}")
    # mark crossing GraphCast / random
    if a_r and abs(b_gc - b_r) > 1e-6:
        L_star = (a_r / a_gc) ** (1.0 / (b_gc - b_r))
        if L_fit[0] < L_star < L_fit[-1]:
            ax.axvline(L_star, color="#d62728", lw=1.0, ls=":", zorder=3)
            ax.text(L_star * 1.04, 0.25, f"L*≈{L_star:.0f} km",
                    fontsize=8, color="#d62728")

# LatentSky
a_l, b_l = powerlaw_fit(scale_arr, tau_ls)
ax.scatter(scale_arr, tau_ls, color="#1f77b4", s=70, zorder=6,
           label="LatentSky (leads 0–6h, B=0)")
if a_l:
    ax.plot(L_fit, a_l * L_fit ** b_l, color="#1f77b4", lw=1.8, ls="--",
            label=f"  τ = {a_l:.3f}·L^{b_l:.2f}")

# FuXi (skip if data unavailable or τ saturated at upper bound)
if tau_fx is not None and np.any(np.isfinite(tau_fx) & (tau_fx < 490)):
    a, b = powerlaw_fit(scale_arr, tau_fx)
    ax.scatter(scale_arr, tau_fx, color="#2ca02c", s=65, marker="s", zorder=5,
               label="FuXi (leads 6–24h, free B)")
    if a:
        ax.plot(L_fit, a * L_fit ** b, color="#2ca02c", lw=1.8, ls="--",
                label=f"  τ = {a:.3f}·L^{b:.2f}")

ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(30, 6.3, "6 h lead", fontsize=8, color="0.4")
ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, 0.25, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent predictability — LatentSky vs GraphCast\n"
             "August 2020, tropics ±40°.  GraphCast B fixed at ERA5 random null",
             fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_vs_scale_wb2_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out}", flush=True)
