"""τ vs scale: LatentSky, persistence, and random null baseline.

Fits POD(t)/POD(0h) = (1-B)*exp(-t/τ) + B per series:
  - LatentSky:   B=0  (1-month training, no climatological bias)
  - persistence: B = clim_POD / pers_POD(0h)  (naturally converges to clim)
  - random:      B=0  (decorrelated prediction, no floor expected)

The scale where LatentSky's τ exceeds τ_random marks the transition from
chaos-dominated (model in the noise) to dynamics-dominated (model learns
something that propagates).
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

REPO = pathlib.Path(__file__).resolve().parents[1]

model_df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
model_df  = model_df[model_df["predictor_set"] == "9var (mse+uv)"]
base_df   = pd.read_csv(REPO / "docs/scale_summary_baselines.csv")

pers_df  = base_df[base_df["predictor_set"] == "persistence"]
clim_df  = base_df[base_df["predictor_set"] == "climatology"]
rand_df  = base_df[base_df["predictor_set"] == "random"]

CMAP = plt.cm.RdYlGn
NORM = mcolors.Normalize(vmin=0, vmax=1)

# climatology POD at lead=0 per scale — used as asymptote B for persistence
clim_pod0 = clim_df[clim_df["lead_h"] == 0].set_index("scale_mid")["POD"]


def exp_decay_b(t, tau, B):
    return (1 - B) * np.exp(-t / tau) + B


def fit_series(df, B_mode="clim"):
    """Fit τ per scale.

    B_mode:
      'clim' — B = clim_POD(0h) / series_POD(0h)  (persistence)
      'zero' — B = 0 forced  (LatentSky, random null)
    """
    scales = sorted(df["scale_mid"].unique())
    pod0   = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    sc_out, tau_out, pod_out, B_out = [], [], [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t   = sub["lead_h"].values.astype(float)
        p0  = pod0.get(s, np.nan)
        if np.isnan(p0) or p0 <= 0:
            continue
        if B_mode == "clim":
            B = min(clim_pod0.get(s, 0.0) / p0, 0.95)
        else:
            B = 0.0
        norm_pod = (sub["POD"] / p0).values
        try:
            popt, _ = curve_fit(
                lambda t, tau: exp_decay_b(t, tau, B),
                t, norm_pod, p0=[2.0], bounds=([0.05], [500.0])
            )
            sc_out.append(s)
            tau_out.append(popt[0])
            pod_out.append(p0)
            B_out.append(B)
        except Exception:
            pass
    return np.array(sc_out), np.array(tau_out), np.array(pod_out), np.array(B_out)


def powerlaw_fit(scales, taus):
    valid = np.isfinite(taus) & (taus > 0)
    if valid.sum() < 2:
        return None, None
    b, lna = np.polyfit(np.log(scales[valid]), np.log(taus[valid]), 1)
    return np.exp(lna), b


ls_sc,   tau_ls,   pod_ls,   B_ls   = fit_series(model_df, B_mode="zero")
pers_sc, tau_pers, pod_pers, B_pers = fit_series(pers_df,  B_mode="zero")
L_fit = np.geomspace(min(ls_sc.min(), pers_sc.min()),
                     max(ls_sc.max(), pers_sc.max()), 200)

# POD_random at lead=0h per scale — used as B for LatentSky fit and as floor reference
rand_pod0 = rand_df[rand_df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
rand_pod6 = rand_df[rand_df["lead_h"] == 6.0].set_index("scale_mid")["POD"]

# find scales where LatentSky at 6h lead is still above random null
ls_pod6 = model_df[model_df["lead_h"] == 6.0].set_index("scale_mid")["POD"]
ls_pod0_by_scale = model_df[model_df["lead_h"] == 0.0].set_index("scale_mid")["POD"]

fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

# shade the "below random null at 6h" region — scales where model lost all skill
chaotic_scales = [s for s in sorted(ls_pod6.index)
                  if ls_pod6.get(s, 0) < rand_pod6.get(s, 0)]
if chaotic_scales:
    L_chaos_max = max(chaotic_scales) * 1.4   # upper edge of chaotic zone
    ax.axvspan(L_fit[0] * 0.9, L_chaos_max, color="#ffcccc", alpha=0.35, zorder=0,
               label="chaotic zone: POD(6h) < random null")
    ax.axvline(L_chaos_max, color="#d62728", lw=1.2, ls=":", zorder=3)
    ax.text(L_chaos_max * 1.03, 0.09,
            f"L* ≈ {L_chaos_max:.0f} km", fontsize=8, color="#d62728", va="bottom")

# persistence
a_p, b_p = powerlaw_fit(pers_sc, tau_pers)
ax.scatter(pers_sc, tau_pers, c=pod_pers, cmap=CMAP, norm=NORM,
           marker="^", s=80, zorder=5,
           edgecolors="k", linewidths=0.5, label="persistence (B=0)")
if a_p:
    ax.plot(L_fit, a_p * L_fit ** b_p, color="#ff7f0e", lw=1.8, ls="--",
            label=f"  τ_pers = {a_p:.3f}·L^{b_p:.2f}")

# LatentSky — B=0 (1-month training, converges to 0)
a_l, b_l = powerlaw_fit(ls_sc, tau_ls)
ax.scatter(ls_sc, tau_ls, c=pod_ls, cmap=CMAP, norm=NORM,
           marker="o", s=80, zorder=6,
           edgecolors="k", linewidths=0.5, label="LatentSky (B=0)")
if a_l:
    ax.plot(L_fit, a_l * L_fit ** b_l, color="#1f77b4", lw=1.8, ls="--",
            label=f"  τ_LS = {a_l:.3f}·L^{b_l:.2f}")

# colorbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("POD(0h)  — initial detection skill", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# reference lines
ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(28, 6.35, "6 h lead", fontsize=8, color="0.4")
ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, 0.07, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent predictability  —  transition scale L*\n"
             "August 2020, tropics ±40°.  Red zone: POD(6h) < random null",
             fontsize=10)

handles, labels = ax.get_legend_handles_labels()
shape_h = [h for h, l in zip(handles, labels) if not l.startswith("  τ")]
shape_l = [l for l in labels if not l.startswith("  τ")]
ax.legend(shape_h, shape_l, frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_vs_scale_baselines.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== τ and B per scale ===")
print(f"{'Scale':>8}  {'τ_LS':>8}  {'B_LS':>6}  {'τ_pers':>8}  {'B_pers':>6}  {'POD_ls(6h)>rand':>16}")
for i, s in enumerate(ls_sc):
    pi = np.argmin(np.abs(pers_sc - s))
    above = "YES" if ls_pod6.get(s, 0) > rand_pod6.get(s, 0) else "no"
    print(f"{s:8.0f}  {tau_ls[i]:8.2f}  {B_ls[i]:6.3f}  "
          f"{tau_pers[pi]:8.2f}  {B_pers[pi]:6.3f}  {above:>16}")
