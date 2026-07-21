"""FAR growth timescale τ vs system scale L — from exponential fits."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from latentsky.registry import REPO_ROOT as REPO

df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
df = df[df["predictor_set"] == "9var (mse+uv)"]

scales = sorted(df["scale_mid"].unique())
far0   = df[df["lead_h"] == 0.0].set_index("scale_mid")["FAR"]
far6   = df[df["lead_h"] == 6.0].set_index("scale_mid")["FAR"]


def decay(t, tau):
    """(1-FAR(t))/(1-FAR(0)) = exp(-t/tau) — assumes FAR→1 at infinite lead."""
    return np.exp(-t / tau)


taus = []
for scale in scales:
    sub = df[df["scale_mid"] == scale].sort_values("lead_h")
    t   = sub["lead_h"].values.astype(float)
    f0  = far0.get(scale, np.nan)
    if np.isnan(f0) or f0 >= 1.0:
        taus.append(np.nan); continue
    norm = (1 - sub["FAR"].values) / (1 - f0)
    try:
        popt, _ = curve_fit(decay, t, norm, p0=[2.0],
                            bounds=([0.1], [200.0]))
        taus.append(popt[0])
    except Exception:
        taus.append(np.nan)

scale_arr = np.array(scales)
tau_arr   = np.array(taus)
reliable  = np.isfinite(tau_arr)

# power-law fit on reliable points
lx = np.log(scale_arr[reliable])
ly = np.log(tau_arr[reliable])
b, lna = np.polyfit(lx, ly, 1)
a_fit  = np.exp(lna)
L_fit  = np.geomspace(scale_arr[0], scale_arr[-1], 200)
tau_fit = a_fit * L_fit ** b

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.scatter(scale_arr, tau_arr, color="#d62728", s=60, zorder=4,
           label="fitted τ  (FAR→1 assumed)")
ax.plot(L_fit, tau_fit, color="#1f77b4", lw=1.8, ls="--",
        label=f"power law:  τ = {a_fit:.2f} · L^{{{b:.2f}}}")

ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(35, 6.2, "6 h lead", fontsize=8, color="0.4")
ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, ax.get_ylim()[0] * 1.2 if ax.get_ylim()[0] > 0 else 0.3,
        "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("FAR growth timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent false-alarm growth — LatentSky / DYAMOND\n"
             "August 2020, tropics ±40°.  Model: (1−FAR(t))/(1−FAR(0)) = exp(−t/τ)", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_far_vs_scale.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")

print(f"power law: τ_FAR = {a_fit:.3f} · L^{b:.3f}")
print("\nScale (km)  |  tau_FAR (h)")
for s, t in zip(scales, taus):
    print(f"  {s:6.0f}    |  {t:8.2f}")
print(f"\nwrote {out}", flush=True)
