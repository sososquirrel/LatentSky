"""Predictability timescale τ vs system scale L — from exponential POD decay fits."""
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
pod0   = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]


def decay(t, tau):
    """Pure exponential — B=0 forced: skill eventually goes to zero."""
    return np.exp(-t / tau)


taus = []
for scale in scales:
    sub = df[df["scale_mid"] == scale].sort_values("lead_h")
    t   = sub["lead_h"].values.astype(float)
    p0  = pod0.get(scale, 1.0)
    if p0 <= 0:
        taus.append(np.nan); continue
    norm = (sub["POD"] / p0).values
    try:
        popt, _ = curve_fit(decay, t, norm, p0=[2.0],
                            bounds=([0.1], [200.0]))
        taus.append(popt[0])
    except Exception:
        taus.append(np.nan)

scale_arr = np.array(scales)
tau_arr   = np.array(taus)

# all points are now reliable — B=0 means 1 parameter, always constrained
reliable = np.isfinite(tau_arr)

# power-law fit: tau = a * L^b
valid = reliable
lx = np.log(scale_arr[valid])
ly = np.log(tau_arr[valid])
b, lna = np.polyfit(lx, ly, 1)
a = np.exp(lna)

L_fit  = np.geomspace(scale_arr[0], scale_arr[-1], 200)
tau_fit = a * L_fit ** b

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.scatter(scale_arr, tau_arr, color="#1f77b4", s=60, zorder=4,
           label="fitted τ  (B=0 forced)")
ax.plot(L_fit, tau_fit, color="#d62728", lw=1.8, ls="--",
        label=f"power law:  τ = {a:.2f} · L^{{{b:.2f}}}")

# reference: 6h lead time
ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(35, 6.2, "6 h lead", fontsize=8, color="0.4")

# ERA5 Nyquist
ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, 0.35, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("predictability timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent predictability — LatentSky / DYAMOND\n"
             "August 2020, tropics ±40°.  Model: exp(−t/τ), B=0 forced.", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_vs_scale.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"power law: τ = {a:.3f} · L^{b:.3f}")
print(f"wrote {out}", flush=True)
