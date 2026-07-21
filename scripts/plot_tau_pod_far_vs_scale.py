"""τ vs scale for both POD decay and FAR growth — LatentSky / DYAMOND."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = pathlib.Path(__file__).resolve().parents[1]

df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
df = df[df["predictor_set"] == "9var (mse+uv)"]

scales = sorted(df["scale_mid"].unique())
scale_arr = np.array(scales)
pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
far0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["FAR"]


def exp_decay(t, tau):
    return np.exp(-t / tau)


# --- fit POD: POD(t)/POD(0) = exp(-t/tau_pod) ---
tau_pod = []
for scale in scales:
    sub  = df[df["scale_mid"] == scale].sort_values("lead_h")
    t    = sub["lead_h"].values.astype(float)
    p0   = pod0.get(scale, 1.0)
    if p0 <= 0:
        tau_pod.append(np.nan); continue
    norm = (sub["POD"] / p0).values
    try:
        popt, _ = curve_fit(exp_decay, t, norm, p0=[2.0], bounds=([0.1], [200.0]))
        tau_pod.append(popt[0])
    except Exception:
        tau_pod.append(np.nan)

# --- fit FAR: (1-FAR(t))/(1-FAR(0)) = exp(-t/tau_far) ---
tau_far = []
for scale in scales:
    sub  = df[df["scale_mid"] == scale].sort_values("lead_h")
    t    = sub["lead_h"].values.astype(float)
    f0   = far0.get(scale, np.nan)
    if np.isnan(f0) or f0 >= 1.0:
        tau_far.append(np.nan); continue
    norm = (1 - sub["FAR"].values) / (1 - f0)
    try:
        popt, _ = curve_fit(exp_decay, t, norm, p0=[2.0], bounds=([0.1], [200.0]))
        tau_far.append(popt[0])
    except Exception:
        tau_far.append(np.nan)

tau_pod = np.array(tau_pod)
tau_far = np.array(tau_far)

# power-law fits
L_fit = np.geomspace(scale_arr[0], scale_arr[-1], 200)

def powerlaw(arr):
    valid = np.isfinite(arr)
    b, lna = np.polyfit(np.log(scale_arr[valid]), np.log(arr[valid]), 1)
    return np.exp(lna), b

a_pod, b_pod = powerlaw(tau_pod)
a_far, b_far = powerlaw(tau_far)

fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)

ax.scatter(scale_arr, tau_pod, color="#1f77b4", s=65, zorder=4)
ax.plot(L_fit, a_pod * L_fit ** b_pod, color="#1f77b4", lw=1.8, ls="--",
        label=f"POD decay:  τ = {a_pod:.3f} · L^{{{b_pod:.2f}}}")

ax.scatter(scale_arr, tau_far, color="#d62728", s=65, marker="s", zorder=4)
ax.plot(L_fit, a_far * L_fit ** b_far, color="#d62728", lw=1.8, ls="--",
        label=f"FAR growth:  τ = {a_far:.3f} · L^{{{b_far:.2f}}}")

ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(35, 6.3, "6 h lead", fontsize=8, color="0.4")

ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
ax.text(58, 0.25, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent predictability — LatentSky / DYAMOND\n"
             "August 2020, tropics ±40°", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_pod_far_vs_scale.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"POD: τ = {a_pod:.3f} · L^{b_pod:.3f}")
print(f"FAR: τ = {a_far:.3f} · L^{b_far:.3f}")
print(f"wrote {out}", flush=True)
