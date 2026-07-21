"""POD vs lead time for each scale bin — exponential decay fit per scale."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

REPO = pathlib.Path(__file__).resolve().parents[1]

df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
df = df[df["predictor_set"] == "9var (mse+uv)"]

scales = sorted(df["scale_mid"].unique())
leads  = sorted(df["lead_h"].unique())
t_fine = np.linspace(0, 6, 200)

cmap   = cm.plasma
n      = len(scales)
colors = [cmap(i / (n - 1)) for i in range(n)]

pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]


def model(t, tau):
    """Pure exponential — B=0 forced: skill eventually goes to zero."""
    return np.exp(-t / tau)


fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
ax_raw, ax_norm = axes

taus = []

for scale, color in zip(scales, colors):
    sub = df[df["scale_mid"] == scale].sort_values("lead_h")
    t   = sub["lead_h"].values.astype(float)
    p0  = pod0.get(scale, 1.0)
    if p0 <= 0:
        continue
    norm = (sub["POD"] / p0).values

    try:
        popt, _ = curve_fit(model, t, norm, p0=[2.0],
                            bounds=([0.1], [200.0]))
        tau_fit = popt[0]
    except Exception:
        tau_fit = np.nan

    taus.append(tau_fit)

    # left: raw POD
    ax_raw.plot(t, sub["POD"].values, color=color, lw=1.8, marker="o", ms=4)

    # right: normalised + fit
    ax_norm.plot(t, norm, color=color, lw=0, marker="o", ms=5)
    if np.isfinite(tau_fit):
        ax_norm.plot(t_fine, model(t_fine, tau_fit),
                     color=color, lw=1.8, alpha=0.85,
                     label=f"{scale:.0f} km  τ={tau_fit:.1f}h")

ax_raw.set_xlabel("lead time [h]", fontsize=11)
ax_raw.set_ylabel("POD", fontsize=11)
ax_raw.set_title("(a) POD vs lead time", fontsize=11, loc="left", fontweight="bold")
ax_raw.set_ylim(0, 1)
ax_raw.grid(alpha=0.25)
ax_raw.set_xticks([0, 1, 2, 3, 4, 5, 6])

ax_norm.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.4)
ax_norm.set_xlabel("lead time [h]", fontsize=11)
ax_norm.set_ylabel("POD / POD(0h)", fontsize=11)
ax_norm.set_title("(b) Normalised skill  —  fit: (1−B)·e^(−t/τ) + B",
                  fontsize=11, loc="left", fontweight="bold")
ax_norm.set_ylim(0, 1.05)
ax_norm.grid(alpha=0.25)
ax_norm.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax_norm.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

# discrete colorbar for left panel
bounds  = np.array(scales + [scales[-1] * 1.4])
d_norm  = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
d_cmap  = matplotlib.colors.ListedColormap(colors)
sm = plt.cm.ScalarMappable(cmap=d_cmap,
     norm=matplotlib.colors.BoundaryNorm(np.arange(n + 1), n))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax_raw, shrink=0.9, pad=0.02, ticks=np.arange(n) + 0.5)
cb.set_ticklabels([f"{int(s)}" for s in scales])
cb.set_label("system scale L [km]", fontsize=10)

fig.suptitle(
    "LatentSky — skill decay: POD/POD(0h) = exp(−t/τ)   [B=0 forced]\n"
    "Assumes skill goes to zero at long lead (no climatological floor learned from 1 month of training).",
    fontsize=9,
)

out = REPO / "docs/figures/pod_decay_by_scale.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")

print("\nScale (km)  |  tau (h)")
for s, t in zip(scales[:len(taus)], taus):
    print(f"  {s:6.0f}    |  {t:5.2f}")
print(f"\nwrote {out}", flush=True)
