"""FAR vs lead time for each scale bin — exponential fit per scale."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import FIGURES_DIR

df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
df = df[df["predictor_set"] == "9var (mse+uv)"]

scales = sorted(df["scale_mid"].unique())
leads  = sorted(df["lead_h"].unique())
t_fine = np.linspace(0, 6, 200)

cmap   = cm.plasma
n      = len(scales)
colors = [cmap(i / (n - 1)) for i in range(n)]

far0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["FAR"]


def growth_model(t, A, tau):
    """Absolute FAR growth: FAR(t) - FAR(0) = A*(1 - exp(-t/tau))."""
    return A * (1 - np.exp(-t / tau))


fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
ax_raw, ax_growth = axes

taus, As = [], []

for scale, color in zip(scales, colors):
    sub  = df[df["scale_mid"] == scale].sort_values("lead_h")
    t    = sub["lead_h"].values.astype(float)
    f0   = far0.get(scale, 0.0)
    far  = sub["FAR"].values
    grow = far - f0   # absolute growth from 0h baseline

    try:
        popt, _ = curve_fit(growth_model, t, grow, p0=[0.3, 2.0],
                            bounds=([0.0, 0.1], [1.0, 50.0]))
        A_fit, tau_fit = popt
    except Exception:
        A_fit, tau_fit = np.nan, np.nan

    taus.append(tau_fit)
    As.append(A_fit)

    # left: raw FAR
    ax_raw.plot(t, far, color=color, lw=1.8, marker="o", ms=4)

    # right: absolute growth + fit
    ax_growth.plot(t, grow, color=color, lw=0, marker="o", ms=5)
    if np.isfinite(tau_fit):
        ax_growth.plot(t_fine, growth_model(t_fine, A_fit, tau_fit),
                       color=color, lw=1.8, alpha=0.85,
                       label=f"{scale:.0f} km  τ={tau_fit:.1f}h")

ax_raw.set_xlabel("lead time [h]", fontsize=11)
ax_raw.set_ylabel("FAR", fontsize=11)
ax_raw.set_title("(a) FAR vs lead time", fontsize=11, loc="left", fontweight="bold")
ax_raw.set_ylim(0, 1)
ax_raw.grid(alpha=0.25)
ax_raw.set_xticks([0, 1, 2, 3, 4, 5, 6])

ax_growth.set_xlabel("lead time [h]", fontsize=11)
ax_growth.set_ylabel("FAR(t) − FAR(0h)", fontsize=11)
ax_growth.set_title("(b) FAR growth from 0h  —  fit: A·(1−e^(−t/τ))",
                    fontsize=11, loc="left", fontweight="bold")
ax_growth.set_ylim(bottom=0)
ax_growth.grid(alpha=0.25)
ax_growth.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax_growth.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

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
    "LatentSky — FAR growth: FAR/FAR(0h) = 1 + C·(1−exp(−t/τ))\n"
    "τ = timescale of false-alarm growth   C = asymptotic amplification factor",
    fontsize=9,
)

out = FIGURES_DIR / "far_decay_by_scale.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")

print("\nScale (km)  |  tau (h)  |  A (max growth)")
for s, tau, a in zip(scales[:len(taus)], taus, As):
    print(f"  {s:6.0f}    |  {tau:5.2f}    |  {a:.3f}")
print(f"\nwrote {out}", flush=True)
