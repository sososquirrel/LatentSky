"""Figure: POD(t) vs lead for every scale bin (left: raw POD, right:
POD(t)/POD(0h) normalized), colored by scale on a single shared colorbar
below both panels, with dashed exponential fits overlaid. 9var (mse+uv).
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

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"


def main():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[df["predictor_set"] == "9var (mse+uv)"]
    scales = sorted(df["scale_mid"].unique())
    leads = sorted(df["lead_h"].unique())

    cmap = plt.cm.viridis
    norm = mcolors.LogNorm(vmin=min(scales), vmax=max(scales))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    t_fit = np.linspace(min(leads), max(leads), 100)

    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].to_numpy(dtype=float)
        pod = sub["POD"].to_numpy()
        p0 = pod[t == 0][0] if (t == 0).any() else pod[0]
        color = cmap(norm(s))

        axes[0].plot(t, pod, "o-", color=color, ms=6, lw=1.4,
                     markeredgecolor="k", markeredgewidth=0.3, zorder=5)
        axes[1].plot(t, pod / p0, "o-", color=color, ms=6, lw=1.4,
                     markeredgecolor="k", markeredgewidth=0.3, zorder=5)

        # Exponential fits (dashed): raw POD uses free amplitude; normalized
        # POD is forced through 1 at t=0 (single-parameter tau fit).
        try:
            popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), t, pod,
                                p0=[pod[0], 2.0], bounds=([1e-4, 0.05], [1.0, 500.0]))
            axes[0].plot(t_fit, popt[0] * np.exp(-t_fit / popt[1]),
                        "--", color=color, lw=0.9, alpha=0.6)
        except Exception:
            pass
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), t, pod / p0,
                                p0=[2.0], bounds=([0.05], [500.0]))
            axes[1].plot(t_fit, np.exp(-t_fit / popt[0]),
                        "--", color=color, lw=0.9, alpha=0.6)
        except Exception:
            pass

    axes[0].set_xlabel("forecast lead  t  [h]", fontsize=12)
    axes[0].set_ylabel("POD", fontsize=12)
    axes[0].set_title("Raw POD vs lead", fontsize=12)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("forecast lead  t  [h]", fontsize=12)
    axes[1].set_ylabel("POD(t) / POD(0h)", fontsize=12)
    axes[1].set_title("Normalized POD vs lead", fontsize=12)
    axes[1].axhline(1.0, color="0.5", lw=0.8, ls=":")
    axes[1].grid(alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.6,
                        pad=0.12, aspect=40)
    cbar.set_label("system scale  L = √A  (km)", fontsize=11)

    fig.suptitle("LatentSky (9var mse+uv): POD vs lead, by scale — dashed: exponential fit",
                fontsize=12)

    out = OUT_DIR / "fig15_pod_vs_lead_all_scales.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
