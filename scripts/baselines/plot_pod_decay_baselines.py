"""POD(t) and POD(t)/POD(0h) across scales for persistence, climatology, and
random-null baselines -- companion to plot_pod_decay.py (which does this for
LatentSky only). Same two-panel layout, same colormap-by-scale convention, so
figures are directly comparable side by side.

Also produces a small-multiples overlay (one subplot per scale bin) with all
four series -- LatentSky, persistence, climatology, random -- on the same
axes, to check whether the baselines sit well below LatentSky's POD(t), as
opposed to just having the same qualitative decay shape.
"""
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

model_df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
model_df = model_df[model_df["predictor_set"] == "9var (mse+uv)"]
base_df  = pd.read_csv(REPO / "docs/scale_summary_baselines.csv")

SERIES = {
    "LatentSky":   model_df,
    "persistence": base_df[base_df["predictor_set"] == "persistence"],
    "climatology": base_df[base_df["predictor_set"] == "climatology"],
    "random":      base_df[base_df["predictor_set"] == "random"],
}
STYLE = {
    "LatentSky":   dict(color="#1f77b4", ls="-",  lw=2.0),
    "persistence": dict(color="#ff7f0e", ls="--", lw=1.6),
    "climatology": dict(color="0.2",     ls="-.", lw=1.6),
    "random":      dict(color="0.6",     ls=":",  lw=1.6),
}


def exp_decay(t, tau):
    return np.exp(-t / tau)


def fit_tau_per_scale(df):
    scales = sorted(df["scale_mid"].unique())
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    out = {}
    for scale in scales:
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(scale, np.nan)
        if not np.isfinite(p0) or p0 <= 0:
            continue
        norm = (sub["POD"] / p0).values
        try:
            popt, _ = curve_fit(exp_decay, t, norm, p0=[2.0], bounds=([0.1], [200.0]))
            out[scale] = popt[0]
        except Exception:
            out[scale] = np.nan
    return out


def make_decay_plot(df, name, out_path):
    """Two-panel POD(t) figure (raw + normalized/tau-fit), colored by scale."""
    scales = sorted(df["scale_mid"].unique())
    t_fine = np.linspace(0, df["lead_h"].max(), 200)
    cmap = cm.plasma
    n = len(scales)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for scale, color in zip(scales, colors):
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(scale, np.nan)
        ax_raw.plot(t, sub["POD"].values, color=color, lw=1.8, marker="o", ms=4)
        if not np.isfinite(p0) or p0 <= 0:
            continue
        norm = (sub["POD"] / p0).values
        ax_norm.plot(t, norm, color=color, lw=0, marker="o", ms=5)
        try:
            popt, _ = curve_fit(exp_decay, t, norm, p0=[2.0], bounds=([0.1], [200.0]))
            ax_norm.plot(t_fine, exp_decay(t_fine, popt[0]), color=color, lw=1.8,
                         alpha=0.85, label=f"{scale:.0f} km  τ={popt[0]:.1f}h")
        except Exception:
            pass

    ax_raw.set_xlabel("lead time [h]", fontsize=11)
    ax_raw.set_ylabel("POD", fontsize=11)
    ax_raw.set_title(f"(a) {name} — POD vs lead time", fontsize=11, loc="left", fontweight="bold")
    ax_raw.set_ylim(0, 1)
    ax_raw.grid(alpha=0.25)

    ax_norm.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.4)
    ax_norm.set_xlabel("lead time [h]", fontsize=11)
    ax_norm.set_ylabel("POD / POD(0h)", fontsize=11)
    ax_norm.set_title("(b) Normalised skill — fit: e^(−t/τ)", fontsize=11, loc="left", fontweight="bold")
    ax_norm.set_ylim(0, 1.05)
    ax_norm.grid(alpha=0.25)
    ax_norm.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    bounds = np.array(scales + [scales[-1] * 1.4])
    d_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    d_cmap = matplotlib.colors.ListedColormap(colors)
    sm = plt.cm.ScalarMappable(cmap=d_cmap, norm=matplotlib.colors.BoundaryNorm(np.arange(n + 1), n))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_raw, shrink=0.9, pad=0.02, ticks=np.arange(n) + 0.5)
    cb.set_ticklabels([f"{int(s)}" for s in scales])
    cb.set_label("system scale L [km]", fontsize=10)

    fig.suptitle(f"{name} — skill decay: POD(t)/POD(0h)", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_overlay_plot(out_path, n_panels=6):
    """Small multiples, one panel per scale bin, all 4 series overlaid."""
    all_scales = sorted(model_df["scale_mid"].unique())
    idx = np.linspace(0, len(all_scales) - 1, n_panels).round().astype(int)
    panel_scales = [all_scales[i] for i in sorted(set(idx))]

    ncols = 3
    nrows = int(np.ceil(len(panel_scales) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                              constrained_layout=True, squeeze=False)

    for ax, scale in zip(axes.flat, panel_scales):
        for name, df in SERIES.items():
            sub = df[np.isclose(df["scale_mid"], scale)].sort_values("lead_h")
            if sub.empty:
                continue
            ax.plot(sub["lead_h"], sub["POD"], marker="o", ms=3.5,
                    label=name, **STYLE[name])
        ax.set_title(f"L ≈ {scale:.0f} km", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(panel_scales):]:
        ax.axis("off")

    for ax in axes[-1]:
        ax.set_xlabel("lead time [h]", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("POD", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("POD(t) by scale: LatentSky vs. persistence / climatology / random null",
                 fontsize=11, y=1.1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    for name in ["persistence", "climatology", "random"]:
        make_decay_plot(SERIES[name], name,
                         FIGURES_DIR / f"pod_decay_{name}.png")

    make_overlay_plot(FIGURES_DIR / "pod_decay_overlay_all.png")

    print("\n=== τ (h) per scale, per series ===")
    taus = {name: fit_tau_per_scale(df) for name, df in SERIES.items()}
    all_scales = sorted(model_df["scale_mid"].unique())
    header = f"{'L (km)':>8}" + "".join(f"{name:>14}" for name in SERIES)
    print(header)
    for s in all_scales:
        row = f"{s:8.0f}"
        for name in SERIES:
            v = taus[name].get(s, np.nan)
            row += f"{v:14.2f}" if np.isfinite(v) else f"{'--':>14}"
        print(row)
