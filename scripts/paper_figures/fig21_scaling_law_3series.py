"""tau(L) scaling law -- LatentSky only.

GraphCast/FuXi were dropped from this figure (they used to be plotted here
as a second/third power-law series). Their public WB2-archived leads are
only 4 points, 6h apart, starting at 6h (no true zero-lead anchor), and
GraphCast's 6h point carries a measurable spin-up artifact (see
scripts/paper_figures/fig15_pod_vs_lead_graphcast.py) -- too sparse and too
contaminated to support a power-law fit with a meaningful exponent. Forcing
one anyway (as earlier versions of this script did) produced tau values
extrapolated 2-7x past the actual fit window, i.e. not a real measurement.

GraphCast/FuXi are instead shown descriptively (raw POD vs lead, no fit) in
fig15_pod_vs_lead_graphcast.png / fig15_pod_vs_lead_fuxi.png.

This is now the same core result as fig16_scaling_law_9var.py; kept as a
separate file since it's referenced as the comparison-slide figure.
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

CMAP = plt.cm.RdYlGn
NORM = mcolors.Normalize(vmin=0, vmax=1)

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"


def fit_pod0_normalized(df):
    scales = sorted(df["scale_mid"].unique())
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    sc_out, tau_out, pod0_out = [], [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].to_numpy(dtype=float)
        p0 = pod0.get(s, np.nan)
        if np.isnan(p0) or p0 <= 0:
            continue
        norm_pod = (sub["POD"] / p0).to_numpy()
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
    b, lna = np.polyfit(np.log(scales), np.log(taus), 1)
    return np.exp(lna), b


def main():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[df["predictor_set"] == "9var (mse+uv)"]
    sc, tau, pod0 = fit_pod0_normalized(df)
    a, b = powerlaw_fit(sc, tau)

    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    ax.scatter(sc, tau, c=pod0, cmap=CMAP, norm=NORM, s=130,
               edgecolors="#3A6EA5", linewidths=2.2, zorder=5)

    L_fit = np.geomspace(sc.min(), sc.max(), 200)
    ax.plot(L_fit, a * L_fit ** b, color="#3A6EA5", lw=2.2, ls="--", alpha=0.85,
            label=f"LatentSky  τ = {a:.4f}·L$^{{{b:.2f}}}$")

    ax.axhline(6.0, color="0.4", lw=1.0, ls=":", zorder=1)
    ax.text(sc.min() * 1.05, 6.35, "τ = 6 h", fontsize=9, color="0.4")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("POD(0h) — initial detection skill", fontsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("system scale  L = √area  [km]", fontsize=13)
    ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=13)
    ax.set_title("Scale-dependent predictability — LatentSky (9var mse+uv)", fontsize=12.5)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    out = OUT_DIR / "fig21_scaling_law_3series.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)
    print(f"power law: a={a:.4f} b={b:.3f} n_scales={len(sc)}", flush=True)


if __name__ == "__main__":
    main()
