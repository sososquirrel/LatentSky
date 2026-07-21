"""Figure: tau(L) scaling law, 9var (mse+uv) only. Markers colored by
POD(0h) (initial detection skill). Dashed horizontal reference line at
tau = 6h.
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
CMAP = plt.cm.RdYlGn
NORM = mcolors.Normalize(vmin=0, vmax=1)


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

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    sca = ax.scatter(sc, tau, c=pod0, cmap=CMAP, norm=NORM, s=110,
                     edgecolors="k", linewidths=0.7, zorder=5)

    L_fit = np.geomspace(sc.min(), sc.max(), 200)
    ax.plot(L_fit, a * L_fit ** b, "k--", lw=1.6, alpha=0.8,
            label=f"τ = {a:.4f}·L^{b:.2f}")

    ax.axhline(6.0, color="0.4", lw=1.2, ls=":")
    ax.text(sc.min() * 1.05, 6.3, "τ = 6 h", fontsize=9, color="0.4")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("POD(0h) — initial detection skill", fontsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("system scale  L = √A  [km]", fontsize=12)
    ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=12)
    ax.set_title("Scale-dependent predictability law — LatentSky (9var mse+uv)", fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    out = OUT_DIR / "fig16_scaling_law_9var.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)
    print(f"power law: a={a:.4f} b={b:.3f} n_scales={len(sc)}", flush=True)


if __name__ == "__main__":
    main()
