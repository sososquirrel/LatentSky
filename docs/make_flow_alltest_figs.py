"""Full-record (+3 h) versions of the R3-preview figures, from the alltest
systems tables (every test block of every day, patch split seed 1919810):
POD/FAR by scale for two-head / flow / persistence, and the band-barycenter
shift on the dlnA/dlnrho plane. CPU pandas, login-safe (~1 min).
Inputs: data/eval_method/systems_alltest{,_flow,_persistence}_lead3h.csv.gz."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from latentsky.metrics import default_scale_edges, summarize_by_scale

REPO = Path("/glade/u/home/sabramian/LatentSky")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = REPO / "docs" / "assets_model_reference"

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def main():
    tabs = {
        "two-head": pd.read_csv(EVAL / "systems_alltest_lead3h.csv.gz"),
        "flow": pd.read_csv(EVAL / "systems_alltest_flow_lead3h.csv.gz"),
        "persistence": pd.read_csv(EVAL / "systems_alltest_persistence_lead3h.csv.gz"),
    }
    for m, df in tabs.items():
        print(m, len(df), df.role.value_counts().to_dict())

    # --- POD and FAR by scale, three models, full record
    edges = default_scale_edges(25, 600, 10)
    colors = {"two-head": "#2166ac", "flow": "#b05e0a", "persistence": "0.45"}
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.5), sharex=True)
    for name in ("two-head", "flow", "persistence"):
        summ = summarize_by_scale(tabs[name], edges).by_scale
        axs[0].plot(summ["scale_mid"], summ["POD"], "o-", color=colors[name],
                    ms=4, label=name)
        axs[1].plot(summ["scale_mid"], summ["FAR"], "s--", color=colors[name],
                    ms=4, label=name)
        print(name, "POD:", np.round(summ["POD"].to_numpy(), 3))
        print(name, "FAR:", np.round(summ["FAR"].to_numpy(), 3))
    for ax, ttl in [(axs[0], "POD — probability of detection"),
                    (axs[1], "FAR — false-alarm ratio")]:
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_xlabel("system scale L = √A (km)")
        ax.set_title(ttl + " (+3 h, full test set)", fontsize=9)
    axs[0].set_ylabel("object-based rate")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / "flow_podfar_alltest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- barycenter shift, full record
    bands = [("small < 65 km", 0, 65, 40), ("65–200 km", 65, 200, 115),
             ("large > 200 km", 200, np.inf, 350)]
    vir = plt.get_cmap("viridis")
    cn = LogNorm(vmin=30, vmax=600)
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    lim = 1.6
    ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
    xs = np.linspace(-lim, lim, 2)
    ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.1)
    for name, lo, hi, lrep in bands:
        pts = {}
        for m in ("two-head", "flow"):
            g = tabs[m][(tabs[m].role == "match") & (tabs[m].scale >= lo)
                        & (tabs[m].scale < hi)]
            w = g["P_t"].to_numpy()
            pts[m] = (float((w * g.dlnA).sum() / w.sum()),
                      float((w * g.dlnrho).sum() / w.sum()), len(g))
        (x0, y0, n0), (x1, y1, n1) = pts["two-head"], pts["flow"]
        col = vir(cn(lrep))
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.plot(x0, y0, "o", color=col, ms=8, mec="0.2")
        ax.plot(x1, y1, "D", color=col, ms=8, mec="0.2")
        print(f"BARY {name}: two-head ({x0:+.2f},{y0:+.2f}) n={n0} -> "
              f"flow ({x1:+.2f},{y1:+.2f}) n={n1}")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="0.3", label="two-head barycenter"),
        Line2D([], [], marker="D", ls="none", color="0.3", label="flow barycenter"),
        Line2D([], [], color=vir(cn(40)), lw=2, label="small < 65 km"),
        Line2D([], [], color=vir(cn(115)), lw=2, label="65–200 km"),
        Line2D([], [], color=vir(cn(350)), lw=2, label="large > 200 km")],
        fontsize=7, frameon=False, loc="lower left")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("dlnA  (area error)")
    ax.set_ylabel(r"dln$\rho$  (intensity error)")
    ax.set_title("prediction 2 at full record: barycenters collapse\n"
                 "(+3 h, all test blocks × all days, precip-weighted)", fontsize=9)
    fig.savefig(OUT / "flow_decomp_shift_alltest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved flow_podfar_alltest.png, flow_decomp_shift_alltest.png")


if __name__ == "__main__":
    main()
