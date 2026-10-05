"""Paper Fig 4 — the error decomposition on matched pairs (+3 h, full record).

(a) two-head cloud in the (dlnA, dlnrho) plane   (b) flow cloud, same axes

Caption sentence: the deterministic model sits in the compensating quadrant
(too large, too weak, worst for small systems); the flow's barycenters
collapse toward the origin at every scale. One thin scale colorbar below the
pair. CPU ~1 min.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from paperfig import W2

EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")

LIM = 2.7      # the clouds end near 2.5; a wider frame only adds white
BANDS = [(0, 65, "small < 65 km"), (65, 200, "65–200 km"),
         (200, 1e9, "large > 200 km")]
# barycenter crosses colored by the colorbar's own scale (viridis, LogNorm):
# representative L per band = 40 / 115 / 450 km
_norm = LogNorm(28, 1500)
_cmap = __import__("matplotlib.pyplot", fromlist=["get_cmap"]).get_cmap("viridis")
BARY_C = [_cmap(_norm(L)) for L in (40, 115, 450)]


def cloud(ax, df, label, letter):
    m = df[df.role == "match"]
    sc = ax.scatter(m.dlnA, m.dlnrho, c=m.scale, cmap="viridis",
                    norm=LogNorm(28, 1500), s=2.2, alpha=0.25, lw=0,
                    rasterized=True)
    for (lo, hi, name), c in zip(BANDS, BARY_C):
        b = m[(m.scale >= lo) & (m.scale < hi)]
        w = b.P_t
        bx = np.average(b.dlnA, weights=w)
        by = np.average(b.dlnrho, weights=w)
        ax.scatter(bx, by, marker="X", s=130, color=c, edgecolors="white",
                   linewidths=1.0, zorder=5, label=name)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    xs = np.array([-LIM, LIM])
    ax.plot(xs, -xs, ":", color="#c0392b", lw=0.9)
    ax.text(1.95, -1.68, r"d$\ln P$ = 0", color="#c0392b", fontsize=7.5,
            rotation=-45, ha="center", va="center")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"too small  $\leftarrow$   area error  d$\ln A$   $\rightarrow$  too large")
    ax.set_aspect("equal")
    ax.set_xticks(range(-2, 3))
    ax.set_yticks(range(-2, 3))
    # panel header: letter, model, number of pairs
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold")
    ax.text(0.10, 0.968, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9)
    ax.text(0.10, 0.915, f"{len(m):,} matched pairs", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color="0.35")
    return sc


def main():
    # two square panels at the full double-column width; included at natural size
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.60), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.985, bottom=0.215,
                        wspace=0.06)
    df_th = pd.read_csv(EVAL / "systems_alltest_lead3h.csv.gz")
    df_fl = pd.read_csv(EVAL / "systems_alltest_flow_lead3h.csv.gz")
    sc = cloud(axs[0], df_th, "two-head U-Net", "A.")
    cloud(axs[1], df_fl, "flow, per draw", "B.")
    axs[0].set_ylabel(r"too weak  $\leftarrow$   intensity error  d$\ln\rho$   $\rightarrow$  too strong")
    axs[0].legend(loc="upper right", fontsize=7.5, title="rain-weighted barycenter",
                  title_fontsize=7.5, handletextpad=0.3, borderaxespad=0.5,
                  frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
    cax = fig.add_axes([0.26, 0.085, 0.54, 0.02])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("system scale L = √A (km)", fontsize=8.5)
    ticks = [30, 50, 100, 200, 500, 1000]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=7.5, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_alpha(1)
    fig.savefig(OUT / "fig4_decomp.pdf", dpi=300)
    print("saved fig4_decomp.pdf")


if __name__ == "__main__":
    main()
