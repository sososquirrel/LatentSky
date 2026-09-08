"""Paper Fig 7c (WB2 decomposition) — month-pooled clouds, 2 x 2.

One panel per model (GraphCast, FuXi, GenCast mean, GenCast member),
+-1.5 axes, barycenter crosses in the colorbar's viridis colors
(L = 65/200/700 km representatives of the 0-100/100-400/>400 km bands),
one shared thin colorbar below (same geometry as the decomposition figure).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from paperfig import W2, panel_label  # noqa: F401

WB = Path("/glade/u/home/sabramian/LatentSky/wb2_benchmark")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")

MODELS = ["graphcast", "fuxi", "gencast mean", "gencast member"]
BARY_BANDS = [(0, 100), (100, 400), (400, 1e9)]
NORM = LogNorm(30, 3000)
CMAP = plt.get_cmap("viridis")
BARY_C = [CMAP(NORM(L)) for L in (65, 200, 700)]
NMIN = 20


def main():
    df = pd.read_csv(WB / "matches_12h_month.csv.gz")
    fig, axs = plt.subplots(2, 2, figsize=(W2 * 0.72, W2 * 0.78),
                            sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.98, bottom=0.20,
                        wspace=0.08, hspace=0.14)
    sc = None
    for ax, model in zip(axs.ravel(), MODELS):
        m = df[df.model == model]
        sc = ax.scatter(m.dlnA, m.dlnrho, c=m.scale, cmap=CMAP, norm=NORM,
                        s=1.8, alpha=0.4, lw=0, rasterized=True)
        for (lo, hi), c in zip(BARY_BANDS, BARY_C):
            b = m[(m.scale >= lo) & (m.scale < hi)]
            if len(b) < NMIN:
                continue
            ax.scatter(np.average(b.dlnA, weights=b.P_t),
                       np.average(b.dlnrho, weights=b.P_t), marker="X",
                       s=80, color=c, edgecolors="white", linewidths=0.8,
                       zorder=5)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(0, color="0.6", lw=0.5)
        xs = np.array([-1.5, 1.5])
        ax.plot(xs, -xs, ":", color="#c0392b", lw=0.8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.text(0.97, 0.03, f"{model}\n{len(m):,} pairs",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.25")
    for ax in axs[1]:
        ax.set_xlabel(r"d$\ln A$  (area error)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"d$\ln\rho$  (intensity error)")

    cax = fig.add_axes([0.16, 0.075, 0.72, 0.018])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("system scale L = √A (km)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_alpha(1)
    fig.savefig(OUT / "fig7c_clouds.pdf", dpi=300)
    print("saved fig7c_clouds.pdf")


if __name__ == "__main__":
    main()
