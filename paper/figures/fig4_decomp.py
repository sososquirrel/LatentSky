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
from paperfig import W2, panel_label

EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")

LIM = 3.2
BANDS = [(0, 65, "small < 65 km"), (65, 200, "65–200 km"),
         (200, 1e9, "large > 200 km")]
# barycenter crosses colored by the colorbar's own scale (viridis, LogNorm):
# representative L per band = 40 / 115 / 450 km
_norm = LogNorm(28, 1500)
_cmap = __import__("matplotlib.pyplot", fromlist=["get_cmap"]).get_cmap("viridis")
BARY_C = [_cmap(_norm(L)) for L in (40, 115, 450)]


def cloud(ax, df, label):
    m = df[df.role == "match"]
    sc = ax.scatter(m.dlnA, m.dlnrho, c=m.scale, cmap="viridis",
                    norm=LogNorm(28, 1500), s=1.5, alpha=0.25, lw=0,
                    rasterized=True)
    for (lo, hi, name), c in zip(BANDS, BARY_C):
        b = m[(m.scale >= lo) & (m.scale < hi)]
        w = b.P_t
        bx = np.average(b.dlnA, weights=w)
        by = np.average(b.dlnrho, weights=w)
        ax.scatter(bx, by, marker="X", s=90, color=c, edgecolors="white",
                   linewidths=0.8, zorder=5, label=name)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    xs = np.array([-LIM, LIM])
    ax.plot(xs, -xs, ":", color="#c0392b", lw=0.9)
    ax.text(2.05, -1.75, "dlnP = 0", color="#c0392b", fontsize=7,
            rotation=-45, ha="center", va="center")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"d$\ln A$  (area error)")
    ax.set_aspect("equal")
    ax.text(0.98, 0.02, f"{label}\n{len(m):,} matched pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="0.25")
    return sc


def main():
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.52), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.97, bottom=0.26,
                        wspace=0.08)
    df_th = pd.read_csv(EVAL / "systems_alltest_lead3h.csv.gz")
    df_fl = pd.read_csv(EVAL / "systems_alltest_flow_lead3h.csv.gz")
    sc = cloud(axs[0], df_th, "two-head")
    cloud(axs[1], df_fl, "flow (per draw)")
    axs[0].set_ylabel(r"d$\ln\rho$  (intensity error)")
    axs[0].legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 0.93),
                  title="precip-weighted barycenter", title_fontsize=7)
    panel_label(axs[0], "A.")
    panel_label(axs[1], "B.")
    cax = fig.add_axes([0.14, 0.10, 0.72, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("system scale L = √A (km)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_alpha(1)
    fig.savefig(OUT / "fig4_decomp.pdf", dpi=300)
    print("saved fig4_decomp.pdf")


if __name__ == "__main__":
    main()
