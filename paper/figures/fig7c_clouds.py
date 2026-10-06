"""Paper Fig 8 (WB2 decomposition) — month-pooled, 2 x 2, in the style of Fig. 5.

One panel per model (GraphCast, FuXi, GenCast mean, GenCast member), +-1.5 axes.
For each size band (< 100, 100-400, > 400 km): the region holding 75 % of the
band's matched pairs shaded by density, the 95 % outline of all pairs, and the
rain-weighted mean error joined to the origin (fig4_decomp.decomposition_panel).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fig4_decomp import PAPER_FONT, decomposition_panel, legend_handles
from paperfig import W2

WB = Path("/glade/u/home/sabramian/LatentSky/wb2_benchmark")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")

MODELS = [("graphcast", "GraphCast"), ("fuxi", "FuXi"),
          ("gencast mean", "GenCast mean"), ("gencast member", "GenCast member")]
_v = plt.get_cmap("viridis")
BANDS = [(0, 100, r"small, $<$ 100 km", _v(0.08)),
         (100, 400, "medium, 100 to 400 km", _v(0.52)),
         (400, 1e9, r"large, $>$ 400 km", _v(0.88))]
LIM = 1.5


def main():
    plt.rcParams.update(PAPER_FONT)
    df = pd.read_csv(WB / "matches_12h_month.csv.gz")
    WF = 0.84 * W2
    fig, axs = plt.subplots(2, 2, figsize=(WF, WF * 1.10), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.955, bottom=0.145, wspace=0.10,
                        hspace=0.30)
    for ax, letter, (key, name) in zip(axs.ravel(), ("A.", "B.", "C.", "D."), MODELS):
        # fewer pairs than at km scale: a slightly wider smoothing, and bands need 20 pairs
        decomposition_panel(ax, df[df.model == key], BANDS, LIM, letter, name, sigma=4.0,
                            n_min=20)
    for ax in axs[0]:
        ax.set_xlabel("")
        for t in ax.texts:                       # the direction hints belong to the bottom row only
            if "too small" in t.get_text() or "too large" in t.get_text():
                t.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"intensity error  $\mathrm{d}\ln\rho$", fontsize=8.5, labelpad=3)
    fig.legend(handles=legend_handles(BANDS), loc="lower center", ncol=3, fontsize=7,
               frameon=False, columnspacing=2.2, handlelength=1.6, handletextpad=0.6,
               bbox_to_anchor=(0.54, 0.0))
    fig.savefig(OUT / "fig7c_clouds.pdf")
    print("saved fig7c_clouds.pdf")


if __name__ == "__main__":
    main()
