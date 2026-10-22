"""Paper Fig 5 (WB2) — the zoom pair of the GraphCast +6 h matching above, and the
size distribution of the systems (ERA5 and the WB2 models, +12 h, August 2020) below;
the detection and false-alarm panels moved to the supplement (figS4_wb2_pod.py). The belt maps
go to the supplement (fig5_wb2maps.py -> figures_si/figS2_wb2_belts.pdf)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from fig3_decomp import PAPER_FONT
from fig5_wb2maps import compute, draw_zoom
from fig5_sizedist import plot_sizedist

REPO = Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "paper" / "figures"


def title(fig, ax, letter, text):
    """Letter and short title above the panel, at a fixed offset in inches."""
    x0, y1 = ax.get_position().x0, ax.get_position().y1
    fw, fh = fig.get_size_inches()
    fig.text(x0, y1 + 0.07 / fh, letter, fontsize=9, ha="left", va="bottom")
    fig.text(x0 + 0.27 / fw, y1 + 0.07 / fh, text, fontsize=8.5, ha="left", va="bottom")


def main():
    plt.rcParams.update(PAPER_FONT)
    d = compute()
    # four square panels of one size on a 2 x 2 grid, text width (inches)
    S = 2.55                                   # panel side
    ML, GX, MR = 0.72, 0.88, 0.12              # left margin, column gap, right margin
    MT, GY, MB = 0.30, 0.62, 0.80              # top margin, row gap (with the rule), bottom margin (with the legend)
    WF = ML + 2 * S + GX + MR
    FH = MT + 2 * S + GY + MB
    fig = plt.figure(figsize=(WF, FH))

    def box(col, row):
        return [(ML + col * (S + GX)) / WF, (MB + (1 - row) * (S + GY)) / FH, S / WF, S / FH]

    ax_z = [fig.add_axes(box(c, 0)) for c in (0, 1)]
    for k, ax in enumerate(ax_z):
        draw_zoom(ax, d, k)
    ax_n, ax_p = fig.add_axes(box(0, 1)), fig.add_axes(box(1, 1))
    plot_sizedist(ax_n, ax_p, titles=None, legend=False)
    # the legend below the bottom row, one line
    h, lab = ax_n.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               handlelength=1.8, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
    # a thin rule between the map row and the distribution row
    yr = (MB + S + 0.45) / FH                  # midway between the (c) title and the maps
    fig.add_artist(plt.Line2D([ML / WF, (WF - MR) / WF], [yr, yr], color="0.6", lw=0.6))
    for ax, letter, text in ((ax_z[0], "(a)", "Truth (ERA5), one block"),
                             (ax_z[1], "(b)", "GraphCast, +6 h, same block"),
                             (ax_n, "(c)", "Number of systems, +12 h"),
                             (ax_p, "(d)", "Mean intensity, +12 h")):
        title(fig, ax, letter, text)
    fig.savefig(OUT / "fig5_wb2.pdf", dpi=300)
    print("saved fig5_wb2.pdf")


if __name__ == "__main__":
    main()
