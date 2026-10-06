"""Paper Fig 6 (WB2) — the zoom pair of the GraphCast +6 h matching above, and the
detection and false-alarm ratios by size of the WB2 models at +12 h below. The belt maps
go to the supplement (fig7a_wb2maps.py -> figS_wb2_belts.pdf)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from fig4_decomp import PAPER_FONT
from fig7a_wb2maps import compute, draw_zoom
from fig7b_pod import plot_podfar, series
from matplotlib.patches import Patch
from paperfig import W2

OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")


def title(ax, letter, text):
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=9, fontfamily="cmr10",
            ha="left", va="bottom")
    ax.text(0.12, 1.03, text, transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom")


def main():
    plt.rcParams.update(PAPER_FONT)
    d = compute()
    fig = plt.figure(figsize=(W2, W2 * 0.78))
    # top: two squares the size of the Fig. 3 panels, centred
    zw, zh = 0.285, 0.285 * W2 / (W2 * 0.78)
    gap = 0.06
    x0 = 0.5 - zw - gap / 2
    ax_z = [fig.add_axes([x0 + k * (zw + gap), 0.60, zw, zh]) for k in (0, 1)]
    for k, ax in enumerate(ax_z):
        draw_zoom(ax, d, k)
    title(ax_z[0], "(a)", "Truth (ERA5), one 128-pixel block")
    title(ax_z[1], "(b)", "GraphCast, +6 h, same block")
    ax_z[1].text(0.0, -0.04, f"{d['n_match']} matched pairs over the belt, same colour in (a) and (b); "
                 f"black: missed truth systems in (a), false alarms in (b)",
                 transform=ax_z[0].transAxes, ha="left", va="top", fontsize=6.5, color="0.35")
    # bottom: detection and false alarms by size
    ax_pod = fig.add_axes([0.075, 0.095, 0.40, 0.40])
    ax_far = fig.add_axes([0.585, 0.095, 0.40, 0.40])
    plot_podfar(ax_pod, ax_far, series())
    # bottom-row letters: (c), (d)
    for ax, letter in ((ax_pod, "(c)"), (ax_far, "(d)")):
        for t in ax.texts:
            if t.get_text() in ("(a)", "(b)"):
                t.set_text(letter)
    fig.savefig(OUT / "fig6_wb2.pdf", dpi=300)
    print("saved fig6_wb2.pdf")


if __name__ == "__main__":
    main()
