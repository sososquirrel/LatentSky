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
    dx = 0.085 if ax.get_position().width > 0.35 else 0.13    # the letter's width in axis units
    ax.text(dx, 1.03, text, transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom")


def main():
    plt.rcParams.update(PAPER_FONT)
    d = compute()
    # sizes borrowed from the other figures: the maps are Fig. 3 panels, the skill panels
    # are Fig. 4 panels; the figure is text width like both
    WF = W2
    map_cm = 0.285 * W2                       # Fig. 3 panel (inches)
    skill_w = 0.41 * W2                       # Fig. 4 panel width
    skill_h = 0.392 * W2                      # Fig. 4 panel height
    FH = map_cm + skill_h + 1.85              # + titles, note, axis labels (inches)
    fig = plt.figure(figsize=(WF, FH))
    # both rows span the same width: the maps sit at the two ends of the span, the skill
    # panels fill it with one gap between them
    S0, S1, GAP = 0.13, 0.95, 0.08
    zw, zh = map_cm / WF, map_cm / FH
    PW = (S1 - S0 - GAP) / 2
    XL, XR = S0, S1 - PW
    ax_z = [fig.add_axes([x, 0.965 - zh, zw, zh]) for x in (S0, S1 - zw)]
    for k, ax in enumerate(ax_z):
        draw_zoom(ax, d, k)
    title(ax_z[0], "(a)", "Truth (ERA5), one 128-pixel block")
    title(ax_z[1], "(b)", "GraphCast, +6 h, same block")
    fig.text(0.5 * (S0 + S1), 0.965 - zh - 0.02,
             f"{d['n_match']} matched pairs over the belt, same colour in (a) and (b); "
             f"black: missed truth systems in (a), false alarms in (b)",
             ha="center", va="top", fontsize=6.5, color="0.35")
    # bottom: detection and false alarms by size
    sh = skill_h / FH
    ax_pod = fig.add_axes([XL, 0.08, PW, sh])
    ax_far = fig.add_axes([XR, 0.08, PW, sh])
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
