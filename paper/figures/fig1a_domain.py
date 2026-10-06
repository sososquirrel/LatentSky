"""Paper Fig 1 — the domain and the 22 blocks (map exactly textwidth).

The map axes are exactly 17.5 cm wide; latitude tick labels hang to the LEFT
of the map (in the page margin — main.tex right-aligns the canvas to the
text block). Colorbar: thin, below, labeled with the normalization actually
used (PW in mm, dataset-wide standardization), not "z-score". CPU seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "docs"))
from make_data_overview_assets import (  # noqa: E402
    BLOCK, H, P, W, add_coastlines, grid_lines, load)

matplotlib.rcParams["savefig.bbox"] = None  # exact canvas, no auto-crop
OUT = REPO / "paper" / "figures"

CM = 1 / 2.54
MAPW = 17.5 * CM                 # map axes width = textwidth
LABW = 0.85 * CM                 # left strip for the latitude labels
MAPH = MAPW * H / W              # equal-aspect map height
CBH, GAP, BOT, TOP = 0.28 * CM, 0.62 * CM, 0.95 * CM, 0.12 * CM
FW = LABW + MAPW
FH = TOP + MAPH + GAP + CBH + BOT


def main():
    # the paper's text font
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})
    pw = load("precipw")
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([LABW / FW, (BOT + CBH + GAP) / FH,
                       MAPW / FW, MAPH / FH])
    im = ax.imshow(pw, origin="lower", cmap="BrBG", vmin=-3, vmax=3,
                   interpolation="nearest", aspect="auto",
                   extent=[0, W, 0, H])
    add_coastlines(ax)
    grid_lines(ax)
    r, c = BLOCK
    ax.add_patch(Rectangle((c * P, r * P), P, P, fill=False, ec="#b2182b",
                           lw=2.0))
    ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels([r"$0^{\circ}$", r"$90^{\circ}$E", r"$180^{\circ}$", r"$90^{\circ}$W",
                        r"$0^{\circ}$"])
    ax.set_yticks([0, 80, 160, 240, 320])
    ax.set_yticklabels([r"$40^{\circ}$S", r"$20^{\circ}$S", r"$0^{\circ}$", r"$20^{\circ}$N",
                        r"$40^{\circ}$N"])
    ax.tick_params(axis="both", labelsize=8, length=2.5, width=0.7)
    xlabs = ax.get_xticklabels()
    xlabs[0].set_ha("left")
    xlabs[-1].set_ha("right")

    cax = fig.add_axes([(LABW + 0.16 * MAPW) / FW, BOT / FH,
                        0.68 * MAPW / FW, CBH / FH])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"precipitable water, standardized: $(\mathrm{PW} - \mu)/\sigma$",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    fig.savefig(OUT / "fig1a_domain.pdf", dpi=300)
    print("saved fig1a_domain.pdf")


if __name__ == "__main__":
    main()
