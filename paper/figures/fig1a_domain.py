"""Paper Fig 1a — the domain, the 22 blocks, and this day's split assignment.

Caption sentence: the tropical belt at 0.25 degrees is tiled into 22 blocks
per day; each (block, day) belongs to train, val, or test for all 96 frames
of its day. Val amber, test blue, train uncolored; red = the example block.
No title, thin z-score colorbar below. CPU seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from paperfig import W2

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "docs"))
from make_data_overview_assets import (  # noqa: E402
    BLOCK, FRAME, H, NC, NR, P, PIECE, W, add_coastlines, grid_lines, load)

OUT = REPO / "paper" / "figures"


def main():
    pw = load("precipw")
    fig, ax = plt.subplots(figsize=(W2, W2 * H / W + 0.75))
    im = ax.imshow(pw, origin="lower", cmap="BrBG", vmin=-3, vmax=3,
                   interpolation="nearest", aspect="equal",
                   extent=[0, W, 0, H])
    add_coastlines(ax)
    grid_lines(ax)
    r, c = BLOCK
    ax.add_patch(Rectangle((c * P, r * P), P, P, fill=False, ec="#b2182b",
                           lw=2.0))
    ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["0°", "90°E", "180°", "90°W", "0°"])
    ax.set_yticks([0, 160, 320])
    ax.set_yticklabels(["40°S", "0°", "40°N"])
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.14, aspect=60)
    cb.set_label("Precipitable Water [z-score]", fontsize=8,
                 fontstyle="italic")
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    fig.savefig(OUT / "fig1a_domain.pdf", dpi=300)
    print("saved fig1a_domain.pdf")


if __name__ == "__main__":
    main()
