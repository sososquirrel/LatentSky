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

from latentsky.training.patch_split import make_patch_split  # noqa: E402

OUT = REPO / "paper" / "figures"
COL = {"val": "#e08214", "test": "#2166ac"}


def main():
    train_b, val_b, test_b = make_patch_split(40, NR, NC, 1919810, 0.7, 0.15)
    pw = load("precipw")
    fig, ax = plt.subplots(figsize=(W2, W2 * H / W + 0.75))
    im = ax.imshow(pw, origin="lower", cmap="BrBG", vmin=-3, vmax=3,
                   interpolation="nearest", aspect="equal",
                   extent=[0, W, 0, H])
    add_coastlines(ax)
    grid_lines(ax)
    for r in range(NR):
        for c in range(NC):
            if (PIECE, r, c) in val_b:
                role = "val"
            elif (PIECE, r, c) in test_b:
                role = "test"
            else:
                continue
            ax.add_patch(Rectangle((c * P, r * P), P, P, fill=True,
                                   fc=COL[role], alpha=0.22, lw=0))
            ax.add_patch(Rectangle((c * P, r * P), P, P, fill=False,
                                   ec=COL[role], lw=1.6))
    r, c = BLOCK
    ax.add_patch(Rectangle((c * P, r * P), P, P, fill=False, ec="#b2182b",
                           lw=2.0))
    ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["0°", "90°E", "180°", "90°W", "0°"])
    ax.set_yticks([0, 160, 320])
    ax.set_yticklabels(["40°S", "0°", "40°N"])
    # small legend swatches, top right above the map
    for i, (name, colr) in enumerate([("val", COL["val"]),
                                      ("test", COL["test"]),
                                      ("example block", "#b2182b")]):
        x = 0.64 + 0.13 * i
        ax.add_patch(Rectangle((x, 1.03), 0.022, 0.05, fill=(name != "example block"),
                               fc=colr, ec=colr, alpha=0.9, lw=1.4,
                               transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.03, 1.055, name, transform=ax.transAxes, fontsize=7.5,
                va="center")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.14, aspect=60)
    cb.set_label("precipitable water (z-score)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    fig.savefig(OUT / "fig1a_domain.pdf", dpi=300)
    print("saved fig1a_domain.pdf")


if __name__ == "__main__":
    main()
