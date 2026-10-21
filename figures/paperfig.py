"""Shared style for the paper figures.

Rules (Sophie, 2026-09-08): modern look, no panel titles (captions carry the
message), consistent physical sizes (8.5 / 17.5 cm), fonts at caption size
(9 pt labels / 8 pt ticks from latentsky.mplstyle), colorbars horizontal,
thin, below the panel. PDFs are included in main.tex at (near-)natural size
so the printed font size is true.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("/glade/u/home/sabramian/LatentSky/docs/latentsky.mplstyle")

CM = 1 / 2.54
W1 = 8.5 * CM    # single column
W2 = 17.5 * CM   # double column

COLOR = {"two-head": "#2166ac", "flow": "#c55a11", "persistence": "#8a8a8a"}
MARKER = {"two-head": "o", "flow": "D", "persistence": "s"}


def panel_label(ax, s):
    """Bold (a)/(b) inside the panel, top-left."""
    ax.text(0.02, 0.98, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="top")


def thin_cbar(fig, mappable, ax, label=""):
    """Thin horizontal colorbar below the axis."""
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal",
                      fraction=0.05, pad=0.3, aspect=40)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    return cb
