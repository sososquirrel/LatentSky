"""Map decorations shared by the paper and documentation figures:
coastlines from the cached 0.25-degree land mask, and the 128-pixel block
grid with its excluded remainder strips."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from latentsky.layout import H, P, W

LANDMASK = Path(__file__).resolve().parents[3] / "data/eval_legacy/_landmask_320x1440_-40_40_50m.npy"


def add_coastlines(ax, color="0.1", lw=0.6, mask=None):
    """Draw coastlines on a (H, W) pixel-coordinate map axis."""
    if mask is None:
        mask = np.load(LANDMASK)
    ax.contour(np.asarray(mask, dtype=float), levels=[0.5], colors=color, linewidths=lw,
               extent=[0, W, 0, H], origin="lower")


def grid_lines(ax, h=H, w=W, p=P):
    """Draw the block grid and shade the remainder strips left out of the split."""
    nr, nc = h // p, w // p             # non-overlapping tiling, as compute_patch_grid
    for r in range(nr + 1):
        ax.axhline(r * p, color="0.15", lw=0.7)
    for c in range(nc + 1):
        ax.axvline(c * p, color="0.15", lw=0.7)
    ax.axhspan(nr * p, h, color="0.5", alpha=0.45, lw=0)
    ax.axvspan(nc * p, w, color="0.5", alpha=0.45, lw=0)
    return nr, nc
