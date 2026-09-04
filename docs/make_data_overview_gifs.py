"""Animated companions to docs/make_data_overview_assets.py's domain and
split figures, for docs/model_reference.html:

- data_domain.gif: the same domain map (piece 8, precipw, block grid, red
  example block) cycled across 12 frames spanning the day, instead of the
  single static frame 64 -- shows storms actually forming/moving.
- data_split.gif: the same "one day's block assignment" panel, cycled
  across 5 different days (pieces) instead of just day 9 -- demonstrates
  the split pattern isn't cherry-picked to look clean on one day.

Reuses the exact rendering constants/helpers from make_data_overview_assets
(same colors, same patch grid, same coastline mask) so the animated and
static versions are visually consistent. Pure numpy/matplotlib + Pillow for
GIF assembly -- login-node safe.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

from latentsky.training.patch_split import compute_patch_grid, make_patch_split

REPO = Path("/glade/u/home/sabramian/LatentSky")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"

P = 128
PIECE, BLOCK = 8, (1, 4)
H, W = 320, 1440
NR, NC = compute_patch_grid(H, W, P)  # 2 x 11
COL = {"train": "#e9edf1", "val": "#e08214", "test": "#2166ac"}
DOMAIN_FRAMES = list(range(0, 96, 8))  # 12 frames, every 2h across the day
SPLIT_PIECES = [0, 8, 19, 29, 39]      # 5 example days, spread across the record

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def load(var, t, pid=PIECE):
    return np.asarray(np.load(PIECES / f"{var}_piece{pid:02d}.npy", mmap_mode="r")[t],
                      dtype=np.float32)


def add_coastlines(ax, color="0.1", lw=0.6):
    mask = np.load(REPO / "data/eval_legacy/_landmask_320x1440_-40_40_50m.npy")
    ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=lw,
               extent=[0, W, 0, H], origin="lower")


def grid_lines(ax):
    for r in range(NR + 1):
        ax.axhline(r * P, color="0.15", lw=0.7)
    for c in range(NC + 1):
        ax.axvline(c * P, color="0.15", lw=0.7)
    ax.axhspan(NR * P, H, color="0.5", alpha=0.45, lw=0)
    ax.axvspan(NC * P, W, color="0.5", alpha=0.45, lw=0)


def fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf).convert("RGB")


def make_domain_gif():
    frames = []
    for t in DOMAIN_FRAMES:
        pw = load("precipw", t)
        fig, ax = plt.subplots(figsize=(10.5, 2.9), dpi=100)
        im = ax.imshow(pw, origin="lower", cmap="BrBG", vmin=-3, vmax=3,
                       interpolation="nearest", aspect="equal", extent=[0, W, 0, H])
        add_coastlines(ax)
        grid_lines(ax)
        r, c = BLOCK
        ax.add_patch(Rectangle((c * P, r * P), P, P, fill=False, ec="#b2182b", lw=2.2))
        ax.set_xticks([0, 360, 720, 1080, 1440])
        ax.set_xticklabels(["0°", "90°E", "180°", "90°W", "0°"], fontsize=8)
        ax.set_yticks([0, 160, 320]); ax.set_yticklabels(["40°S", "0°", "40°N"], fontsize=8)
        ax.set_title(f"precipw (standardized), piece {PIECE}, frame {t} "
                     f"({t * 15 // 60}h{t * 15 % 60:02d}m into the day) — red = example block",
                     fontsize=9)
        cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
        cb.set_label("z-score", fontsize=8)
        frames.append(fig_to_image(fig))
        plt.close(fig)

    out = OUT / "data_domain.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=500, loop=0,
                   optimize=True)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")


def make_split_gif():
    train_b, val_b, test_b = make_patch_split(40, NR, NC, 1919810, 0.7, 0.15)

    def role(p, r, c):
        if (p, r, c) in test_b:
            return "test"
        if (p, r, c) in val_b:
            return "val"
        return "train"

    frames = []
    for pid in SPLIT_PIECES:
        pw = load("precipw", 64, pid)
        fig, ax = plt.subplots(figsize=(10.5, 3.2), dpi=100)
        ax.imshow(pw, origin="lower", cmap="Greys", vmin=-3, vmax=3,
                  interpolation="nearest", aspect="equal", extent=[0, W, 0, H], alpha=0.55)
        add_coastlines(ax, color="0.05", lw=0.7)
        for r in range(NR):
            for c in range(NC):
                ax.add_patch(Rectangle((c * P, r * P), P, P,
                                       fc=COL[role(pid, r, c)], ec="0.2", lw=0.6, alpha=0.55))
                ax.text(c * P + P / 2, r * P + P / 2, str(r * NC + c),
                        ha="center", va="center", fontsize=9, color="0.15", fontweight="bold")
        ax.axhspan(NR * P, H, color="0.5", alpha=0.5, lw=0)
        ax.axvspan(NC * P, W, color="0.5", alpha=0.5, lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"piece {pid} (day {pid + 1} of 40): block assignment — "
                     "same split, 5 different days (not cherry-picked)", fontsize=9)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(fc=COL["train"], ec="0.3", label="train"),
                           Patch(fc=COL["val"], label="val"),
                           Patch(fc=COL["test"], label="test")],
                  loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8)
        frames.append(fig_to_image(fig))
        plt.close(fig)

    out = OUT / "data_split.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000, loop=0,
                   optimize=True)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    make_domain_gif()
    make_split_gif()
