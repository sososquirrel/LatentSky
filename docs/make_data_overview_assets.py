"""Data-overview figures for docs/model_reference.html: the domain with its
patch tiling, one sample of all 10 input channels + target, and the
train/val/test patch split. Pure numpy/matplotlib — login-node safe."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle

from latentsky.training.patch_split import compute_patch_grid, make_patch_split

REPO = Path("/glade/u/home/sabramian/LatentSky")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"

INPUT_VARS = [
    "precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
    "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
    "umeridional_850hPa", "rain_log1p",
]
P = 128
PIECE, FRAME = 8, 64          # same case as the model example
BLOCK = (1, 4)
DELTA = 13                    # 3 h, matching the example
H, W = 320, 1440
NR, NC = compute_patch_grid(H, W, P)  # 2 x 11
COL = {"train": "#e9edf1", "val": "#e08214", "test": "#2166ac"}

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def load(var, t=FRAME, pid=PIECE):
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
    # unused remainder strips
    ax.axhspan(NR * P, H, color="0.5", alpha=0.45, lw=0)
    ax.axvspan(NC * P, W, color="0.5", alpha=0.45, lw=0)


def fig_domain():
    pw = load("precipw")
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    im = ax.imshow(pw, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest", aspect="equal",
                   extent=[0, W, 0, H])
    add_coastlines(ax)
    grid_lines(ax)
    r, c = BLOCK
    ax.add_patch(Rectangle((c * P, r * P), P, P, fill=False, ec="#b2182b", lw=2.2))
    ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["0°", "90°E", "180°", "90°W", "0°"], fontsize=8)
    ax.set_yticks([0, 160, 320]); ax.set_yticklabels(["40°S", "0°", "40°N"], fontsize=8)
    ax.set_title(f"precipw (standardized), piece {PIECE}, frame {FRAME} — "
                 f"{NR}×{NC} = {NR*NC} blocks of {P}²; gray strips unused; "
                 "red = the example block", fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
    cb.set_label("z-score", fontsize=8)
    fig.savefig(OUT / "data_domain.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_sample_channels():
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    cmaps = {
        "precipw": "BrBG", "mse_500hPa": "BrBG", "mse_850hPa": "BrBG",
        "t2m": "RdYlBu_r", "olrtoa": "Greys",
        "uzonal_500hPa": "RdBu_r", "uzonal_850hPa": "RdBu_r",
        "umeridional_500hPa": "RdBu_r", "umeridional_850hPa": "RdBu_r",
        "rain_log1p": "Blues",
    }
    fig, axes = plt.subplots(3, 4, figsize=(11.2, 7.6))
    axes = axes.ravel()
    for i, v in enumerate(INPUT_VARS):
        f = load(v)[sl]
        if v == "rain_log1p":
            kw = dict(vmin=float(f.min()), vmax=float(f.max()))
        else:
            lim = max(float(np.abs(f).max()), 1e-6)
            kw = dict(vmin=-lim, vmax=lim)
        im = axes[i].imshow(f, origin="lower", cmap=cmaps[v],
                            interpolation="nearest", **kw)
        axes[i].set_title(v, fontsize=8)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        cb = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    y = load("rain_total", t=FRAME + DELTA)[sl]
    imt = axes[10].imshow(y, origin="lower", cmap="Blues",
                          norm=LogNorm(vmin=0.1, vmax=max(float(y.max()), 0.5)),
                          interpolation="nearest")
    axes[10].set_title("target: rain_total (+3 h, mm)", fontsize=8, color="#b2182b")
    for s in axes[10].spines.values():
        s.set_color("#b2182b"); s.set_linewidth(2)
    axes[10].set_xticks([]); axes[10].set_yticks([])
    cb = fig.colorbar(imt, ax=axes[10], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    axes[11].axis("off")
    fig.suptitle(f"one training sample — piece {PIECE}, frame {FRAME}, block {BLOCK} — "
                 "each panel on its own scale (z-score for inputs, mm for the target)",
                 fontsize=10, y=0.995)
    fig.savefig(OUT / "data_sample_channels.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_split():
    train_b, val_b, test_b = make_patch_split(40, NR, NC, 1919810, 0.7, 0.15)

    def role(p, r, c):
        if (p, r, c) in test_b:
            return "test"
        if (p, r, c) in val_b:
            return "val"
        return "train"

    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.5], hspace=0.35)

    # top: spatial view for the example piece
    ax = fig.add_subplot(gs[0])
    pw = load("precipw")
    ax.imshow(pw, origin="lower", cmap="Greys", vmin=-3, vmax=3,
              interpolation="nearest", aspect="equal", extent=[0, W, 0, H], alpha=0.55)
    add_coastlines(ax, color="0.05", lw=0.7)
    for r in range(NR):
        for c in range(NC):
            ax.add_patch(Rectangle((c * P, r * P), P, P,
                                   fc=COL[role(PIECE, r, c)], ec="0.2", lw=0.6,
                                   alpha=0.55))
    ax.axhspan(NR * P, H, color="0.5", alpha=0.5, lw=0)
    ax.axvspan(NC * P, W, color="0.5", alpha=0.5, lw=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"piece {PIECE}: block assignment over the domain", fontsize=9)

    # bottom: full 40-day assignment matrix
    ax2 = fig.add_subplot(gs[1])
    mat = np.zeros((40, NR * NC, 3))
    from matplotlib.colors import to_rgb
    for p in range(40):
        for r in range(NR):
            for c in range(NC):
                mat[p, r * NC + c] = to_rgb(COL[role(p, r, c)])
    ax2.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")
    ax2.set_xlabel("block index (row·11 + col)", fontsize=8)
    ax2.set_ylabel("piece (day)", fontsize=8)
    ax2.set_title("all 880 blocks: every day contributes to every split", fontsize=9)
    ax2.legend(handles=[Patch(fc=COL["train"], ec="0.3", label="train 616"),
                        Patch(fc=COL["val"], label="val 132"),
                        Patch(fc=COL["test"], label="test 132")],
               loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8)
    fig.savefig(OUT / "data_split.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_domain()
    fig_sample_channels()
    fig_split()
    print("done")
