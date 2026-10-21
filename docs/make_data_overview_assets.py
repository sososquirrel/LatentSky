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

from latentsky.evaluation.maps import add_coastlines, grid_lines  # noqa: F401  (re-exported)
from latentsky.layout import (  # noqa: F401  (re-exported for deck/ and docs/)
    EXAMPLE_BLOCK, EXAMPLE_FRAME, EXAMPLE_PIECE, H, INPUT_VARS, P, PIECES, W, load_frame)
from latentsky.training.patch_split import compute_patch_grid, make_patch_split

REPO = Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "docs" / "assets_model_reference"

PIECE, FRAME, BLOCK = EXAMPLE_PIECE, EXAMPLE_FRAME, EXAMPLE_BLOCK
DELTA = 13                    # 3 h, matching the example
NR, NC = compute_patch_grid(H, W, P)  # 2 x 11
COL = {"train": "#e9edf1", "val": "#e08214", "test": "#2166ac"}

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def load(var, t=FRAME, pid=PIECE):
    return load_frame(var, t, pid)


def fig_domain():
    pw = load("precipw")
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    im = ax.imshow(pw, origin="lower", cmap="BrBG", vmin=-3, vmax=3,
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
    y = load("rain_total", t=FRAME + DELTA)[sl]
    # rain_log1p inverted to mm with the recovered standardization constants
    # (model_reference §1), so it shares the target's exact color scale:
    # equal color = equal rain.
    rl = load("rain_log1p")[sl]
    rain_now = np.clip(np.expm1(rl * 0.118232 + 0.023456), 0, None)
    rain_norm = LogNorm(vmin=0.1, vmax=max(float(y.max()), float(rain_now.max()), 0.5))

    fig, axes = plt.subplots(3, 4, figsize=(11.2, 7.6))
    axes = axes.ravel()
    for i, v in enumerate(INPUT_VARS):
        if v == "rain_log1p":
            im = axes[i].imshow(rain_now, origin="lower", cmap="Blues",
                                norm=rain_norm, interpolation="nearest")
            axes[i].set_title("rain_log1p → rain now (mm)", fontsize=8)
        else:
            f = load(v)[sl]
            lim = max(float(np.abs(f).max()), 1e-6)
            im = axes[i].imshow(f, origin="lower", cmap=cmaps[v],
                                interpolation="nearest", vmin=-lim, vmax=lim)
            axes[i].set_title(v, fontsize=8)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        cb = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    imt = axes[10].imshow(y, origin="lower", cmap="Blues",
                          norm=rain_norm, interpolation="nearest")
    axes[10].set_title("target: rain_total (+3 h, mm)", fontsize=8, color="#b2182b")
    for s in axes[10].spines.values():
        s.set_color("#b2182b"); s.set_linewidth(2)
    axes[10].set_xticks([]); axes[10].set_yticks([])
    cb = fig.colorbar(imt, ax=axes[10], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    from matplotlib.colors import SymLogNorm
    diff = y - rain_now
    dlim = max(float(np.abs(diff).max()), 0.5)
    imd = axes[11].imshow(diff, origin="lower", cmap="RdBu_r",
                          norm=SymLogNorm(linthresh=0.1, vmin=-dlim, vmax=dlim),
                          interpolation="nearest")
    axes[11].set_title("Δ rain: target − now (mm)", fontsize=8)
    axes[11].set_xticks([]); axes[11].set_yticks([])
    cb = fig.colorbar(imd, ax=axes[11], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
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
            ax.text(c * P + P / 2, r * P + P / 2, str(r * NC + c),
                    ha="center", va="center", fontsize=9, color="0.15",
                    fontweight="bold")
    ax.axhspan(NR * P, H, color="0.5", alpha=0.5, lw=0)
    ax.axvspan(NC * P, W, color="0.5", alpha=0.5, lw=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"piece {PIECE} (day {PIECE + 1} of the 40-day DYAMOND record): "
                 "block assignment, numbered as in the matrix below", fontsize=9)

    # bottom: full 40-day assignment matrix
    ax2 = fig.add_subplot(gs[1])
    mat = np.zeros((40, NR * NC, 3))
    from matplotlib.colors import to_rgb
    for p in range(40):
        for r in range(NR):
            for c in range(NC):
                mat[p, r * NC + c] = to_rgb(COL[role(p, r, c)])
    ax2.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")
    ax2.add_patch(Rectangle((-0.5, PIECE - 0.5), NR * NC, 1, fill=False,
                            ec="#b2182b", lw=1.8, zorder=5))
    ax2.annotate(f"day {PIECE + 1} (top panel)", xy=(NR * NC - 0.5, PIECE),
                 xytext=(NR * NC + 0.4, PIECE), color="#b2182b", fontsize=8,
                 va="center", annotation_clip=False)
    ax2.set_xticks(range(0, NR * NC, 2))
    ax2.set_xlabel("block index (row·11 + col, as numbered above)", fontsize=8)
    ax2.set_ylabel("piece = day of record", fontsize=8)
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
