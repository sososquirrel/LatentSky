"""Supplement Fig S1 — the U-Net from input to output, in the style of main Fig 1.

Left, the ten input fields of the example block as the model receives them
(standardized, 3 x 4 grid at the size of the outputs; the current rain
last, blue frame); centre, the U-Net glyph of
Fig 1c (fig2_pipeline.unet_glyph); right, its two outputs at +3 h, the rain (mm)
and the wet/dry logits. Runs the +3 h U-Net on CPU (~30 s).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "paper_figures"))
from figS1_io import CMAPS, LABELS, load_block, predict  # noqa: E402
from fig1_pipeline import BLUE, BLUE_BG, INK, unet_glyph  # noqa: E402
from paperfig import W2  # noqa: E402

from latentsky.layout import INPUT_VARS  # noqa: E402

OUT = REPO / "paper" / "figures_si"
PAPER_FONT = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
              "axes.unicode_minus": False, "axes.formatter.use_mathtext": True}


def main():
    x = load_block()
    rain, logit = predict(x)
    plt.rcParams.update(PAPER_FONT)
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.50))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=INK,
                                     lw=0.9, shrinkA=0, shrinkB=0, zorder=6))

    # inputs: 3 x 4 maps, each with its variable name, the size of the outputs
    S, G, NC = 11.0, 1.0, 4
    for k, var in enumerate(INPUT_VARS):
        r, c = divmod(k, NC)
        x0, y0 = 1.0 + c * (S + G), 33.0 - r * (S + 3.6)
        f = x[k]
        if CMAPS[var] == "RdBu_r":
            m = float(np.percentile(np.abs(f), 99))
            vmin, vmax = -m, m
        else:
            vmin, vmax = np.percentile(f, [1, 99])
        ax.imshow(f, origin="lower", cmap=CMAPS[var], vmin=vmin, vmax=vmax, aspect="auto",
                  extent=(x0, x0 + S, y0, y0 + S), interpolation="nearest", zorder=3)
        last = var == "rain_log1p"
        ax.add_patch(Rectangle((x0, y0), S, S, fill=False, ec=BLUE if last else "0.35",
                               lw=1.4 if last else 0.5, zorder=4))
        ax.text(x0 + S / 2, y0 + S + 0.5, LABELS[var], ha="center", va="bottom", fontsize=6.5)
    ax.text(1.0 + NC / 2 * (S + G) - G / 2, 47.4, r"10 input fields at time $t$", ha="center",
            va="bottom", fontsize=8.5)
    # the U-Net, as in Fig 1c
    xu, yc = 54.0, 25.0
    xu1, _ = unet_glyph(ax, xu, yc, BLUE, BLUE_BG, "U-Net", 10, 2)
    arrow((1.0 + NC * (S + G) - G + 0.6, yc), (xu, yc))
    # outputs: rain and wet/dry logits
    So, xo = S, 80.0
    for y0, img, kw, title in (
            (29.0, rain, dict(cmap="Blues", norm=LogNorm(0.1, max(float(rain.max()), 1.0))),
             r"rain at $t + 3$ h (mm)"),
            (7.0, logit, dict(cmap="BrBG",
                              vmin=-np.percentile(np.abs(logit), 99),
                              vmax=np.percentile(np.abs(logit), 99)),
             "wet/dry logits")):
        ax.imshow(img, origin="lower", aspect="auto", extent=(xo, xo + So, y0, y0 + So),
                  interpolation="nearest", zorder=3, **kw)
        ax.add_patch(Rectangle((xo, y0), So, So, fill=False, ec=BLUE, lw=1.0, zorder=4))
        ax.text(xo + So / 2, y0 + So + 0.5, title, ha="center", va="bottom", fontsize=7.5)
        arrow((xu1, yc), (xo - 0.3, y0 + So / 2))
    fig.savefig(OUT / "figS1_unet.pdf")
    print("saved figS1_unet.pdf")


if __name__ == "__main__":
    main()
