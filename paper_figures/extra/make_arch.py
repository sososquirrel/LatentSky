"""Paper architecture figure — two classic vertical U-Nets, modern style.

(a) the deterministic two-head model    (b) the flow's velocity network

Same trunk drawn identically in both panels: encoder descending on the left,
decoder ascending on the right, dashed skip connections across, box width
tracking the channel count. Panel differences: input channels (10 vs 11),
bottleneck FiLM conditioning (OLR vs t), and the output heads.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.style.use("/glade/u/home/sabramian/LatentSky/docs/latentsky.mplstyle")

ENC, ENC_E = "#aab8f0", "#4f63d2"
BOT, BOT_E = "#c8a8ee", "#8b5cf6"
DEC, DEC_E = "#9fd8cf", "#0f9d8f"
BLUE, ORANGE, GRAY = "#1a73e8", "#e8710a", "#9aa0a6"

CH = [64, 128, 256, 512, 1024]
RES = [128, 64, 32, 16, 8]
ROW_Y = [57, 47, 37, 27, 17]
Y_BOT = 6
BH = 6.2
XL, XR = 27, 73


def w_of(ch):
    return 10 + 4.5 * np.log2(ch / 64)


def rbox(ax, xc, y, w, fc, ec, label):
    ax.add_patch(FancyBboxPatch((xc - w / 2, y), w, BH,
                                boxstyle="round,pad=0.5,rounding_size=1.3",
                                fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(xc, y + BH / 2, label, ha="center", va="center", fontsize=7,
            color="0.15", zorder=4)


def arrow(ax, p0, p1, color=GRAY, lw=1.0, ls="-", rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def chip(ax, xc, y, w, text, color, fc="white"):
    ax.add_patch(FancyBboxPatch((xc - w / 2, y), w, 6,
                                boxstyle="round,pad=0.5,rounding_size=1.3",
                                fc=fc, ec=color, lw=1.2, zorder=3))
    ax.text(xc, y + 3, text, ha="center", va="center", fontsize=6.5,
            color=color, zorder=4)


def draw_unet(ax, flow):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 84)
    ax.axis("off")
    acc = ORANGE if flow else BLUE

    for i, (ch, res, y) in enumerate(zip(CH, RES, ROW_Y)):
        w = w_of(ch)
        rbox(ax, XL, y, w, ENC, ENC_E, f"{ch}")
        rbox(ax, XR, y, w, DEC, DEC_E, f"{ch}")
        ax.text(3, y + BH / 2, f"{res}²", ha="left", va="center",
                fontsize=6.5, color="0.45")
        # skip connection
        arrow(ax, (XL + w / 2 + 1, y + BH / 2), (XR - w / 2 - 1, y + BH / 2),
              ls=(0, (4, 3)), lw=0.9, ms=7)
        # down (pool) / up (transpose conv)
        if i < 4:
            wn = w_of(CH[i + 1])
            arrow(ax, (XL - w / 2, y + 1), (XL - wn / 2 + 1, ROW_Y[i + 1] + BH - 1))
            arrow(ax, (XR + wn / 2 - 1, ROW_Y[i + 1] + BH - 1), (XR + w / 2, y + 1))
    ax.text(3, Y_BOT + BH / 2, "4²", ha="left", va="center", fontsize=6.5,
            color="0.45")

    # bottleneck
    wb = w_of(2048)
    rbox(ax, 50, Y_BOT, wb, BOT, BOT_E, "2048")
    w5 = w_of(1024)
    arrow(ax, (XL - w5 / 2, ROW_Y[4] + 1), (50 - wb / 2 + 1, Y_BOT + BH - 1))
    arrow(ax, (50 + wb / 2 - 1, Y_BOT + BH - 1), (XR + w5 / 2, ROW_Y[4] + 1))

    # input chip
    n_in = "11 ch" if flow else "10 ch"
    chip(ax, XL, 73, 16, f"input {n_in} @ 128²", "0.35", fc="#f1f3f4")
    arrow(ax, (XL, 72.5), (XL, ROW_Y[0] + BH + 0.8))
    if flow:
        ax.text(XL, 80.5, "10 channels + flow state $x_t$", ha="center",
                fontsize=6.5, color=ORANGE)
    else:
        ax.text(XL, 80.5, "10 weather channels", ha="center", fontsize=6.5,
                color="0.45")

    # output heads
    if flow:
        chip(ax, XR, 73, 17, r"velocity $\hat{v}$", ORANGE)
        arrow(ax, (XR, ROW_Y[0] + BH + 0.8), (XR, 72.5), color=ORANGE, lw=1.1)
    else:
        chip(ax, XR - 10, 73, 17, "rain (mm)", BLUE)
        chip(ax, XR + 10.5, 73, 19, "wet / dry (logits)", BLUE)
        arrow(ax, (XR - 2, ROW_Y[0] + BH + 0.8), (XR - 9, 72.5), color=BLUE,
              lw=1.1, rad=-0.1)
        arrow(ax, (XR + 2, ROW_Y[0] + BH + 0.8), (XR + 9, 72.5), color=BLUE,
              lw=1.1, rad=0.1)

    # FiLM chip -> bottleneck
    film = "FiLM($t$)" if flow else "OLR-FiLM"
    chip(ax, 87, Y_BOT + 1, 15, film, acc)
    arrow(ax, (87 - 8, Y_BOT + 4), (50 + wb / 2 + 1, Y_BOT + 3.2),
          color=acc, lw=1.0)

    # side annotations
    ax.text(XL - 12.5, 38, "encoder", rotation=90, ha="center", va="center",
            fontsize=7, color=ENC_E)
    ax.text(XR + 12.5, 38, "decoder", rotation=90, ha="center", va="center",
            fontsize=7, color=DEC_E)
    ax.text(50, ROW_Y[0] + BH + 2.0, "skip connections", ha="center",
            fontsize=6, color=GRAY)


CM = 1 / 2.54
fig, axs = plt.subplots(1, 2, figsize=(17.5 * CM, 8.6 * CM))
draw_unet(axs[0], flow=False)
draw_unet(axs[1], flow=True)
axs[0].text(0.0, 1.0, "A. two-head (deterministic)",
            transform=axs[0].transAxes, fontsize=8, fontweight="bold",
            ha="left", va="top")
axs[1].text(0.0, 1.0, "B. velocity network (flow matching)",
            transform=axs[1].transAxes, fontsize=8, fontweight="bold",
            ha="left", va="top")
fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.99,
                    bottom=0.01)
fig.savefig("/glade/u/home/sabramian/LatentSky/paper_figures/extra/fig1b_architecture.pdf",
            bbox_inches="tight")
print("written")
