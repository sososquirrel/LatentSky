"""Paper architecture figure — the shared U-Net trunk, modern style.

Rounded vertical bars (height tracks spatial resolution, label carries the
channel count), soft indigo encoder / violet bottleneck / teal decoder,
curved dashed skips, colored chips for the model-specific edges: blue =
two-head heads + OLR-FiLM, orange = flow velocity head + FiLM(t).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.style.use("/glade/u/home/sabramian/LatentSky/docs/latentsky.mplstyle")

# palette (soft, "google-paper")
ENC = "#aab8f0"      # indigo fill
ENC_E = "#4f63d2"
BOT = "#c8a8ee"      # violet
BOT_E = "#8b5cf6"
DEC = "#9fd8cf"      # teal
DEC_E = "#0f9d8f"
BLUE = "#1a73e8"     # two-head accents
ORANGE = "#e8710a"   # flow accents
GRAY = "#9aa0a6"

CH = [64, 128, 256, 512, 1024]
RES = [128, 64, 32, 16, 8]

CM = 1 / 2.54
fig, ax = plt.subplots(figsize=(17.5 * CM, 7.2 * CM))
ax.set_xlim(0, 104)
ax.set_ylim(-14, 60)
ax.axis("off")

BW = 4.2            # bar width
X0, DX = 10, 6.2    # encoder start / step
XB = X0 + 5 * DX + 1.5          # bottleneck x
XD0 = XB + DX + 1.5             # decoder start


def bar(x, h, fc, ec, label, res=None):
    y0 = -h / 2 + 20
    ax.add_patch(FancyBboxPatch((x, y0), BW, h,
                                boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x + BW / 2, y0 + h / 2, label, rotation=90, ha="center",
            va="center", fontsize=7, color="0.15", zorder=4)
    if res is not None:
        ax.text(x + BW / 2, y0 - 3.2, f"{res}²", ha="center", va="top",
                fontsize=6.5, color="0.45")
    return x + BW / 2, y0 + h


def harrow(x0, x1, y, color=GRAY, lw=1.1, ls="-", rad=0.0, zorder=2):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=8, color=color, lw=lw,
                                 ls=ls, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}"))


# resolution -> bar height
def h_of(res):
    return 8 + 30 * np.log2(res / 4) / np.log2(32)


tops_e, tops_d = [], []
for i, (ch, res) in enumerate(zip(CH, RES)):
    xe = X0 + i * DX
    xd = XD0 + (4 - i) * DX
    _, te = bar(xe, h_of(res), ENC, ENC_E, f"{ch}", res)
    _, td = bar(xd, h_of(res), DEC, DEC_E, f"{ch}", res)
    tops_e.append((xe, te))
    tops_d.append((xd, td))
    if i < 4:
        y = 20
        harrow(xe + BW + 0.7, X0 + (i + 1) * DX - 0.7, y)
        harrow(XD0 + (4 - i - 1) * DX + BW + 0.7, xd - 0.7, y)

bar(XB, h_of(4), BOT, BOT_E, "2048", 4)
harrow(X0 + 4 * DX + BW + 0.7, XB - 0.7, 20)
harrow(XB + BW + 0.7, XD0 - 0.7, 20)

# skips: curved dashed arcs over the top
for (xe, te), (xd, td) in zip(tops_e, tops_d):
    ax.add_patch(FancyArrowPatch((xe + BW / 2, te + 0.8),
                                 (xd + BW / 2, td + 0.8),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color=GRAY, lw=0.9, ls=(0, (4, 3)),
                                 connectionstyle="arc3,rad=-0.25", zorder=1))

# input chip
ax.add_patch(FancyBboxPatch((1.0, 15.5), 6.4, 9,
                            boxstyle="round,pad=0.6,rounding_size=1.6",
                            fc="#f1f3f4", ec=GRAY, lw=1.1))
ax.text(4.2, 21.5, "10 ch", ha="center", va="center", fontsize=7)
ax.text(4.2, 17.8, "128²", ha="center", va="center", fontsize=6.5,
        color="0.45")
harrow(7.9, X0 - 0.7, 20)
ax.text(4.2, 27.5, "+ $x_t$ (flow: 11 ch)", ha="center", fontsize=6.5,
        color=ORANGE)

# output head chips
xh = XD0 + 4 * DX + BW + 2.2


def chip(y, text, color):
    ax.add_patch(FancyBboxPatch((xh, y), 19.5, 6,
                                boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc="white", ec=color, lw=1.3))
    ax.text(xh + 9.75, y + 3, text, ha="center", va="center", fontsize=6.5,
            color=color)
    ax.add_patch(FancyArrowPatch((XD0 + 4 * DX + BW + 0.4, 20),
                                 (xh - 0.2, y + 3), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=1.1,
                                 connectionstyle="arc3,rad=0.12"))


chip(33, "rain (softplus, mm)", BLUE)
chip(24, "wet / dry (logits)", BLUE)
chip(12, r"velocity $\hat{v}$", ORANGE)

# FiLM chips under the bottleneck
ax.add_patch(FancyBboxPatch((XB - 8.5, -6.5), 10.5, 5,
                            boxstyle="round,pad=0.5,rounding_size=1.4",
                            fc="white", ec=BLUE, lw=1.1))
ax.text(XB - 3.25, -4, "OLR-FiLM", ha="center", va="center", fontsize=6.5,
        color=BLUE)
ax.add_patch(FancyBboxPatch((XB + 4.0, -6.5), 10.5, 5,
                            boxstyle="round,pad=0.5,rounding_size=1.4",
                            fc="white", ec=ORANGE, lw=1.1))
ax.text(XB + 9.25, -4, "FiLM($t$)", ha="center", va="center", fontsize=6.5,
        color=ORANGE)
for x, colr in [(XB - 3.25, BLUE), (XB + 9.25, ORANGE)]:
    ax.add_patch(FancyArrowPatch((x, -1.2), (XB + BW / 2, 12.5),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color=colr, lw=1.0,
                                 connectionstyle="arc3,rad=0.15"))

# minimal annotations
ax.text(X0 + 2 * DX, 47.5, "encoder", ha="center", fontsize=7.5,
        color=ENC_E)
ax.text(XD0 + 2 * DX, 47.5, "decoder", ha="center", fontsize=7.5,
        color=DEC_E)
ax.text((X0 + XD0 + 4 * DX) / 2 + BW / 2, 54, "skip connections",
        ha="center", fontsize=6.5, color=GRAY)
ax.text(xh + 9.75, 41, "two-head", ha="center", fontsize=7, color=BLUE)
ax.text(xh + 9.75, 7, "flow", ha="center", fontsize=7, color=ORANGE)

fig.savefig("/glade/u/home/sabramian/LatentSky/paper/figures/fig1b_architecture.pdf",
            bbox_inches="tight")
print("written")
