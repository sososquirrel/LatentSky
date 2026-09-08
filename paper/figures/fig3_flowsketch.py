"""Paper Fig 3 — flow-matching schematic (Sophie's sketch, pro version).

Pipeline row: U-Net forecast -> [flow matching] -> draw A / draw B (stacked,
aligned, same frame size) | truth. The box expands, via a dashed zoom inset,
into a detail panel enclosing the transport equation and a block-diagram
integrator: x_t -> v_theta -> [x Delta t] -> (+) -> x_{t+Dt}, bypass into the
sum, dashed return arc "repeat x32". Real +3 h fields. CPU ~2 min.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "docs"))
sys.path.insert(0, str(REPO / "paper" / "figures"))
from fig2_flowmethod import compute  # noqa: E402
from make_flow_steps import SCALE  # noqa: E402

matplotlib.rcParams["savefig.bbox"] = "tight"
OUT = REPO / "paper" / "figures"
CM = 1 / 2.54
ORANGE = "#c55a11"
ORANGE_BG = "#fdf0e6"
GRAY = "#666666"
INK = "0.2"


def to_rain(z):
    return np.clip(np.expm1(SCALE * np.clip(z, 0, 2.5)), 0, None)


def main():
    d = compute()
    fc = to_rain(d["zf"])
    dA = to_rain(d["drawA"])
    dB = to_rain(d["drawB"])
    truth = d["truth"]
    vmax = max(truth.max(), 1.0)
    norm = LogNorm(0.1, vmax)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#ffffff")

    fig, ax = plt.subplots(figsize=(17.5 * CM, 10.9 * CM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def show(f, x0, y0, label=None, side=None):
        MS = 15.0
        ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                  extent=(x0, x0 + MS, y0, y0 + MS),
                  interpolation="nearest", aspect="auto", zorder=2)
        ax.add_patch(plt.Rectangle((x0, y0), MS, MS, fill=False, ec="0.25",
                                   lw=0.9, zorder=3))
        if label and side == "below":
            ax.text(x0 + MS / 2, y0 - 1.5, label, ha="center", va="top",
                    fontsize=7.5)
        elif label and side == "right":
            ax.text(x0 + MS + 1.2, y0 + MS / 2, label, ha="left",
                    va="center", fontsize=7.5)
        return x0, x0 + MS, y0, y0 + MS

    def arrow(p0, p1, color=INK, lw=1.1, ls="-", ms=9, rad=0.0, z=4):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                     mutation_scale=ms, color=color, lw=lw,
                                     ls=ls, zorder=z,
                                     connectionstyle=f"arc3,rad={rad}"))

    # ---------------- pipeline row -----------------------------------------
    y_mid = 40.0
    f0, f1, fy0, fy1 = show(fc, 4, y_mid - 7.5, "U-Net forecast (+3 h)",
                            "below")
    # flow matching module
    bx0, bx1 = 23.5, 36.5
    by0, by1 = y_mid - 4.5, y_mid + 4.5
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                                boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc=ORANGE_BG, ec=ORANGE, lw=1.5, zorder=3))
    ax.text((bx0 + bx1) / 2, y_mid, "flow\nmatching", ha="center",
            va="center", fontsize=8.5, color=ORANGE, zorder=4)
    arrow((f1 + 0.6, y_mid), (bx0 - 0.9, y_mid))
    ax.text((f1 + bx0 - 0.3) / 2, y_mid + 1.2,
            r"$x_0$", ha="center", va="bottom", fontsize=7.5, color=INK)
    # conditioning
    arrow(((bx0 + bx1) / 2, 60.5), ((bx0 + bx1) / 2, by1 + 0.9))
    ax.text((bx0 + bx1) / 2 + 1.3, 58.5, "10 channels $c$", ha="left",
            va="center", fontsize=7)
    # draws, stacked and aligned
    dx = 44.5
    a0, a1, ay0, ay1 = show(dA, dx, y_mid + 1.5, "draw A", "right")
    b0, b1, bby0, bby1 = show(dB, dx, y_mid - 16.5, "draw B", "right")
    arrow((bx1 + 0.9, y_mid + 1.8), (dx - 0.9, ay0 + 5.5))
    arrow((bx1 + 0.9, y_mid - 1.8), (dx - 0.9, bby1 - 5.5))
    # separator + truth
    ax.plot([70.5, 70.5], [bby0, ay1], color="0.6", lw=0.8)
    show(truth, 76, y_mid - 7.5, "truth (+3 h)", "below")

    # ---------------- zoom inset: detail panel -----------------------------
    px0, px1, py0, py1 = 4, 96, 1.0, 21.5
    ax.add_patch(FancyBboxPatch((px0, py0), px1 - px0, py1 - py0,
                                boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc="#fafafa", ec="0.55", lw=1.0,
                                ls=(0, (5, 3)), zorder=1))
    # trapezoid guides from box bottom corners to the panel top
    for xa, xb in [(bx0 + 1.0, 20), (bx1 - 1.0, 40)]:
        ax.plot([xa, xb], [by0 - 0.8, py1 + 0.6], color="0.55", lw=0.9,
                ls=(0, (5, 3)), zorder=1)

    # equation, left half of the panel
    ax.text(19, 12.5,
            r"$x_1 \;=\; x_0 \,+\, \int_0^1"
            r" \hat{v}_\theta\,(x_t,\, t,\, c)\;\mathrm{d}t$",
            ha="center", va="center", fontsize=9.5)

    # block-diagram integrator, right half
    ly = 11.5
    bh = 5.6

    def node(xc, w, text, ec=INK, fc="white", tc="0.15", fs=7.5):
        ax.add_patch(FancyBboxPatch((xc - w / 2, ly - bh / 2), w, bh,
                                    boxstyle="round,pad=0.45,rounding_size=1.1",
                                    fc=fc, ec=ec, lw=1.1, zorder=3))
        ax.text(xc, ly, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4)

    x_xt, x_fm, x_dt, x_sum, x_out = 42.5, 55.5, 68, 77.5, 88.5
    node(x_xt, 7, r"$x_t$")
    node(x_fm, 9, r"$\hat{v}_\theta$", ec=ORANGE, fc=ORANGE_BG, tc=ORANGE,
         fs=8)
    node(x_dt, 7.5, r"$\times\,\Delta t$", fs=7)
    node(x_out, 9, r"$x_{t+\Delta t}$")
    # summation node: circle with a crisp plus
    R = 1.7
    ax.add_patch(Circle((x_sum, ly), R, fc="white", ec=INK, lw=1.1,
                        zorder=3))
    ax.plot([x_sum - 0.9, x_sum + 0.9], [ly, ly], color=INK, lw=1.0,
            zorder=4)
    ax.plot([x_sum, x_sum], [ly - 0.9, ly + 0.9], color=INK, lw=1.0,
            zorder=4)
    # main path
    arrow((x_xt + 4.1, ly), (x_fm - 5.1, ly))
    arrow((x_fm + 5.1, ly), (x_dt - 4.4, ly))
    arrow((x_dt + 4.4, ly), (x_sum - R - 0.3, ly))
    arrow((x_sum + R + 0.3, ly), (x_out - 5.1, ly))
    # bypass: x_t branches under the row into the sum
    xb = x_xt + 5.6
    ax.plot([xb], [ly], marker="o", ms=2.6, color=INK, zorder=5)
    ax.plot([xb, xb, x_sum], [ly, 4.6, 4.6], color=INK, lw=1.1, zorder=2,
            solid_capstyle="round")
    arrow((x_sum, 4.6), (x_sum, ly - R - 0.3))
    # return arc: repeat x32
    arrow((x_out - 1.5, ly + bh / 2 + 0.6), (x_xt + 1.5, ly + bh / 2 + 0.6),
          ls=(0, (4, 3)), lw=0.9, rad=0.28, ms=7, color=GRAY)
    ax.text((x_xt + x_out) / 2, 17.6,
            r"repeat $\times 32$ (Heun),  $t: 0 \rightarrow 1$",
            ha="center", va="center", fontsize=6.5, color=GRAY)

    fig.savefig(OUT / "fig3_flowsketch.pdf", dpi=300)
    print("saved fig3_flowsketch.pdf")


if __name__ == "__main__":
    main()
