"""Paper Fig 3 — the flow-matching schematic with real fields (Sophie's
sketch, 2026-09-08): U-Net forecast -> [flow matching] -> two draws | truth;
the box expands below into the transport equation and the unrolled
integration loop  x_t -> [FM] -> v_t -> (x dt) -> (+) -> x_t.

Fields are the +3 h standard example (compute() of fig2_flowmethod).
CPU ~2 min (two 32-step draws).
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
GRAY = "#666666"


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

    fig, ax = plt.subplots(figsize=(17.5 * CM, 9.6 * CM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.set_aspect("auto")
    ax.axis("off")

    def show(f, x0, x1, y0, y1, label=None, below=True):
        ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                  extent=(x0, x1, y0, y1), interpolation="nearest",
                  aspect="auto", zorder=2)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="0.25", lw=0.9, zorder=3))
        if label:
            ax.text((x0 + x1) / 2, y0 - 1.4 if below else y1 + 1.2, label,
                    ha="center", va="top" if below else "bottom",
                    fontsize=7.5)

    def arrow(p0, p1, color="0.2", lw=1.1, ls="-", ms=9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                     mutation_scale=ms, color=color, lw=lw,
                                     ls=ls, zorder=4))

    # --- top row -----------------------------------------------------------
    MS = 13.5                       # map size
    y_mid = 33                      # vertical center of the row
    show(fc, 2, 2 + MS, y_mid - MS / 2, y_mid + MS / 2,
         "U-Net forecast (+3 h)")
    # flow matching box
    bx0, bx1, by0, by1 = 22.5, 36.5, y_mid - 4.2, y_mid + 4.2
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                                boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc="#fdf0e6", ec=ORANGE, lw=1.5, zorder=3))
    ax.text((bx0 + bx1) / 2, y_mid, "flow\nmatching", ha="center",
            va="center", fontsize=8, color=ORANGE, zorder=4)
    arrow((2 + MS + 0.6, y_mid), (bx0 - 0.8, y_mid))
    ax.text(18.3, y_mid + 1.3,
            r"$x_0 = z_{\mathrm{fc}} + \sigma\varepsilon$", ha="center",
            va="bottom", fontsize=6, color=GRAY)
    # conditioning arrow from above
    arrow(((bx0 + bx1) / 2, 53.5), ((bx0 + bx1) / 2, by1 + 0.9))
    ax.text((bx0 + bx1) / 2 + 1.2, 51.5, "10 channels $c$", ha="left",
            va="center", fontsize=7)
    # two draws, stacked
    dx0 = 48
    show(dA, dx0, dx0 + MS, y_mid + 1.2, y_mid + 1.2 + MS, None)
    ax.text(dx0 + MS + 1.0, y_mid + 1.2 + MS / 2, "draw A", ha="left",
            va="center", fontsize=7.5)
    show(dB, dx0 + 2.5, dx0 + 2.5 + MS, y_mid - 1.2 - MS, y_mid - 1.2, None)
    ax.text(dx0 + 2.5 + MS + 1.0, y_mid - 1.2 - MS / 2, "draw B", ha="left",
            va="center", fontsize=7.5)
    arrow((bx1 + 0.9, y_mid + 1.5), (dx0 - 0.8, y_mid + 5.5))
    arrow((bx1 + 0.9, y_mid - 1.5), (dx0 + 1.7, y_mid - 5.5))
    # separator + truth
    ax.plot([74.5, 74.5], [y_mid - MS / 2 - 2, y_mid + MS / 2 + 2],
            color="0.6", lw=0.8)
    show(truth, 78, 78 + MS, y_mid - MS / 2, y_mid + MS / 2, "truth (+3 h)")

    # --- expansion: equation ----------------------------------------------
    arrow((bx0 + 1.5, by0 - 0.8), (22, 22.5), color=GRAY, lw=0.8,
          ls=(0, (3, 3)), ms=0.1)
    arrow((bx1 - 1.5, by0 - 0.8), (44, 22.5), color=GRAY, lw=0.8,
          ls=(0, (3, 3)), ms=0.1)
    ax.text(32, 19.5,
            r"$x_1 \;=\; x_0 \,+\, \int_0^1 \hat{v}\,(x_t,\, t,\, c)"
            r"\;\mathrm{d}t$",
            ha="center", va="center", fontsize=9.5)

    # --- unrolled integration loop ----------------------------------------
    ly = 9.5                        # box row center
    lh = 5.6

    def lbox(xc, w, text, ec="0.25", fc="white", tc="0.15"):
        ax.add_patch(FancyBboxPatch((xc - w / 2, ly - lh / 2), w, lh,
                                    boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2, zorder=3))
        ax.text(xc, ly, text, ha="center", va="center", fontsize=7.5,
                color=tc, zorder=4)

    lbox(20, 8, r"$x_t$")
    lbox(36.5, 10, "FM", ec=ORANGE, fc="#fdf0e6", tc=ORANGE)
    lbox(53, 8, r"$\hat{v}_t$")
    arrow((24.2, ly), (31.3, ly))
    arrow((41.7, ly), (48.8, ly))
    # times dt circle under v_t
    cy = 2.6
    ax.add_patch(Circle((53, cy), 1.7, fc="white", ec="0.25", lw=1.1,
                        zorder=3))
    ax.text(53, cy, r"$\times\,\mathrm{d}t$", ha="center", va="center",
            fontsize=6, zorder=4)
    arrow((53, ly - lh / 2 - 0.2), (53, cy + 1.9))
    # plus circle under x_t, feedback line
    ax.add_patch(Circle((20, cy), 1.7, fc="white", ec="0.25", lw=1.1,
                        zorder=3))
    ax.text(20, cy, "+", ha="center", va="center", fontsize=8, zorder=4)
    ax.plot([51.3, 21.7], [cy, cy], color="0.2", lw=1.1, zorder=2)
    arrow((20, cy + 1.9), (20, ly - lh / 2 - 0.2))
    ax.text(63, ly - 1.0,
            r"$x_{t+\mathrm{d}t} = x_t + \hat{v}_t\,\mathrm{d}t$",
            ha="left", va="center", fontsize=7.5)
    ax.text(63, ly - 4.6, r"32 Heun steps, $t: 0 \rightarrow 1$",
            ha="left", va="center", fontsize=6.5, color=GRAY)

    fig.savefig(OUT / "fig3_flowsketch.pdf", dpi=300)
    print("saved fig3_flowsketch.pdf")


if __name__ == "__main__":
    main()
