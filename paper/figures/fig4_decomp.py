"""Paper Fig 5 — the error decomposition on matched pairs (+3 h, full record).

(A) two-head U-Net, (B) flow per draw, in the (dlnA, dlnrho) plane, same axes.
For each of three size bands: the region holding half of the matched pairs
(shaded) and 90 % of them (line), from a smoothed 2-D histogram, and the
rain-weighted barycenter joined to the origin (the mean bias of the band).

Caption sentence: the deterministic model sits in the compensating quadrant
(too large, too weak, worst for small systems); the flow's barycenters
collapse toward the origin at every scale. CPU ~1 min.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paperfig import W2
from scipy.ndimage import gaussian_filter

EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")

LIM = 2.4
# three size bands, small to large, in three muted colours
BANDS = [(0, 65, r"small, $<$ 65 km", "#c8843c"),
         (65, 200, "medium, 65 to 200 km", "#4f8f86"),
         (200, 1e9, r"large, $>$ 200 km", "#2f4868")]
EDGES = np.linspace(-3.2, 3.2, 129)
MIDS = 0.5 * (EDGES[1:] + EDGES[:-1])


def hdr_levels(h, fracs):
    """Density levels whose super-level sets hold the given fractions of the mass."""
    v = np.sort(h.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return [v[np.searchsorted(c, f)] for f in fracs]


def panel(ax, df, letter, label):
    m = df[df.role == "match"]
    ax.axhline(0, color="0.75", lw=0.5, zorder=1)
    ax.axvline(0, color="0.75", lw=0.5, zorder=1)
    xs = np.array([-LIM, LIM])
    ax.plot(xs, -xs, color="0.45", lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.text(1.72, -1.60, r"$\mathrm{d}\ln P = 0$", color="0.4", fontsize=6.5,
            rotation=-45, ha="center", va="center")
    for z, (lo, hi, _, col) in enumerate(reversed(BANDS)):
        b = m[(m.scale >= lo) & (m.scale < hi)]
        h, _, _ = np.histogram2d(b.dlnA, b.dlnrho, bins=[EDGES, EDGES])
        h = gaussian_filter(h.T, 3.5)                 # removes the pixel-count striping in dlnA
        l90, l50 = hdr_levels(h, (0.9, 0.5))
        ax.contourf(MIDS, MIDS, h, levels=[l50, h.max() * 1.01], colors=[col], alpha=0.32,
                    zorder=2 + z)
        ax.contour(MIDS, MIDS, h, levels=[l50], colors=[col], linewidths=0.9, zorder=2.5 + z)
        ax.contour(MIDS, MIDS, h, levels=[l90], colors=[col], linewidths=0.6,
                   linestyles=[(0, (4, 2))], zorder=2.5 + z)
    for lo, hi, _, col in BANDS:                      # barycentres on top: the mean bias
        b = m[(m.scale >= lo) & (m.scale < hi)]
        bx = np.average(b.dlnA, weights=b.P_t)
        by = np.average(b.dlnrho, weights=b.P_t)
        ax.plot([0, bx], [0, by], color=col, lw=1.2, zorder=8, solid_capstyle="round")
        ax.plot(bx, by, "o", ms=5.5, mfc=col, mec="white", mew=0.9, zorder=9)
    ax.plot(0, 0, "+", ms=5, color="0.2", mew=0.7, zorder=7)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks(range(-2, 3))
    ax.set_yticks(range(-2, 3))
    ax.tick_params(length=2.5, width=0.5, labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5)
    ax.set_xlabel(r"area error  $\mathrm{d}\ln A$", fontsize=8.5, labelpad=3)
    ax.text(0.0, -0.10, r"$\leftarrow$ too small", transform=ax.transAxes, ha="left",
            va="top", fontsize=6.5, color="0.4")
    ax.text(1.0, -0.10, r"too large $\rightarrow$", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color="0.4")
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontfamily="cmb10")
    ax.text(0.075, 1.03, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
    ax.text(1.0, 1.03, f"${len(m):,}$ matched pairs".replace(",", r"\,"), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="0.4")


def main():
    # the paper's text font
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})
    WF = 0.84 * W2
    fig, axs = plt.subplots(1, 2, figsize=(WF, WF * 0.60), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.235, wspace=0.08)
    panel(axs[0], pd.read_csv(EVAL / "systems_alltest_lead3h.csv.gz"), "A.", "Two-head U-Net")
    panel(axs[1], pd.read_csv(EVAL / "systems_alltest_flow_lead3h.csv.gz"), "B.", "Flow, per draw")
    axs[0].set_ylabel(r"intensity error  $\mathrm{d}\ln\rho$", fontsize=8.5, labelpad=3)
    axs[0].text(-0.125, 0.0, r"$\leftarrow$ too weak", transform=axs[0].transAxes, ha="center",
                va="bottom", rotation=90, fontsize=6.5, color="0.4")
    axs[0].text(-0.125, 1.0, r"too strong $\rightarrow$", transform=axs[0].transAxes, ha="center",
                va="top", rotation=90, fontsize=6.5, color="0.4")
    handles = [Patch(fc=c, ec=c, alpha=0.6, label=name) for _, _, name, c in BANDS]
    handles += [Patch(fc="0.6", ec="0.3", alpha=0.5, label="half of the pairs"),
                Line2D([], [], color="0.3", lw=0.6, ls=(0, (4, 2)), label="90% of the pairs"),
                Line2D([], [], color="0.3", lw=1.2, marker="o", ms=4.5, mfc="0.3", mec="white",
                       label="rain-weighted mean error")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False,
               columnspacing=2.2, handlelength=1.6, handletextpad=0.6,
               bbox_to_anchor=(0.54, 0.0))
    fig.savefig(OUT / "fig4_decomp.pdf")
    print("saved fig4_decomp.pdf")


if __name__ == "__main__":
    main()
