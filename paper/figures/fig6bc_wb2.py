"""Paper Fig 6 panels (b)+(c) — WB2 at +12 h, one styled PDF.

(b) POD by scale, August 2020 month of the belt map's models: GraphCast,
    FuXi, GenCast mean, one GenCast member (per-draw), persistence
    — computed from the cached 2-init 12 h file (2020-08-15, as on the page).
(c) month-pooled decomposition clouds (matches_12h_month.csv.gz), one panel
    per model, +-1.5 axes, precip-weighted barycenters, thin scale colorbar.

Caption sentence: the member adds detection at 150-500 km at a FAR cost and
erases the too-weak bias; below 60 km nothing detects. CPU ~1 min.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from paperfig import W2, panel_label

from latentsky.metrics import cell_area_km2, match_systems

CACHE = Path("/glade/derecho/scratch/sabramian/wb2_eval")
WB = Path("/glade/u/home/sabramian/LatentSky/wb2_benchmark")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")

THR, IOU_MIN = 4.8, 0.1
BANDS = np.geomspace(30, 3000, 11)
MIDS = np.sqrt(BANDS[:-1] * BANDS[1:])
NMIN = 20

COLW = {"graphcast": "#1b7837", "fuxi": "#762a83", "gencast mean": "#35978f",
        "gencast member": "#b2182b", "persistence": "#8a8a8a"}
CLOUD_MODELS = ["graphcast", "fuxi", "gencast mean", "gencast member"]
BARY_BANDS = [(0, 100), (100, 400), (400, 1e9)]
BARY_C = ["#40004b", "#1b7837", "#c51b7d"]


def pod_bands(dfs):
    df = pd.concat(dfs, ignore_index=True)
    b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
    pod = np.full(len(MIDS), np.nan)
    for k in range(len(MIDS)):
        g = df[(b == k) & df.role.isin(["match", "miss"])]
        if len(g) >= NMIN:
            pod[k] = (g.role == "match").mean()
    return pod


def main():
    ds = xr.open_dataset(CACHE / "wb2_12h_models_2020-08-15.nc")
    area_row = cell_area_km2(ds.lat.size, ds.lon.size,
                             lat_extent=(float(ds.lat[0]) - 0.125,
                                         float(ds.lat[-1]) + 0.125))

    def to_mm(a):
        return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a

    truths = [to_mm(ds.truth.values[i]) for i in range(ds.init.size)]

    def score(fields):
        return pod_bands([match_systems(yt, pf, THR, area_row,
                                        iou_min=IOU_MIN)
                          for yt, pf in zip(truths, fields)])

    series = {
        "graphcast": score([to_mm(ds.graphcast.values[i])
                            for i in range(ds.init.size)]),
        "fuxi": score([to_mm(ds.fuxi.values[i])
                       for i in range(ds.init.size)]),
        "gencast mean": score([to_mm(ds["mean"].values[i])
                               for i in range(ds.init.size)]),
        "gencast member": score([to_mm(ds.member.values[i, 0])
                                 for i in range(ds.init.size)]),
        "persistence": score([to_mm(ds.persistence.values[i])
                              for i in range(ds.init.size)]),
    }

    fig = plt.figure(figsize=(W2, W2 * 0.78))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0],
                          hspace=0.42, wspace=0.12)
    ax_pod = fig.add_subplot(gs[0, 1:3])

    for name, pod in series.items():
        ls = "--" if name == "persistence" else "-"
        ax_pod.plot(MIDS, pod, ls, color=COLW[name], marker="o", ms=3,
                    lw=1.4, label=name)
    ax_pod.set_xscale("log")
    ax_pod.set_xlim(40, 3000)
    ax_pod.set_ylim(0, 1.0)
    ax_pod.set_xlabel("system scale L = √A (km)")
    ax_pod.set_ylabel("POD")
    ax_pod.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    panel_label(ax_pod, "(b)")

    df = pd.read_csv(WB / "matches_12h_month.csv.gz")
    sc = None
    axes_c = []
    for j, model in enumerate(CLOUD_MODELS):
        ax = fig.add_subplot(gs[1, j])
        axes_c.append(ax)
        m = df[df.model == model]
        sc = ax.scatter(m.dlnA, m.dlnrho, c=m.scale, cmap="viridis",
                        norm=LogNorm(30, 3000), s=1.8, alpha=0.4, lw=0,
                        rasterized=True)
        for (lo, hi), c in zip(BARY_BANDS, BARY_C):
            b = m[(m.scale >= lo) & (m.scale < hi)]
            if len(b) < NMIN:
                continue
            ax.scatter(np.average(b.dlnA, weights=b.P_t),
                       np.average(b.dlnrho, weights=b.P_t), marker="X",
                       s=70, color=c, edgecolors="white", linewidths=0.7,
                       zorder=5)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(0, color="0.6", lw=0.5)
        xs = np.array([-1.5, 1.5])
        ax.plot(xs, -xs, ":", color="#c0392b", lw=0.8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"d$\ln A$")
        if j == 0:
            ax.set_ylabel(r"d$\ln\rho$")
            panel_label(ax, "(c)")
        else:
            ax.set_yticklabels([])
        ax.text(0.97, 0.03, f"{model}\n{len(m):,} pairs",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.25")

    cb = fig.colorbar(sc, ax=axes_c, orientation="horizontal",
                      fraction=0.05, pad=0.22, aspect=60)
    cb.set_label("system scale L = √A (km)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_alpha(1)

    fig.savefig(OUT / "fig6bc_wb2.pdf", dpi=300)
    print("saved fig6bc_wb2.pdf")


if __name__ == "__main__":
    main()
