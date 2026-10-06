"""Paper Fig 7b (WB2 POD) — POD by scale at +12 h, single panel.

Same computation as fig6bc_wb2.py panel B (cached 2020-08-15 12 h file):
GraphCast, FuXi, GenCast mean, one GenCast member (per-draw), persistence.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from fig6bc_wb2 import COLW, pod_bands, MIDS
from fig4_decomp import PAPER_FONT
from fig5_horizon import log_ticks
from paperfig import W2

from latentsky.metrics import cell_area_km2, match_systems

CACHE = Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = Path("/glade/u/home/sabramian/LatentSky/paper/figures")
THR, IOU_MIN = 4.8, 0.1


def main():
    ds = xr.open_dataset(CACHE / "wb2_12h_models_2020-08-15.nc")
    area_row = cell_area_km2(ds.lat.size, ds.lon.size,
                             lat_extent=(float(ds.lat[0]) - 0.125,
                                         float(ds.lat[-1]) + 0.125))

    def to_mm(a):
        import numpy as np
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

    NAMES = {"graphcast": "GraphCast", "fuxi": "FuXi", "gencast mean": "GenCast mean",
             "gencast member": "GenCast member", "persistence": "persistence"}
    plt.rcParams.update(PAPER_FONT)
    fig, ax = plt.subplots(figsize=(W2 * 0.62, W2 * 0.45))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.16)
    ax.grid(True, color="0.88", lw=0.4)
    ax.set_axisbelow(True)
    for name, pod in series.items():
        ls = "--" if name == "persistence" else "-"
        ax.plot(MIDS, pod, ls, color=COLW[name], marker="o", ms=3.5,
                lw=1.5, label=NAMES[name])
    ax.set_xscale("log")
    ax.set_xlim(30, 3000)      # the bands run from 38 to 2380 km; margins on both sides
    ax.set_ylim(0, 1.04)
    log_ticks(ax.xaxis, [30, 50, 100, 200, 500, 1000, 2000])
    ax.set_xlabel(r"system scale $L = \sqrt{A}$ (km)")
    ax.set_ylabel("Probability of Detection (POD)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.0, 1.03, "Detection by system size, WeatherBench 2 models, +12 h",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
    fig.savefig(OUT / "fig7b_pod.pdf")
    print("saved fig7b_pod.pdf")


if __name__ == "__main__":
    main()
