"""Paper Fig 7b (WB2 POD) — POD by scale at +12 h, single panel.

Same computation as fig6bc_wb2.py panel B (cached 2020-08-15 12 h file):
GraphCast, FuXi, GenCast mean, one GenCast member (per-draw), persistence.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from fig6bc_wb2 import BANDS, COLW, MIDS, NMIN
from fig4_decomp import PAPER_FONT
from fig5_horizon import log_ticks
from paperfig import W2

from latentsky.metrics import cell_area_km2, match_systems, summarize_by_scale

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
        """POD and FAR per size band, pooled over the initializations (bands with
        fewer than NMIN systems are left blank)."""
        df = pd.concat([match_systems(yt, pf, THR, area_row, iou_min=IOU_MIN)
                        for yt, pf in zip(truths, fields)], ignore_index=True)
        summ = summarize_by_scale(df, BANDS).by_scale
        pod, far = summ["POD"].to_numpy(dtype=float).copy(), summ["FAR"].to_numpy(dtype=float).copy()
        b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
        for k in range(len(MIDS)):
            if (df.role.isin(["match", "miss"]) & (b == k)).sum() < NMIN:
                pod[k] = np.nan
            if (df.role.isin(["match", "false_alarm"]) & (b == k)).sum() < NMIN:
                far[k] = np.nan
        return pod, far

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
    fig, (ax_pod, ax_far) = plt.subplots(1, 2, figsize=(W2, W2 * 0.46))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.15, wspace=0.22)
    for ax in (ax_pod, ax_far):
        ax.grid(True, color="0.88", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_xlim(30, 1300)      # plotted bands run from 38 to 950 km; margins on both sides
        log_ticks(ax.xaxis, [30, 50, 100, 200, 500, 1000])
        ax.set_xlabel(r"system scale $L = \sqrt{A}$ (km)")
    for name, (pod, far) in series.items():
        ls = "--" if name == "persistence" else "-"
        ax_pod.plot(MIDS, pod, ls, color=COLW[name], marker="o", ms=3.5, lw=1.5, label=NAMES[name])
        ax_far.plot(MIDS, far, ls, color=COLW[name], marker="o", ms=3.5, lw=1.5)
    ax_pod.set_ylim(0, 1.04)
    ax_far.set_ylim(0, 1.04)
    ax_pod.set_ylabel("Probability of Detection (POD)")
    ax_far.set_ylabel("False-Alarm Ratio (FAR)")
    ax_pod.legend(fontsize=7.5, loc="upper left")
    for ax, letter, text in ((ax_pod, "(a)", "Detection by system size, +12 h"),
                             (ax_far, "(b)", "False alarms by system size, +12 h")):
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
                fontfamily="cmr10")
        ax.text(0.10, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
    fig.savefig(OUT / "fig7b_pod.pdf")
    print("saved fig7b_pod.pdf")


if __name__ == "__main__":
    main()
