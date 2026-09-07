"""Sophie's point 1: do AI weather forecasts carry a SIZE bias? The rho*A
decomposition applied to the WB2 models at +12 h (same day/year/null):
precip-weighted band barycenters (dlnA, dlnrho) for GraphCast, FuXi,
GenCast mean and GenCast single member (per-draw x3). Point 2 read off the
small band. Uses the cached wb2_12h_models nc. Outputs
fig_wb2_decomp_12h.png + printed barycenters."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from latentsky.metrics import cell_area_km2, match_systems

CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "wb2_benchmark"
THR, IOU_MIN = 4.8, 0.1
BANDS = [("small < 100 km", 0, 100, 60), ("100–400 km", 100, 400, 200),
         ("large > 400 km", 400, 1e9, 700)]

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def main():
    ds = xr.open_dataset(CACHE / "wb2_12h_models_2020-08-15.nc")
    area_row = cell_area_km2(ds.lat.size, ds.lon.size,
                             lat_extent=(float(ds.lat[0]) - 0.125,
                                         float(ds.lat[-1]) + 0.125))

    def to_mm(a):
        return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a

    truths = [to_mm(ds.truth.values[i]) for i in range(ds.init.size)]

    def matches(fields):
        dfs = [match_systems(yt, pf, THR, area_row, iou_min=IOU_MIN)
               for yt, pf in zip(truths, fields)]
        d = pd.concat(dfs, ignore_index=True)
        return d[d.role == "match"]

    series = {
        "graphcast": [matches([to_mm(ds.graphcast.values[i])
                               for i in range(ds.init.size)])],
        "fuxi": [matches([to_mm(ds.fuxi.values[i])
                          for i in range(ds.init.size)])],
        "gencast mean": [matches([to_mm(ds["mean"].values[i])
                                  for i in range(ds.init.size)])],
        "gencast member": [matches([to_mm(ds.member.values[i, s])
                                    for i in range(ds.init.size)])
                           for s in range(ds.sample.size)],
    }

    # barycenters per model/band
    bary = {}
    for name, dfs in series.items():
        for bname, lo, hi, lrep in BANDS:
            pts = []
            for m in dfs:
                g = m[(m.scale >= lo) & (m.scale < hi)]
                if len(g) < 5:
                    continue
                w = g["P_t"].to_numpy()
                pts.append(((w * g.dlnA).sum() / w.sum(),
                            (w * g.dlnrho).sum() / w.sum()))
            if pts:
                bary[(name, bname)] = tuple(np.mean(pts, axis=0))
                print(f"{name:15s} {bname:15s}: ({bary[(name,bname)][0]:+.2f}, "
                      f"{bary[(name,bname)][1]:+.2f})")

    vir = plt.get_cmap("viridis")
    from matplotlib.colors import LogNorm
    cn = LogNorm(vmin=40, vmax=1000)
    lim = 0.9
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 4.0), sharex=True, sharey=True)
    nm = {name: int(np.mean([len(m) for m in dfs]))
          for name, dfs in series.items()}
    panels = [("graphcast", axs[0], f"graphcast  (n = {nm['graphcast']})"),
              ("fuxi", axs[1], f"fuxi  (n = {nm['fuxi']})"),
              ("gencast", axs[2],
               f"gencast:  mean ■ (n = {nm['gencast mean']}) → "
               f"member ◆ (n = {nm['gencast member']})")]
    for key, ax, title in panels:
        ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
        xs = np.linspace(-lim, lim, 2)
        ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.0)
        for bname, lo, hi, lrep in BANDS:
            col = vir(cn(lrep))
            if key != "gencast":
                bA, bR = bary[(key, bname)]
                ax.plot(bA, bR, "X", ms=13, mfc=col, mec="0.1", mew=1.1,
                        zorder=5)
            else:
                mA, mR = bary[("gencast mean", bname)]
                sA, sR = bary[("gencast member", bname)]
                ax.annotate("", xy=(sA, sR), xytext=(mA, mR),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=1.8), zorder=4)
                ax.plot(mA, mR, "s", ms=9, mfc=col, mec="0.1", mew=1.0,
                        zorder=5)
                ax.plot(sA, sR, "D", ms=9, mfc=col, mec="#b2182b", mew=1.8,
                        zorder=6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(True, which="major", ls=":", lw=0.5, color="0.85")
        ax.set_axisbelow(True)
        ax.set_xlabel("dlnA  (area error)")
        ax.set_title(title, fontsize=9.5)
    axs[0].set_ylabel(r"dln$\rho$  (intensity error)")
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[
        Line2D([], [], marker="X", ls="none", ms=9, mfc=vir(cn(l)), mec="0.1",
               label=b) for b, _, _, l in BANDS],
        fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("the size/intensity bias of AI weather forecasts — +12 h, "
                 "2020-08-15 (two initializations), matched-pair band "
                 "barycenters, precip-weighted", fontsize=10, y=1.0)
    fig.savefig(OUT / "fig_wb2_decomp_12h.png", dpi=150, bbox_inches="tight")
    print("saved fig_wb2_decomp_12h.png")


if __name__ == "__main__":
    main()
