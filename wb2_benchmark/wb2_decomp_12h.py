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

    STYLE = {"graphcast": ("o", "#1b7837"), "fuxi": ("^", "#762a83"),
             "gencast mean": ("s", "#35978f"), "gencast member": ("D", "#b2182b")}
    vir = plt.get_cmap("viridis")
    from matplotlib.colors import LogNorm
    cn = LogNorm(vmin=40, vmax=1000)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lim = 1.3
    ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
    xs = np.linspace(-lim, lim, 2)
    ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.0)
    for name, dfs in series.items():
        mk, colr = STYLE[name]
        for bname, lo, hi, lrep in BANDS:
            pts = []
            for m in dfs:  # per draw, then average the barycenters
                g = m[(m.scale >= lo) & (m.scale < hi)]
                if len(g) < 5:
                    continue
                w = g["P_t"].to_numpy()
                pts.append(((w * g.dlnA).sum() / w.sum(),
                            (w * g.dlnrho).sum() / w.sum()))
            if not pts:
                continue
            bA, bR = np.mean(pts, axis=0)
            ax.plot(bA, bR, mk, ms=10, mfc=vir(cn(lrep)), mec=colr, mew=1.8,
                    zorder=5)
            print(f"{name:15s} {bname:15s}: ({bA:+.2f}, {bR:+.2f})")
    handles = ([Line2D([], [], marker=STYLE[n][0], ls="none", ms=8, mfc="0.85",
                       mec=STYLE[n][1], mew=1.8, label=n) for n in STYLE]
               + [Line2D([], [], marker="s", ls="none", ms=8,
                         mfc=vir(cn(l)), mec="0.4", label=b)
                  for b, _, _, l in BANDS])
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower left",
              ncol=2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, which="major", ls=":", lw=0.5, color="0.85")
    ax.set_axisbelow(True)
    ax.set_xlabel("dlnA  (area error)")
    ax.set_ylabel(r"dln$\rho$  (intensity error)")
    ax.set_title("the size/intensity bias of AI weather forecasts\n"
                 "(+12 h, 2020-08-15, precip-weighted band barycenters)",
                 fontsize=9.5)
    fig.savefig(OUT / "fig_wb2_decomp_12h.png", dpi=150, bbox_inches="tight")
    print("saved fig_wb2_decomp_12h.png")


if __name__ == "__main__":
    main()
