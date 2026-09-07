"""Four-panel dlnA/dlnrho scatter clouds for the AI weather models at +12 h,
August 2020 month (cached wb2m12 fields): graphcast, fuxi, gencast mean,
gencast member — same conventions as the LatentSky decomposition clouds
(fixed axes +-3.2, scale colors LogNorm(28,1500), s=1.5, alpha=0.25),
band barycenter X markers on top. Saves matches to
matches_12h_month.csv.gz for reuse. CPU ~10 min."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm

from latentsky.metrics import cell_area_km2, match_systems

CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "wb2_benchmark"
THR = 4.8
BANDS = [("small < 100 km", 0, 100, 60), ("100–400 km", 100, 400, 200),
         ("large > 400 km", 400, 1e9, 700)]
NAMES = {"graphcast": "graphcast", "fuxi": "fuxi",
         "gencast mean": "gencast_mean", "gencast member": "gencast_member0"}

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def to_mm(a):
    return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a


def main():
    cachecsv = OUT / "matches_12h_month.csv.gz"
    if cachecsv.exists():
        allm = pd.read_csv(cachecsv)
    else:
        truth = xr.open_dataset(CACHE / "wb2m12_truth.nc")["field"]
        area_row = cell_area_km2(321, 1440, lat_extent=(-40.125, 40.125))
        parts = []
        for label, fn in NAMES.items():
            fld = xr.open_dataset(CACHE / f"wb2m12_{fn}.nc")["field"]
            for i in range(truth.shape[0]):
                yt = to_mm(truth.values[i])
                pf = to_mm(fld.values[i])
                if np.isnan(pf).all():
                    continue
                d = match_systems(yt, pf, THR, area_row, iou_min=0.1)
                d = d[d.role == "match"][["scale", "dlnA", "dlnrho", "P_t"]]
                d.insert(0, "model", label)
                parts.append(d)
            print(label, "matched", flush=True)
        allm = pd.concat(parts, ignore_index=True)
        allm.to_csv(cachecsv, index=False, compression="gzip")

    cnorm = LogNorm(vmin=28.0, vmax=1500.0)
    vir = plt.get_cmap("viridis")
    lim = 3.2
    fig, axs = plt.subplots(1, 4, figsize=(14.4, 3.9), sharex=True, sharey=True)
    for ax, label in zip(axs, NAMES):
        m = allm[allm.model == label]
        ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
        xs = np.linspace(-lim, lim, 2)
        ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.0)
        sc = ax.scatter(m["dlnA"], m["dlnrho"], s=1.5, c=m["scale"],
                        cmap="viridis", norm=cnorm, alpha=0.25, linewidths=0,
                        rasterized=True)
        for bname, lo, hi, lrep in BANDS:
            g = m[(m.scale >= lo) & (m.scale < hi)]
            w = g["P_t"].to_numpy()
            ax.plot(float((w * g.dlnA).sum() / w.sum()),
                    float((w * g.dlnrho).sum() / w.sum()),
                    "X", ms=11, mfc=vir(cnorm(lrep)), mec="0.05", mew=1.1,
                    zorder=6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(True, which="major", ls=":", lw=0.5, color="0.88")
        ax.set_axisbelow(True)
        ax.set_xlabel("dlnA  (area error)")
        ax.set_title(f"{label}  (n = {len(m):,})", fontsize=9.5)
    axs[0].set_ylabel(r"dln$\rho$  (intensity error)")
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.012)
    cb.set_label("system scale L = √A (km)", fontsize=8)
    fig.suptitle("matched-pair decomposition clouds — +12 h, August 2020 "
                 "(62 initializations); X = precip-weighted band barycenters",
                 fontsize=10, y=1.02)
    fig.savefig(OUT / "fig_wb2_scatter_12h.png", dpi=150, bbox_inches="tight")
    print("saved fig_wb2_scatter_12h.png")


if __name__ == "__main__":
    main()
