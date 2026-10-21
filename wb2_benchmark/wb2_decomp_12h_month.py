"""Month-scale version of the +12 h decomposition: August 2020, all 62
initializations (00/12 UTC), GraphCast / FuXi (6h+12h leads summed),
GenCast ensemble mean and one GenCast member (sample 0, per-draw).
Band barycenters (dlnA, dlnrho) precip-weighted; same 3-panel figure,
overwriting fig_wb2_decomp_12h.png. Fetches cached per model to
$SCRATCH/wb2_eval/wb2m12_{name}.nc (~110 MB each). CPU eval ~10 min."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from latentsky.metrics import cell_area_km2, match_systems
from latentsky.comparison.wb2 import crop

CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "wb2_benchmark"
THR, IOU_MIN = 4.8, 0.1
MONTH = "2020-08"
BANDS = [("small < 100 km", 0, 100, 60), ("100–400 km", 100, 400, 200),
         ("large > 400 km", 400, 1e9, 700)]
ERA5 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
SO = {"token": "anon"}
LEAD = np.timedelta64(12, "h")




def cached(name, build):
    fn = CACHE / f"wb2m12_{name}.nc"
    if fn.exists():
        return xr.open_dataset(fn)["field"]
    arr, inits, lat, lon = build()
    xr.Dataset({"field": (("init", "lat", "lon"), arr.astype(np.float32))},
               coords={"init": inits, "lat": lat, "lon": lon}).to_netcdf(fn)
    print(f"fetched {name}", flush=True)
    return xr.open_dataset(fn)["field"]


def main():
    g = xr.open_zarr("gs://weatherbench2/datasets/graphcast/2020/"
                     "date_range_2019-11-16_2021-02-01_12_hours.zarr",
                     storage_options=SO, decode_timedelta=True)
    inits = g.time.sel(time=MONTH).values
    print(f"{len(inits)} inits in {MONTH}", flush=True)

    def sum612(url, var="total_precipitation_6hr"):
        def build():
            ds = xr.open_zarr(url, storage_options=SO, decode_timedelta=True)
            da = crop(ds[var].sel(
                time=inits,
                prediction_timedelta=[np.timedelta64(6, "h"), LEAD]
            ).sum("prediction_timedelta")).load()
            return da.values, inits, da.lat.values, da.lon.values
        return build

    def lead12(url, var="total_precipitation_12hr", sample=None):
        def build():
            ds = xr.open_zarr(url, storage_options=SO, decode_timedelta=True)
            da = ds[var].sel(time=inits, prediction_timedelta=LEAD)
            if sample is not None:
                da = da.isel(sample=sample)
            da = crop(da).load()
            return da.values, inits, da.lat.values, da.lon.values
        return build

    def truth_build():
        era = xr.open_zarr(ERA5, storage_options=SO, decode_timedelta=True)
        tp = crop(era["total_precipitation_6hr"])
        t1 = tp.sel(time=inits + LEAD).load()
        t2 = tp.sel(time=inits + np.timedelta64(6, "h")).load()
        return (t1.values + t2.values), inits, t1.lat.values, t1.lon.values

    fields = {
        "truth": cached("truth", truth_build),
        "graphcast": cached("graphcast", sum612(
            "gs://weatherbench2/datasets/graphcast/2020/"
            "date_range_2019-11-16_2021-02-01_12_hours.zarr")),
        "fuxi": cached("fuxi", sum612(
            "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr")),
        "gencast mean": cached("gencast_mean", lead12(
            "gs://weatherbench2/datasets/gencast/2020-1440x721_mean.zarr")),
        "gencast member": cached("gencast_member0", lead12(
            "gs://weatherbench2/datasets/gencast/2020-1440x721.zarr", sample=0)),
    }
    area_row = cell_area_km2(321, 1440, lat_extent=(-40.125, 40.125))

    def to_mm(a):
        return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a

    matches = {}
    for name in ("graphcast", "fuxi", "gencast mean", "gencast member"):
        dfs = []
        for i in range(len(inits)):
            yt = to_mm(fields["truth"].values[i])
            pf = to_mm(fields[name].values[i])
            if np.isnan(pf).all():
                continue
            d = match_systems(yt, pf, THR, area_row, iou_min=IOU_MIN)
            dfs.append(d[d.role == "match"])
        matches[name] = pd.concat(dfs, ignore_index=True)
        print(f"{name}: {len(matches[name])} matched pairs", flush=True)

    bary, nm = {}, {}
    for name, m in matches.items():
        nm[name] = len(m)
        for bname, lo, hi, lrep in BANDS:
            g = m[(m.scale >= lo) & (m.scale < hi)]
            w = g["P_t"].to_numpy()
            bary[(name, bname)] = (float((w * g.dlnA).sum() / w.sum()),
                                   float((w * g.dlnrho).sum() / w.sum()))
            print(f"{name:15s} {bname:15s}: ({bary[(name,bname)][0]:+.2f}, "
                  f"{bary[(name,bname)][1]:+.2f}) n={len(g)}")

    vir = plt.get_cmap("viridis")
    from matplotlib.colors import LogNorm
    cn = LogNorm(vmin=40, vmax=1000)
    lim = 0.9
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 4.0), sharex=True, sharey=True)
    panels = [("graphcast", axs[0], f"graphcast  (n = {nm['graphcast']:,})"),
              ("fuxi", axs[1], f"fuxi  (n = {nm['fuxi']:,})"),
              ("gencast", axs[2],
               f"gencast:  mean ■ (n = {nm['gencast mean']:,}) → "
               f"member ◆ (n = {nm['gencast member']:,})")]
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
                ax.plot(mA, mR, "s", ms=9, mfc=col, mec="0.1", mew=1.0, zorder=5)
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
                 "August 2020 (62 initializations), matched-pair band "
                 "barycenters, precip-weighted", fontsize=10, y=1.0)
    fig.savefig(OUT / "fig_wb2_decomp_12h.png", dpi=150, bbox_inches="tight")
    print("saved fig_wb2_decomp_12h.png (month)")


if __name__ == "__main__":
    main()
