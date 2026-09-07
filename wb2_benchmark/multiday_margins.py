"""Is the FGN-vs-GraphCast margin gap real or one-day noise? Same +6 h
object evaluation over FIVE August days per model (their respective years),
margins over each model's own persistence null, mid-scale aggregate
(100-400 km) with day-to-day spread.

Models: graphcast + fuxi (2020-08-03/09/15/21/27), fgn mean + fgn member 0
(2022, same dates). Threshold 2.4 mm/6h, IoU >= 0.1. Fetches cached per
model-day to $SCRATCH/wb2_eval. Prints a table and writes
multiday_margins.csv."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from latentsky.metrics import cell_area_km2, match_systems

CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = pathlib.Path("/glade/u/home/sabramian/LatentSky/wb2_benchmark")
THR, IOU_MIN = 2.4, 0.1
DAYS = ["08-03", "08-09", "08-15", "08-21", "08-27"]
ERA5_2020 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
ERA5_2022 = ("gs://weatherbench2/datasets/era5/"
             "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
CFG = {
    "graphcast": dict(url="gs://weatherbench2/datasets/graphcast/2020/"
                          "date_range_2019-11-16_2021-02-01_12_hours.zarr",
                      year="2020", sample=None),
    "fuxi": dict(url="gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
                 year="2020", sample=None),
    "fgn-mean": dict(url="gs://weatherbench2/datasets/fgn/2022-1440x721_mean.zarr",
                     year="2022", sample=None),
    "fgn-member": dict(url="gs://weatherbench2/datasets/fgn/2022-1440x721.zarr",
                       year="2022", sample=0),
}


def crop(da):
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
           if k in da.coords}
    if ren:
        da = da.rename(ren)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    return da.sel(lat=slice(-40, 40))


def to_mm(a):
    return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a


def fetch(model, day):
    cfg = CFG[model]
    date = f"{cfg['year']}-{day}"
    fn = CACHE / f"wb2md_{model}_{date}.nc"
    if fn.exists():
        return xr.open_dataset(fn)
    so = {"token": "anon"}
    ds = xr.open_zarr(cfg["url"], storage_options=so, decode_timedelta=True)
    inits = ds.time.sel(time=date).values
    da = ds["total_precipitation_6hr"].sel(
        time=inits, prediction_timedelta=np.timedelta64(6, "h"))
    if cfg["sample"] is not None:
        da = da.isel(sample=cfg["sample"])
    fc = crop(da).load()
    era = xr.open_zarr(ERA5_2022 if cfg["year"] == "2022" else ERA5_2020,
                       storage_options=so, decode_timedelta=True)
    tp = crop(era["total_precipitation_6hr"])
    truth = tp.sel(time=inits + np.timedelta64(6, "h")).load()
    persist = tp.sel(time=inits).load()
    out = xr.Dataset({
        "forecast": (("init", "lat", "lon"), fc.values.astype(np.float32)),
        "truth": (("init", "lat", "lon"), truth.values.astype(np.float32)),
        "persistence": (("init", "lat", "lon"),
                        persist.values.astype(np.float32)),
    }, coords={"init": inits, "lat": fc.lat.values, "lon": fc.lon.values})
    out.to_netcdf(fn)
    print(f"fetched {model} {date} ({len(inits)} inits)", flush=True)
    return out


def midscale_pod(field_list, truth_list, area_row):
    """POD over systems with 100 <= L < 400 km, pooled over the day."""
    nm = nmiss = 0
    for yt, pf in zip(truth_list, field_list):
        df = match_systems(yt, pf, THR, area_row, iou_min=IOU_MIN)
        g = df[(df.scale >= 100) & (df.scale < 400)
               & df.role.isin(["match", "miss"])]
        nm += int((g.role == "match").sum())
        nmiss += int((g.role == "miss").sum())
    return nm / max(nm + nmiss, 1), nm + nmiss


def main():
    rows = []
    for model in CFG:
        for day in DAYS:
            ds = fetch(model, day)
            area_row = cell_area_km2(ds.lat.size, ds.lon.size,
                                     lat_extent=(float(ds.lat[0]) - 0.125,
                                                 float(ds.lat[-1]) + 0.125))
            truths = [to_mm(ds.truth.values[i]) for i in range(ds.init.size)]
            fc = [to_mm(ds.forecast.values[i]) for i in range(ds.init.size)]
            pe = [to_mm(ds.persistence.values[i]) for i in range(ds.init.size)]
            pod_m, n = midscale_pod(fc, truths, area_row)
            pod_p, _ = midscale_pod(pe, truths, area_row)
            rows.append(dict(model=model, day=day, pod=pod_m, pod_null=pod_p,
                             margin=pod_m - pod_p, n=n))
            print(f"{model} {day}: POD {pod_m:.3f} null {pod_p:.3f} "
                  f"margin {pod_m - pod_p:+.3f} (n={n})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multiday_margins.csv", index=False)
    print("\nmid-scale (100-400 km) POD margin over own null, 5 August days:")
    for model, g in df.groupby("model"):
        print(f"  {model:11s}: mean {g.margin.mean():+.3f}  "
              f"day-to-day sd {g.margin.std():.3f}  "
              f"range [{g.margin.min():+.3f}, {g.margin.max():+.3f}]")


if __name__ == "__main__":
    main()
