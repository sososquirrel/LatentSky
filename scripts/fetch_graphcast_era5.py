"""Pull GraphCast 6h-precip predictions + ERA5 6h-precip truth from
WeatherBench2, cropped to the tropics, for a single month. Writes one
NetCDF per dataset under /glade/derecho/scratch/<user>/graphcast_eval/.

GraphCast forecasts are initialized every 12 h; we keep only the lead=6h
step so each output is a single forecast at the IC+06h valid time.
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import xarray as xr
import fsspec


REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "graphcast_eval"

GRAPHCAST_URL = ("gs://weatherbench2/datasets/graphcast/2020/"
                 "date_range_2019-11-16_2021-02-01_12_hours.zarr")
HRES_URL = ("gs://weatherbench2/datasets/hres/"
            "2016-2022-0012-1440x721.zarr")
ERA5_URL = ("gs://weatherbench2/datasets/era5/"
            "1959-2022-6h-1440x721.zarr")

MODEL_URLS = {"graphcast": GRAPHCAST_URL, "hres": HRES_URL}

LAT_LO, LAT_HI = -40.0, 40.0
LEAD_HOURS = 6


def crop_lat(da: xr.DataArray) -> xr.DataArray:
    """Crop to tropical band, ordering ascending in lat so it matches our
    SAM-derived grid convention (-40..40, equator near the middle).

    Handles both coord-naming conventions: GraphCast uses lat/lon,
    WeatherBench2 ERA5 uses latitude/longitude — rename to lat/lon."""
    rename = {}
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    return da.sel(lat=slice(LAT_LO, LAT_HI))


def _sel_lead(da: xr.DataArray) -> xr.DataArray:
    """Select the LEAD_HOURS step, handling both int64-hours (GraphCast) and
    timedelta64 (HRES) encodings of prediction_timedelta."""
    td = da["prediction_timedelta"]
    if td.dtype.kind == "m":              # timedelta64
        return da.sel(prediction_timedelta=np.timedelta64(LEAD_HOURS, "h"))
    return da.sel(prediction_timedelta=LEAD_HOURS)   # int64 hours


def fetch_model(model: str, month: str, out_path: pathlib.Path) -> None:
    """model in {'graphcast','hres'}. Writes (valid_time, lat, lon) precip in mm.

    Both store total_precipitation_6hr as (time, prediction_timedelta, lat, lon)
    in metres; we keep the lead=6h step and set a valid_time = init + 6h axis.
    Parallel GCS reads via dask (each chunk is one 2-D field)."""
    ds = xr.open_zarr(fsspec.get_mapper(MODEL_URLS[model], anon=True),
                       consolidated=True, chunks={"time": 4})
    da = ds["total_precipitation_6hr"]
    da = _sel_lead(da)
    da = da.sel(time=slice(f"{month}-01", f"{month}-31T23:59"))
    da = crop_lat(da)
    print(f"{model} lazy: {dict(da.sizes)}", flush=True)
    print("loading from GCS (parallel)...", flush=True)
    da = da.compute()
    da = da * 1000.0   # m -> mm
    lead_td = np.timedelta64(LEAD_HOURS, "h")
    da = da.assign_coords(valid_time=("time", da["time"].values + lead_td))
    da = da.rename({"time": "init_time"}).swap_dims({"init_time": "valid_time"})
    da.attrs["units"] = "mm/6h"
    print(f"{model} loaded: {dict(da.sizes)}  range {da.valid_time.values[[0,-1]]}",
          flush=True)
    da.to_netcdf(out_path)
    print(f"wrote {out_path}", flush=True)


def fetch_era5(month: str, out_path: pathlib.Path) -> None:
    """Match GraphCast's `total_precipitation_6hr` at the same valid times."""
    ds = xr.open_zarr(fsspec.get_mapper(ERA5_URL, anon=True),
                       consolidated=True, chunks={"time": 4})
    da = ds["total_precipitation_6hr"]  # m, (time, lat, lon)
    da = da.sel(time=slice(f"{month}-01", f"{month}-31T23:59"))
    da = crop_lat(da)
    print(f"era5 lazy: {dict(da.sizes)}", flush=True)
    print("loading from GCS (parallel)...", flush=True)
    da = da.compute()
    da = da * 1000.0
    da = da.rename({"time": "valid_time"})
    da.attrs["units"] = "mm/6h"
    print(f"era5 loaded: {dict(da.sizes)}  range {da.valid_time.values[[0,-1]]}",
          flush=True)
    da.to_netcdf(out_path)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--month", default="2020-08")
    p.add_argument("--models", default="graphcast",
                   help="Comma/colon-separated subset of {graphcast,hres}.")
    p.add_argument("--out-dir", default=str(CACHE))
    args = p.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.replace(":", ",").split(",") if m.strip()]
    for model in models:
        mp = out_dir / f"{model}_tp6hr_{args.month}_tropics.nc"
        if mp.exists():
            print(f"skip (cached): {mp}", flush=True)
        else:
            fetch_model(model, args.month, mp)

    er_path = out_dir / f"era5_tp6hr_{args.month}_tropics.nc"
    if er_path.exists():
        print(f"skip (cached): {er_path}", flush=True)
    else:
        fetch_era5(args.month, er_path)


if __name__ == "__main__":
    main()
