"""Fetch FuXi 6-hr precipitation from WeatherBench2 for August 2020.

Saves to the same graphcast_eval cache directory so eval_graphcast_scale.py
can read it with --model fuxi.

Output: $SCRATCH/graphcast_eval/fuxi_tp6hr_2020-08_tropics.nc  (mm/6h)
"""
from __future__ import annotations

import os
import pathlib

import fsspec
import numpy as np
import xarray as xr

FUXI_URL  = "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr"
ERA5_URL  = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "graphcast_eval"
MONTH = "2020-08"
LAT_SLICE = slice(-40, 40)
LEAD_H = 6


def _sel_td(ds: xr.Dataset, var: str, td_hours: int) -> xr.DataArray:
    da = ds[var]
    td = ds["prediction_timedelta"]
    if np.issubdtype(td.dtype, np.timedelta64):
        return da.sel(prediction_timedelta=np.timedelta64(td_hours, "h"))
    return da.sel(prediction_timedelta=td_hours)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)

    # --- FuXi 6hr precip (already in mm — mm_scale=1.0) ---
    out_fuxi = CACHE / f"fuxi_tp6hr_{MONTH}_tropics.nc"
    if out_fuxi.exists():
        print(f"already cached: {out_fuxi}", flush=True)
    else:
        print("opening FuXi zarr...", flush=True)
        ds = xr.open_zarr(fsspec.get_mapper(FUXI_URL, anon=True), consolidated=True)
        da = _sel_td(ds, "total_precipitation_6hr", LEAD_H)
        # valid_time = time (init) + lead; select August inits
        da = da.sel(time=da.time.dt.strftime("%Y-%m") == MONTH)
        da = da.sel(latitude=LAT_SLICE) if "latitude" in da.coords else da.sel(lat=LAT_SLICE)
        # rename latitude/longitude → lat/lon
        rename = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in da.coords}
        if rename:
            da = da.rename(rename)
        # valid_time coordinate: init + lead
        lead_td = np.timedelta64(LEAD_H, "h")
        valid_times = da.time.values + lead_td
        da = da.assign_coords(valid_time=("time", valid_times)).swap_dims({"time": "valid_time"})
        print("loading FuXi...", flush=True)
        da = da.compute()
        # FuXi precip is already in mm (unlike other models which are in m)
        ds_out = da.to_dataset(name="total_precipitation_6hr")
        ds_out["total_precipitation_6hr"].attrs["units"] = "mm/6h"
        ds_out.to_netcdf(out_fuxi)
        print(f"wrote {out_fuxi}  shape={da.shape}", flush=True)

    # --- ERA5 6hr truth at same valid times (if not already cached) ---
    out_era5 = CACHE / f"era5_tp6hr_{MONTH}_tropics.nc"
    if out_era5.exists():
        print(f"era5 already cached: {out_era5}", flush=True)
        return
    print("opening ERA5 zarr...", flush=True)
    # Load from FuXi valid times to align
    fuxi_ds = xr.open_dataset(out_fuxi)
    target_times = fuxi_ds["valid_time"].values
    ds_era5 = xr.open_zarr(fsspec.get_mapper(ERA5_URL, anon=True), consolidated=True)
    da_era5 = ds_era5["total_precipitation_6hr"].sel(time=target_times)
    da_era5 = da_era5.sel(latitude=LAT_SLICE) if "latitude" in da_era5.coords \
        else da_era5.sel(lat=LAT_SLICE)
    rename = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")]
              if k in da_era5.coords}
    if rename:
        da_era5 = da_era5.rename(rename)
    da_era5 = da_era5.rename({"time": "valid_time"})
    print("loading ERA5...", flush=True)
    da_era5 = da_era5.compute() * 1000.0  # m → mm
    ds_era5_out = da_era5.to_dataset(name="total_precipitation_6hr")
    ds_era5_out["total_precipitation_6hr"].attrs["units"] = "mm/6h"
    ds_era5_out.to_netcdf(out_era5)
    print(f"wrote {out_era5}  shape={da_era5.shape}", flush=True)


if __name__ == "__main__":
    main()
