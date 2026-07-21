"""Fetch GraphCast + FuXi 6-hr precip at multiple lead times (6, 12, 18, 24h).

Saves separate NetCDFs per model per lead:
  $SCRATCH/graphcast_eval/{model}_tp6hr_{month}_lead{lead}h_tropics.nc

ERA5 truth is also fetched at all corresponding valid times.
"""
from __future__ import annotations
import os
import pathlib
import numpy as np
import xarray as xr
import fsspec

CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "graphcast_eval"
MONTH = "2020-08"
LAT_SLICE = slice(-40, 40)
LEADS_H = [6, 12, 18, 24]

SOURCES = {
    "graphcast": ("gs://weatherbench2/datasets/graphcast/2020/"
                  "date_range_2019-11-16_2021-02-01_12_hours.zarr",
                  "total_precipitation_6hr", 1000.0),  # m → mm
    "fuxi":      ("gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
                  "total_precipitation_6hr", 1.0),     # already mm
}
ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"


def _sel_td(ds: xr.Dataset, var: str, td_hours: int) -> xr.DataArray:
    da = ds[var]
    td = ds["prediction_timedelta"]
    if np.issubdtype(td.dtype, np.timedelta64):
        return da.sel(prediction_timedelta=np.timedelta64(td_hours, "h"))
    return da.sel(prediction_timedelta=td_hours)


def _fix_coords(da: xr.DataArray) -> xr.DataArray:
    rename = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")]
              if k in da.coords}
    return da.rename(rename) if rename else da


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    all_valid_times = set()

    for model, (url, var, scale) in SOURCES.items():
        print(f"\n=== {model} ===", flush=True)
        ds = xr.open_zarr(fsspec.get_mapper(url, anon=True), consolidated=True)

        for lead in LEADS_H:
            out = CACHE / f"{model}_tp6hr_{MONTH}_lead{lead}h_tropics.nc"
            if out.exists():
                print(f"  lead={lead}h already cached", flush=True)
                vt = xr.open_dataset(out)["valid_time"].values
                all_valid_times.update(vt.tolist())
                continue

            print(f"  fetching lead={lead}h ...", flush=True)
            da = _sel_td(ds, var, lead)
            da = da.sel(time=da.time.dt.strftime("%Y-%m") == MONTH)
            da = da.sel(latitude=LAT_SLICE) if "latitude" in da.coords \
                else da.sel(lat=LAT_SLICE)
            da = _fix_coords(da)
            lead_td = np.timedelta64(lead, "h")
            valid_times = da.time.values + lead_td
            da = da.assign_coords(
                valid_time=("time", valid_times)
            ).swap_dims({"time": "valid_time"})
            da = da.compute() * scale
            all_valid_times.update(valid_times.tolist())

            ds_out = da.to_dataset(name="total_precipitation_6hr")
            ds_out["total_precipitation_6hr"].attrs["units"] = "mm/6h"
            ds_out.to_netcdf(out)
            print(f"  wrote {out}  shape={da.shape}", flush=True)

    # ERA5 truth at all valid times
    out_era5 = CACHE / f"era5_tp6hr_{MONTH}_multilead_tropics.nc"
    if out_era5.exists():
        print(f"\nERA5 already cached: {out_era5}", flush=True)
        return

    print("\n=== ERA5 truth ===", flush=True)
    target_times = np.array(sorted(all_valid_times), dtype="datetime64[ns]")
    ds_era5 = xr.open_zarr(fsspec.get_mapper(ERA5_URL, anon=True), consolidated=True)
    da_era5 = ds_era5["total_precipitation_6hr"].sel(time=target_times)
    da_era5 = _fix_coords(da_era5).rename({"time": "valid_time"})
    # ERA5 zarr stores lat descending (90 -> -40); sortby first so the
    # ascending LAT_SLICE bounds actually match (a slice on a descending
    # coord silently returns 0 elements instead of raising).
    if da_era5.lat.values[0] > da_era5.lat.values[-1]:
        da_era5 = da_era5.sortby("lat")
    da_era5 = da_era5.sel(lat=LAT_SLICE)
    da_era5 = da_era5.compute() * 1000.0  # m → mm
    ds_out = da_era5.to_dataset(name="total_precipitation_6hr")
    ds_out["total_precipitation_6hr"].attrs["units"] = "mm/6h"
    ds_out.to_netcdf(out_era5)
    print(f"wrote {out_era5}  shape={da_era5.shape}", flush=True)


if __name__ == "__main__":
    main()
