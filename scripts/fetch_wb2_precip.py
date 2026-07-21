"""Pull 24-hr precipitation forecasts + ERA5 24-hr truth from WeatherBench2,
cropped to the tropics, for one model.

Model / month pairs (August of each model's available year):
  graphcast  2020-08   sums 4 × 6-hr leads (6,12,18,24 h) to 24 hr
  fuxi       2020-08   total_precipitation_24hr_from_6hr at lead = 24 h
  gencast    2020-08   ensemble-mean zarr, total_precipitation_24hr at lead = 24 h
  fgn        2022-08   ensemble-mean zarr, total_precipitation_24hr at lead = 24 h

Output (written to --out-dir):
  {model}_tp24hr_{month}_tropics.nc   (valid_time, lat, lon)  mm/24 h
  era5_tp24hr_{month}_tropics.nc      (valid_time, lat, lon)  mm/24 h
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import xarray as xr
import fsspec

CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "wb2_eval"
LAT_LO, LAT_HI = -40.0, 40.0
ERA5_URL   = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"       # covers 1959–2021
HREST0_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"  # covers 2016–2023

MODEL_CFG: dict[str, dict] = {
    "graphcast": dict(
        url="gs://weatherbench2/datasets/graphcast/2020/"
            "date_range_2019-11-16_2021-02-01_12_hours.zarr",
        month="2020-08",
        var="total_precipitation_6hr",
        mode="sum4",   # sum int64 leads 6, 12, 18, 24 h
        mm_scale=1000.0,  # stored in m
    ),
    "fuxi": dict(
        url="gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
        month="2020-08",
        var="total_precipitation_24hr_from_6hr",
        mode="lead24",
        mm_scale=1.0,   # already in mm (derived variable, not raw ERA5 convention)
    ),
    "gencast": dict(
        url="gs://weatherbench2/datasets/gencast/2020-1440x721_mean.zarr",
        month="2020-08",
        var="total_precipitation_24hr",
        mode="lead24",
        mm_scale=1000.0,  # stored in m (ERA5 convention)
    ),
    "fgn": dict(
        url="gs://weatherbench2/datasets/fgn/2022-1440x721_mean.zarr",
        month="2022-08",
        var="total_precipitation_24hr",
        mode="lead24",
        mm_scale=1000.0,  # stored in m (ERA5 convention)
    ),
}


def _crop_lat(da: xr.DataArray) -> xr.DataArray:
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


def _strip_bad_attrs(da: xr.DataArray) -> xr.DataArray:
    """Drop coordinate/variable attrs with types netCDF4 can't write (e.g. bool)."""
    ok = (bool, np.bool_)
    da = da.copy()
    da.attrs = {k: v for k, v in da.attrs.items() if not isinstance(v, ok)}
    new_coords = {
        name: coord.assign_attrs(
            {k: v for k, v in coord.attrs.items() if not isinstance(v, ok)}
        )
        for name, coord in da.coords.items()
    }
    return da.assign_coords(new_coords)


def fetch_model(model: str, out_dir: pathlib.Path) -> pathlib.Path:
    cfg = MODEL_CFG[model]
    month = cfg["month"]
    out = out_dir / f"{model}_tp24hr_{month}_tropics.nc"
    if out.exists():
        print(f"skip (cached): {out}", flush=True)
        return out

    ds = xr.open_zarr(fsspec.get_mapper(cfg["url"], anon=True),
                      consolidated=True, chunks={"time": 4})
    da = ds[cfg["var"]]

    # prediction_timedelta may be int64 hours (graphcast/fuxi/gencast) or
    # timedelta64 (fgn has dtype:timedelta64[ns] in its zarr attrs)
    td_coord = ds["prediction_timedelta"]
    if np.issubdtype(td_coord.dtype, np.timedelta64):
        leads_sum = [np.timedelta64(h, "h") for h in [6, 12, 18, 24]]
        lead24 = np.timedelta64(24, "h")
    else:
        leads_sum = [6, 12, 18, 24]
        lead24 = 24

    if cfg["mode"] == "sum4":
        da = da.sel(prediction_timedelta=leads_sum).sum("prediction_timedelta")
    else:
        da = da.sel(prediction_timedelta=lead24)

    # Keep only August inits
    import pandas as pd
    init_times = pd.DatetimeIndex(da["time"].values)
    year = int(month[:4])
    aug_mask = np.asarray((init_times.month == 8) & (init_times.year == year))
    da = da.isel(time=aug_mask)
    valid_time_vals = da["time"].values + np.timedelta64(24, "h")

    da = _crop_lat(da)
    print(f"{model} lazy: {dict(da.sizes)}", flush=True)
    da = da.compute()
    da = da * cfg["mm_scale"]
    da.name = "total_precipitation_24hr"
    da = da.assign_coords(valid_time=("time", valid_time_vals))
    da = da.rename({"time": "init_time"}).swap_dims({"init_time": "valid_time"})
    # Reconstruct with plain coords to drop any JAX/zarr-specific attrs
    # that netCDF4 can't write (e.g. _jax_coord bool).
    da_out = xr.DataArray(
        data=da.values,
        coords={
            "valid_time": valid_time_vals,
            "lat": da.lat.values,
            "lon": da.lon.values,
        },
        dims=["valid_time", "lat", "lon"],
        name="total_precipitation_24hr",
        attrs={"units": "mm/24h"},
    )
    print(f"{model}: {dict(da_out.sizes)}  "
          f"{da_out.valid_time.values[[0, -1]]}", flush=True)
    da_out.to_netcdf(out)
    print(f"wrote {out}", flush=True)
    return out


def fetch_era5(month: str, valid_times: np.ndarray, out_dir: pathlib.Path) -> pathlib.Path:
    """24-hr ERA5 truth matched to model valid times.

    The 24-hr accumulation ending at valid_time V is the rolling sum of the
    four consecutive 6-hr ERA5 values at V-18h, V-12h, V-6h, V.
    """
    out = out_dir / f"era5_tp24hr_{month}_tropics.nc"
    if out.exists():
        print(f"skip (cached): {out}", flush=True)
        return out

    year = month[:4]
    # ERA5 only covers through 2021; use HRES-T0 analysis for 2022+
    truth_url = ERA5_URL if int(year) <= 2021 else HREST0_URL
    ds = xr.open_zarr(fsspec.get_mapper(truth_url, anon=True),
                      consolidated=True, chunks={"time": 48})
    da = ds["total_precipitation_6hr"]
    # Load July–Sep to cover the rolling-sum buffer at the month boundary
    da = da.sel(time=slice(f"{year}-07-31", f"{year}-09-01"))
    da = _crop_lat(da)
    print(f"era5 lazy: {dict(da.sizes)}", flush=True)
    da = da.compute()

    # Rolling sum of 4 steps = 24-hr accumulation ending at each time step
    da_24 = da.rolling(time=4, min_periods=4).sum()
    da_24 = da_24.rename({"time": "valid_time"})
    da_24 = da_24.sel(valid_time=valid_times)
    da_24 = da_24 * 1000.0   # m → mm
    da_24.name = "total_precipitation_24hr"
    da_24.attrs["units"] = "mm/24h"
    print(f"era5: {dict(da_24.sizes)}  "
          f"{da_24.valid_time.values[[0, -1]]}", flush=True)
    da_24.to_netcdf(out)
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_CFG))
    p.add_argument("--out-dir", default=str(CACHE))
    args = p.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = fetch_model(args.model, out_dir)

    ds_m = xr.open_dataset(model_path)
    valid_times = ds_m.valid_time.values
    ds_m.close()

    fetch_era5(MODEL_CFG[args.model]["month"], valid_times, out_dir)


if __name__ == "__main__":
    main()
