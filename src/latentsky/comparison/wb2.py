"""Helpers for the WeatherBench 2 fields (GraphCast, FuXi, GenCast, ERA5)."""
from __future__ import annotations

import numpy as np


def to_mm(a):
    """Precipitation in mm. WB2 stores most accumulations in metres (ERA5
    convention); a tropical-belt maximum below 2 can only be metres."""
    return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a


def crop(da, lat_max=40.0):
    """Rename to lat/lon, sort latitude ascending, and keep lat_max S - lat_max N."""
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.coords}
    if ren:
        da = da.rename(ren)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    return da.sel(lat=slice(-lat_max, lat_max))
