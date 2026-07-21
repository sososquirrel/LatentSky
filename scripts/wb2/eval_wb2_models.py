"""Scale-aware object decomposition of 24-hr precipitation forecasts against
ERA5 truth, for any WeatherBench2 model fetched by fetch_wb2_precip.py.

Reads:
  cache/{model}_tp24hr_{month}_tropics.nc   (pred,  mm/24h)
  cache/era5_tp24hr_{month}_tropics.nc      (truth, mm/24h)

Writes:
  docs/scale_systems_{model}_{month}_24h.csv.gz
  docs/scale_summary_{model}_{month}_24h.csv
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from latentsky.scale_analysis import (
    system_table, summarize_by_scale, default_scale_edges,
)

from latentsky.registry import REPO_ROOT as REPO
CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "wb2_eval"

# Number of forecast valid times per calendar day (init cadence)
_FRAMES_PER_DAY = {"graphcast": 2, "fuxi": 2, "gencast": 2, "fgn": 4}
_MONTH = {"graphcast": "2020-08", "fuxi": "2020-08", "gencast": "2020-08", "fgn": "2022-08"}


def load_pair(
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    month = _MONTH[model]
    mp = xr.open_dataset(CACHE / f"{model}_tp24hr_{month}_tropics.nc")
    er = xr.open_dataset(CACHE / f"era5_tp24hr_{month}_tropics.nc")
    pred = mp["total_precipitation_24hr"]
    truth = er["total_precipitation_24hr"]

    common = np.intersect1d(pred.valid_time.values, truth.valid_time.values)
    pred = pred.sel(valid_time=common)
    truth = truth.sel(valid_time=common)

    if pred.lon.values[0] < 0:
        pred = pred.assign_coords(lon=(pred.lon % 360)).sortby("lon")
    if truth.lon.values[0] < 0:
        truth = truth.assign_coords(lon=(truth.lon % 360)).sortby("lon")
    if pred.lat.values[0] > pred.lat.values[-1]:
        pred = pred.sortby("lat")
    if truth.lat.values[0] > truth.lat.values[-1]:
        truth = truth.sortby("lat")

    assert pred.shape[1:] == truth.shape[1:], (pred.shape, truth.shape)
    return (
        truth.values.astype("float32"),
        pred.values.astype("float32"),
        common,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(_MONTH))
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Wet-pixel threshold in mm/24h (default 1.0).")
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=600.0)
    p.add_argument("--scale-bins", type=int, default=10)
    p.add_argument("--label", default=None,
                   help="predictor_set label in the output (default: model name).")
    p.add_argument("--out-summary", default=None)
    p.add_argument("--out-systems", default=None)
    args = p.parse_args()

    label = args.label or args.model
    month = _MONTH[args.model]
    truth, pred, valid_times = load_pair(args.model)
    print(f"loaded {truth.shape[0]} frames, grid {truth.shape[1:]}", flush=True)
    print(f"valid_times: {valid_times[0]} .. {valid_times[-1]}", flush=True)

    frames_per_day = _FRAMES_PER_DAY[args.model]
    hours = pd.to_datetime(valid_times).hour
    frame_day_idx = (hours // (24 // frames_per_day)).to_numpy()

    df = system_table(
        truth, pred,
        threshold=args.threshold, iou_min=args.iou_min,
        frame_day_idx=frame_day_idx, frames_per_day=frames_per_day,
    )
    name = f"{args.model}_{month}_24h"
    df.insert(0, "name", name)
    df.insert(0, "lead_h", 24.0)
    df.insert(0, "predictor_set", label)
    print(
        f"systems: {len(df)}  "
        f"matched={(df['role'] == 'match').sum()}  "
        f"miss={(df['role'] == 'miss').sum()}  "
        f"fa={(df['role'] == 'false_alarm').sum()}",
        flush=True,
    )

    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)
    summ = summarize_by_scale(df, edges).by_scale
    summ.insert(0, "name", name)
    summ.insert(0, "lead_h", 24.0)
    summ.insert(0, "predictor_set", label)

    out_systems = pathlib.Path(
        args.out_systems or REPO / f"docs/scale_systems_{args.model}_{month}_24h.csv.gz"
    )
    out_summary = pathlib.Path(
        args.out_summary or REPO / f"docs/scale_summary_{args.model}_{month}_24h.csv"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_summary, index=False)
    df.to_csv(out_systems, index=False, compression="gzip")
    print(f"wrote {out_summary}", flush=True)
    print(f"wrote {out_systems}", flush=True)


if __name__ == "__main__":
    main()
