"""ERA5-domain persistence baseline: predict tp(t+lead) = tp(t) (freeze the
initial condition forward), scored against actual ERA5 truth with the same
object-decomposition pipeline used for GraphCast/FuXi.

This is deliberately on the SAME truth source, grid, threshold, scale bins,
and leads as scripts/eval_graphcast_scale.py, so its tau(L) is directly
comparable to GraphCast's and FuXi's -- unlike docs/scale_summary_baselines.csv,
which is a persistence baseline computed on LatentSky's own held-out DYAMOND
pieces (different domain/threshold/bins/leads entirely).

Reads:
  cache/era5_tp6hr_2020-08_tropics.nc   (single ERA5 file, 6h cadence, whole month)

Writes:
  docs/scale_systems_era5persistence_<month>_lead<H>h_thr<T>.csv.gz
  docs/scale_summary_era5persistence_<month>_lead<H>h_thr<T>.csv
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
CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "graphcast_eval"


def load_shifted(month: str, lead_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (truth, pred, valid_times) each shaped (T, H, W), where
    pred[i] = truth at (valid_time[i] - lead_h) -- the persisted IC."""
    era5_path = CACHE / f"era5_tp6hr_{month}_tropics.nc"
    ds = xr.open_dataset(era5_path)
    da = ds["total_precipitation_6hr"]
    if da.lon.values[0] < 0:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")

    dt_h = int((da.valid_time.values[1] - da.valid_time.values[0])
               .astype("timedelta64[h]").astype(int))
    shift = lead_h // dt_h
    assert shift * dt_h == lead_h, f"lead_h={lead_h} not a multiple of cadence={dt_h}h"
    assert shift > 0

    truth = da.isel(valid_time=slice(shift, None))
    pred = da.isel(valid_time=slice(0, -shift))
    valid_times = truth.valid_time.values
    return (truth.values.astype("float32"), pred.values.astype("float32"), valid_times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--month", default="2020-08")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=600.0)
    p.add_argument("--scale-bins", type=int, default=10)
    p.add_argument("--label", default="era5_persistence")
    p.add_argument("--out-summary", default=None)
    p.add_argument("--out-systems", default=None)
    p.add_argument("--lead-override", type=int, default=6,
                   help="Lead time in hours for the persisted initial condition.")
    args = p.parse_args()

    label = args.label
    truth, pred, valid_times = load_shifted(args.month, args.lead_override)
    print(f"loaded {truth.shape[0]} frames, grid {truth.shape[1:]}", flush=True)
    print(f"valid_times: {valid_times[0]} .. {valid_times[-1]}", flush=True)

    hours = pd.to_datetime(valid_times).hour
    frame_day_idx = (hours // 6).to_numpy()
    frames_per_day = 4

    df = system_table(
        truth, pred,
        threshold=args.threshold, iou_min=args.iou_min,
        frame_day_idx=frame_day_idx, frames_per_day=frames_per_day,
    )
    name = f"{label}_{args.month}_lead{args.lead_override}h"
    df.insert(0, "name", name)
    df.insert(0, "lead_h", float(args.lead_override))
    df.insert(0, "predictor_set", label)
    print(f"systems: {len(df)}  "
          f"({(df['role']=='match').sum()} matched, "
          f"{(df['role']=='miss').sum()} missed, "
          f"{(df['role']=='false_alarm').sum()} fa)", flush=True)

    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)
    summ = summarize_by_scale(df, edges).by_scale
    summ.insert(0, "name", name)
    summ.insert(0, "lead_h", float(args.lead_override))
    summ.insert(0, "predictor_set", label)

    out_systems = pathlib.Path(args.out_systems or
        REPO / f"docs/scale_systems_{label}_{args.month}.csv.gz")
    out_summary = pathlib.Path(args.out_summary or
        REPO / f"docs/scale_summary_{label}_{args.month}.csv")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_summary, index=False)
    df.to_csv(out_systems, index=False, compression="gzip")
    print(f"wrote {out_summary}", flush=True)
    print(f"wrote {out_systems}", flush=True)


if __name__ == "__main__":
    main()
