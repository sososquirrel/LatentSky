"""Scale-aware object decomposition of GraphCast 6h precipitation against
ERA5 truth, using the same pipeline we use for LatentSky.

Reads:
  cache/graphcast_tp6hr_<month>_tropics.nc   (pred,  mm/6h)
  cache/era5_tp6hr_<month>_tropics.nc        (truth, mm/6h)

Writes:
  docs/scale_systems_graphcast_<month>.csv.gz
  docs/scale_summary_graphcast_<month>.csv

The two NetCDFs share the same (lat, lon, valid_time) grid; we intersect
valid_time so every output frame is a matched (truth, pred) pair.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path("/glade/derecho/scratch") / os.environ["USER"] / "graphcast_eval"


def load_pair(month: str, model: str = "graphcast",
              lead_h: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (truth, pred, valid_times) each shaped (T, H, W)."""
    pred_path = CACHE / f"{model}_tp6hr_{month}_lead{lead_h}h_tropics.nc"
    if not pred_path.exists():
        fallback = CACHE / f"{model}_tp6hr_{month}_tropics.nc"
        if lead_h == 6 and fallback.exists():
            pred_path = fallback
        else:
            raise FileNotFoundError(
                f"no per-lead file for {model} at lead={lead_h}h: {pred_path} "
                "(and no lead=6h fallback either). Run "
                "scripts/fetch_wb2_multlead.py first."
            )
    era5_path = CACHE / f"era5_tp6hr_{month}_multilead_tropics.nc"
    if not era5_path.exists():
        era5_path = CACHE / f"era5_tp6hr_{month}_tropics.nc"
    mp = xr.open_dataset(pred_path)
    er = xr.open_dataset(era5_path)
    pred = mp["total_precipitation_6hr"]
    truth = er["total_precipitation_6hr"]
    # Align on valid_time.
    common = np.intersect1d(pred.valid_time.values, truth.valid_time.values)
    pred = pred.sel(valid_time=common)
    truth = truth.sel(valid_time=common)
    # Roll lon if pred is on (-180..180) and truth is on (0..360) or vice
    # versa; both should be on 0..360 ascending.
    if pred.lon.values[0] < 0:
        pred = pred.assign_coords(lon=(pred.lon % 360)).sortby("lon")
    if truth.lon.values[0] < 0:
        truth = truth.assign_coords(lon=(truth.lon % 360)).sortby("lon")
    if pred.lat.values[0] > pred.lat.values[-1]:
        pred = pred.sortby("lat")
    if truth.lat.values[0] > truth.lat.values[-1]:
        truth = truth.sortby("lat")
    # Sanity check on shape match.
    assert pred.shape[1:] == truth.shape[1:], (pred.shape, truth.shape)
    return truth.values.astype("float32"), pred.values.astype("float32"), common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--month", default="2020-08")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Precip threshold (mm/6h). Lower than the 15-min "
                        "instantaneous threshold of 0.1 mm we use for "
                        "LatentSky — over 6 h, 0.1 mm/6h is roughly the "
                        "noise floor of a wet pixel.")
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=600.0)
    p.add_argument("--scale-bins", type=int, default=10)
    p.add_argument("--model", default="graphcast", choices=["graphcast", "hres", "fuxi"])
    p.add_argument("--label", default=None,
                   help="predictor_set label in the output (default: model name).")
    p.add_argument("--out-summary", default=None)
    p.add_argument("--out-systems", default=None)
    p.add_argument("--lead-override", type=int, default=6,
                   help="Lead time in hours for multi-lead fetched files (default: 6).")
    args = p.parse_args()

    label = args.label or args.model
    truth, pred, valid_times = load_pair(args.month, args.model, lead_h=args.lead_override)
    print(f"loaded {truth.shape[0]} frames, grid {truth.shape[1:]}", flush=True)
    print(f"valid_times: {valid_times[0]} .. {valid_times[-1]}", flush=True)

    # The system_table API expects (T, 1, H, W) or (T, H, W).
    # Build a synthetic frame_day_idx so lst_hour is computed correctly:
    # each valid_time has its UTC hour, and frames_per_day=4 (since 6h step).
    hours = pd.to_datetime(valid_times).hour
    frame_day_idx = (hours // 6).to_numpy()
    frames_per_day = 4

    df = system_table(
        truth, pred,
        threshold=args.threshold, iou_min=args.iou_min,
        frame_day_idx=frame_day_idx, frames_per_day=frames_per_day,
    )
    name = f"{args.model}_{args.month}_6h"
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
        REPO / f"docs/scale_systems_{args.model}_{args.month}.csv.gz")
    out_summary = pathlib.Path(args.out_summary or
        REPO / f"docs/scale_summary_{args.model}_{args.month}.csv")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_summary, index=False)
    df.to_csv(out_systems, index=False, compression="gzip")
    print(f"wrote {out_summary}", flush=True)
    print(f"wrote {out_systems}", flush=True)


if __name__ == "__main__":
    main()
