"""Random null baseline for the GraphCast multi-lead τ(L) analysis.

For each GraphCast lead (6, 12, 18, 24h) we load the same ERA5 truth frames
used in the GraphCast eval, then pair each truth frame against an ERA5 frame
shifted by ~half the month (T//2 positions).  This removes all weather-scale
correlation while keeping the August 2020 spatial rain distribution intact.

Outputs (one per lead, matching the GraphCast naming convention):
  docs/scale_summary_random_era5_2020-08_lead{N}h_thr2p4.csv
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
MONTH = "2020-08"
LEADS = [6, 12, 18, 24]


def load_truth_for_lead(lead_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (truth, valid_times) for the frames used at this lead."""
    pred_path = CACHE / f"graphcast_tp6hr_{MONTH}_lead{lead_h}h_tropics.nc"
    era5_path = CACHE / f"era5_tp6hr_{MONTH}_tropics.nc"

    mp = xr.open_dataset(pred_path)
    er = xr.open_dataset(era5_path)

    pred  = mp["total_precipitation_6hr"]
    truth = er["total_precipitation_6hr"]

    common = np.intersect1d(pred.valid_time.values, truth.valid_time.values)
    truth  = truth.sel(valid_time=common)

    if truth.lon.values[0] < 0:
        truth = truth.assign_coords(lon=(truth.lon % 360)).sortby("lon")
    if truth.lat.values[0] > truth.lat.values[-1]:
        truth = truth.sortby("lat")

    return truth.values.astype("float32"), common


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=2.4,
                   help="Precip threshold mm/6h (default 2.4, matching GraphCast eval).")
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=600.0)
    p.add_argument("--scale-bins", type=int, default=10)
    args = p.parse_args()

    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)

    for lead in LEADS:
        print(f"\n=== lead {lead}h ===", flush=True)
        truth, valid_times = load_truth_for_lead(lead)
        T = truth.shape[0]

        # Shift by half the number of available frames (~15 days at 6-hr cadence).
        # This guarantees the paired frames are from a different weather regime
        # while drawing from the same August 2020 distribution.
        offset = T // 2
        rand_idx = np.arange(T)
        rand_idx = (rand_idx + offset) % T
        rand_pred = truth[rand_idx]  # shuffled ERA5 frames as prediction

        print(f"  T={T}  offset={offset} frames ({offset * 6}h shift)", flush=True)

        hours = pd.to_datetime(valid_times).hour
        frame_day_idx = (hours // 6).to_numpy()

        df = system_table(
            truth, rand_pred,
            threshold=args.threshold, iou_min=args.iou_min,
            frame_day_idx=frame_day_idx, frames_per_day=4,
        )
        df.insert(0, "lead_h", float(lead))
        df.insert(0, "predictor_set", "random_era5")

        matched = (df["role"] == "match").sum()
        missed  = (df["role"] == "miss").sum()
        fa      = (df["role"] == "false_alarm").sum()
        print(f"  match={matched}  miss={missed}  fa={fa}", flush=True)

        summ = summarize_by_scale(df, edges).by_scale
        summ.insert(0, "lead_h", float(lead))
        summ.insert(0, "predictor_set", "random_era5")

        out = REPO / f"docs/scale_summary_random_era5_{MONTH}_lead{lead}h_thr2p4.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out, index=False)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
