"""Persistence, climatological, and random baseline scale summaries.

For each lead time, computes scale-binned POD/FAR using:

  persistence  — predicts Y[t+delta] = Y[t-1]  (freeze rain from one step before)
  climatology  — predicts Y[t+delta] = mean_Y   (August-mean over test pieces)
  random       — predicts Y[t+delta] = Y_other[t-1]  (rain from a different piece,
                 zero temporal correlation with truth — null model for chance overlap)

Uses the same system_table + summarize_by_scale pipeline as LatentSky, with
identical thresholds and piece/delta conventions so results drop straight
into docs/scale_summary_v3.csv comparisons.

Output: docs/scale_summary_baselines.csv
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from latentsky.scale_analysis import (
    system_table,
    summarize_by_scale,
    default_scale_edges,
)

PIECES_DIR = pathlib.Path(
    "/glade/derecho/scratch/tzhou/splitted_1440x720_9vars_40pieces_v2"
)
REPO = pathlib.Path(__file__).resolve().parents[1]

TEST_PIECES = [2, 12, 22, 32]

# Match delta_index used by each trained run (delta = 1 + 4*lead_h).
LEADS = {
    0.0: 1,
    1.0: 5,
    2.0: 9,
    3.0: 13,
    6.0: 25,
}


def load_rain(pid: int) -> np.ndarray:
    """Load rain piece as float32 array of shape (96, H, W)."""
    path = PIECES_DIR / f"rain_total_piece{pid:02d}.npy"
    return np.load(path, mmap_mode="r").astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=3000.0)
    p.add_argument("--scale-bins", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)
    thr = args.threshold
    iou = args.iou_min

    # --- build climatological mean over all test-piece frames ---
    print("computing climatological mean ...", flush=True)
    clim_sum = None
    clim_n = 0
    for pid in TEST_PIECES:
        Y = load_rain(pid)          # (96, H, W)
        clim_sum = Y.sum(axis=0) if clim_sum is None else clim_sum + Y.sum(axis=0)
        clim_n += Y.shape[0]
    clim_mean = (clim_sum / clim_n)   # (H, W) — mean rain per 15-min frame

    summary_rows = []

    for lead_h, delta in LEADS.items():
        print(f"\nlead {lead_h}h  (delta={delta}) ...", flush=True)

        sys_pers, sys_clim, sys_rand = [], [], []

        # Pre-load the "other" piece for each test piece (offset by 2 in the list
        # so the rain source is ~20 days away — fully decorrelated from truth).
        rand_pieces = {
            pid: load_rain(TEST_PIECES[(i + 2) % len(TEST_PIECES)])
            for i, pid in enumerate(TEST_PIECES)
        }

        for pid in TEST_PIECES:
            Y      = load_rain(pid)          # (96, H, W)
            Y_rand = rand_pieces[pid]        # rain from a different piece

            # t-1 persistence: at forecast time t, latest rain is Y[t-1].
            # pair pred=Y[t-1] with truth=Y[t+delta], t starting at 1.
            T = Y.shape[0] - delta - 1  # t = 1 .. 96-delta-1
            if T <= 0:
                continue

            truth = np.ascontiguousarray(Y[delta + 1: delta + 1 + T])  # Y[t+delta]
            pers  = np.ascontiguousarray(Y[0:T])                        # Y[t-1]
            clim  = np.broadcast_to(clim_mean[None], truth.shape).copy()
            # random null: rain from a decorrelated piece at the same local index
            rand  = np.ascontiguousarray(Y_rand[0:T])                   # Y_other[t-1]

            df_p = system_table(truth, pers, threshold=thr, iou_min=iou)
            df_c = system_table(truth, clim, threshold=thr, iou_min=iou)
            df_r = system_table(truth, rand, threshold=thr, iou_min=iou)

            if not df_p.empty:
                sys_pers.append(df_p)
            if not df_c.empty:
                sys_clim.append(df_c)
            if not df_r.empty:
                sys_rand.append(df_r)

            print(f"  piece {pid}: T={T}  pers={len(df_p)}  "
                  f"clim={len(df_c)}  rand={len(df_r)}", flush=True)

        for label, frames in [("persistence", sys_pers), ("climatology", sys_clim),
                               ("random", sys_rand)]:
            if not frames:
                continue
            all_df = pd.concat(frames, ignore_index=True)
            summ = summarize_by_scale(all_df, edges).by_scale
            summ.insert(0, "lead_h", lead_h)
            summ.insert(0, "predictor_set", label)
            summary_rows.append(summ)
            matched = (all_df["role"] == "match").sum()
            missed  = (all_df["role"] == "miss").sum()
            fa      = (all_df["role"] == "false_alarm").sum()
            print(f"  {label}: match={matched}  miss={missed}  fa={fa}", flush=True)

    out = pathlib.Path(args.out or REPO / "docs/scale_summary_baselines.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(summary_rows, ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
