"""Persistence/climatology/random baselines, scored with the SAME
object-detection scale-decomposition pipeline as the trained runs, on the
SAME patch-split test blocks used by patchsplit_onehead_lead*/
patchsplit_twohead_olr_fss_lead*/diffusion_lead* -- so baselines and
models are compared on identical held-out data, not different subsets.

Writes docs/scale_systems_baselines_patchsplit.csv.gz /
docs/scale_summary_baselines_patchsplit.csv with the same
name/lead_h/predictor_set columns as
scripts/analysis/eval_scale_decomposition.py, so every downstream
plot_*.py / fit_*.py script picks these up unmodified via --runs
persistence_lead<L>h,climatology_lead<L>h,random_lead<L>h.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from latentsky.baselines import evaluate_baseline_systems
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.registry import REPO_ROOT as REPO
from latentsky.scale_analysis import default_scale_edges, summarize_by_scale

PIECES_DIR = "/glade/derecho/scratch/sabramian/latentsky_pieces"
N_TOTAL = 40
LEADS_H = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0]
KINDS = ["persistence", "climatology", "random"]


def restrict_to_patch_test_blocks(df: pd.DataFrame, patch_size: int, test_blocks: set) -> pd.DataFrame:
    if df.empty:
        return df
    keep = np.zeros(len(df), dtype=bool)
    for pid, sub in df.groupby("piece"):
        blocks = blocks_for_piece(test_blocks, pid)
        if not blocks:
            continue
        prow = (sub["row"].to_numpy() // patch_size).astype(int)
        pcol = (sub["col"].to_numpy() // patch_size).astype(int)
        in_test = np.array([(r, c) in blocks for r, c in zip(prow, pcol)])
        keep[sub.index.to_numpy()] = in_test
    return df[keep].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-frames-per-piece", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=3000.0)
    p.add_argument("--scale-bins", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--patch-split-seed", type=int, default=1919810)
    p.add_argument("--patch-split-train-frac", type=float, default=0.7)
    p.add_argument("--patch-split-val-frac", type=float, default=0.15)
    p.add_argument("--systems-out", default="docs/scale_systems_baselines_patchsplit.csv.gz")
    p.add_argument("--summary-out", default="docs/scale_summary_baselines_patchsplit.csv")
    args = p.parse_args()

    probe = np.load(f"{PIECES_DIR}/rain_total_piece00.npy", mmap_mode="r")
    _, H, W = probe.shape
    nrows, ncols = compute_patch_grid(H, W, args.patch_size)
    _, _, test_blocks = make_patch_split(
        N_TOTAL, nrows, ncols, args.patch_split_seed,
        args.patch_split_train_frac, args.patch_split_val_frac,
    )
    print(f"grid={nrows}x{ncols}  total test blocks={len(test_blocks)}", flush=True)

    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)

    clim_mean = None

    sys_frames, summary_rows = [], []
    i = 0
    total = len(KINDS) * len(LEADS_H)
    for kind in KINDS:
        if kind == "climatology" and clim_mean is None:
            print("computing climatology mean over all 40 pieces (once)...", flush=True)
            from latentsky.baselines import compute_climatology_mean
            clim_mean = compute_climatology_mean(PIECES_DIR, range(N_TOTAL))
        for lead_h in LEADS_H:
            i += 1
            name = f"{kind}_lead{lead_h:g}h"
            print(f"[{i}/{total}] {name}", flush=True)
            df = evaluate_baseline_systems(
                kind, PIECES_DIR, lead_h, range(N_TOTAL),
                max_frames_per_piece=args.max_frames_per_piece,
                threshold=args.threshold, iou_min=args.iou_min, n_total_pieces=N_TOTAL,
                clim_mean=clim_mean,
            )
            if df.empty:
                print(f"  no systems for {name}", flush=True)
                continue
            n_before = len(df)
            df = restrict_to_patch_test_blocks(df, args.patch_size, test_blocks)
            print(f"  patch-split filter: {n_before} -> {len(df)} systems", flush=True)
            if df.empty:
                continue
            df.insert(0, "name", name)
            df.insert(0, "lead_h", lead_h)
            df.insert(0, "predictor_set", "baseline")
            sys_frames.append(df)

            summ = summarize_by_scale(df, edges).by_scale
            summ.insert(0, "name", name)
            summ.insert(0, "lead_h", lead_h)
            summ.insert(0, "predictor_set", "baseline")
            summary_rows.append(summ)

    systems = pd.concat(sys_frames, ignore_index=True)
    summary = pd.concat(summary_rows, ignore_index=True)
    out_sys = REPO / args.systems_out
    out_summ = REPO / args.summary_out
    systems.to_csv(out_sys, index=False, compression="gzip" if str(out_sys).endswith(".gz") else None)
    summary.to_csv(out_summ, index=False)
    print(f"wrote {out_summ}  ({len(summary)} rows)")
    print(f"wrote {out_sys}  ({len(systems)} systems)")


if __name__ == "__main__":
    main()
