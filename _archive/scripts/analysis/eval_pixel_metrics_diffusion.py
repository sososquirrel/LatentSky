"""Unified full-scan pixel-metrics accumulator for the diffusion track:
confusion histogram (fig 01), wet-pixel value histograms (fig 02), and
exact R2/RMSE moment accumulators (fig 03) -- all in ONE pass per lead
over the full 40-piece patch-split test-block scope, so diffusion's
expensive inference (t_start+1 sequential steps x patches x ensemble x
frames x pieces) only has to run once per lead, not three times.

Persists everything to docs/pixel_metrics_diffusion_lead{L}h.npz so none
of figs 01/02/03 ever need a repeat multi-hour diffusion pass just to
change a bin count or color scale -- including the raw (masked to
held-out test blocks) truth/pred pixel arrays themselves, not just the
aggregated histograms/moments, so ANY future pixel-level metric can be
derived without ever re-running diffusion inference again.

R2/RMSE are computed from exact running-sum moments (n, sum_y, sum_y2,
sum_yp2, sum_y_yp), not approximated from the histogram bins:
  SS_res = sum_y2 - 2*sum_y_yp + sum_yp2
  SS_tot = sum_y2 - n*mean_y**2
  R2 = 1 - SS_res/SS_tot ; RMSE = sqrt(SS_res/n)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.predict_dispatch import predict_piece_track
from latentsky.registry import REPO_ROOT as REPO

N_TOTAL_PIECES = 40
CONF_VMAX = 25.0
CONF_NBINS = 30
HIST_VMIN, HIST_VMAX, HIST_NBINS = 0.05, 50.0, 60


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=float, required=True)
    p.add_argument("--pieces", type=int, nargs="+", default=None,
                    help="Bound the piece scope (for smoke-testing); default: all 40.")
    p.add_argument("--max-frames-per-piece", type=int, default=16)
    p.add_argument("--t-start", type=int, default=100)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    lead_tag = f"{args.lead:g}h"
    run_name = f"diffusion_lead{lead_tag}"
    track = {"kind": "diffusion", "run": run_name, "t_start": args.t_start,
             "ensemble_size": args.ensemble_size}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}  run={run_name}  "
          f"pieces={'all 40' if args.pieces is None else args.pieces}  "
          f"max_frames_per_piece={args.max_frames_per_piece}  "
          f"t_start={args.t_start}  ensemble_size={args.ensemble_size}", flush=True)

    conf_bins = np.linspace(0.0, CONF_VMAX, CONF_NBINS + 1)
    hist_bins = np.geomspace(HIST_VMIN, HIST_VMAX, HIST_NBINS + 1)

    conf_total = np.zeros((CONF_NBINS, CONF_NBINS), dtype=np.int64)
    truth_hist = np.zeros(HIST_NBINS, dtype=np.int64)
    pred_hist = np.zeros(HIST_NBINS, dtype=np.int64)
    n = 0
    sum_y = sum_y2 = sum_yp = sum_yp2 = sum_y_yp = 0.0
    raw_yt_chunks, raw_yp_chunks, raw_piece_ids = [], [], []

    piece_ids = args.pieces if args.pieces is not None else range(N_TOTAL_PIECES)
    test_blocks = None
    for pid in piece_ids:
        try:
            _, Y, Yp = predict_piece_track(track, pid, args.max_frames_per_piece, device=device)
        except Exception as e:
            print(f"  piece {pid} failed: {e}", flush=True)
            continue
        if Y.shape[0] == 0:
            continue
        H, W = Y.shape[-2], Y.shape[-1]
        if test_blocks is None:
            nrows, ncols = compute_patch_grid(H, W, 128)
            _, _, test_blocks = make_patch_split(N_TOTAL_PIECES, nrows, ncols, 1919810, 0.7, 0.15)
        blocks = blocks_for_piece(test_blocks, pid)
        if not blocks:
            print(f"  piece {pid}: no test blocks, skipping", flush=True)
            continue
        mask = np.zeros((H, W), dtype=bool)
        for r, c in blocks:
            i, j = r * 128, c * 128
            mask[i:i + 128, j:j + 128] = True
        yt = Y[:, 0][:, mask]
        yp = Yp[:, 0][:, mask]
        if yt.size == 0:
            continue

        conf, _, _ = np.histogram2d(yt.ravel(), yp.ravel(), bins=[conf_bins, conf_bins])
        conf_total += conf.astype(np.int64)

        yt_wet = yt[yt > HIST_VMIN]
        yp_wet = yp[yp > HIST_VMIN]
        th, _ = np.histogram(yt_wet, bins=hist_bins)
        ph, _ = np.histogram(yp_wet, bins=hist_bins)
        truth_hist += th.astype(np.int64)
        pred_hist += ph.astype(np.int64)

        yt_f = yt.ravel().astype(np.float64)
        yp_f = yp.ravel().astype(np.float64)
        n += yt_f.size
        sum_y += yt_f.sum()
        sum_y2 += (yt_f ** 2).sum()
        sum_yp += yp_f.sum()
        sum_yp2 += (yp_f ** 2).sum()
        sum_y_yp += (yt_f * yp_f).sum()

        raw_yt_chunks.append(yt.ravel().astype(np.float32))
        raw_yp_chunks.append(yp.ravel().astype(np.float32))
        raw_piece_ids.append(pid)

        print(f"  piece {pid}: {yt_f.size} pixels accumulated (running n={n})", flush=True)

    if n == 0:
        raise SystemExit("no pixels accumulated -- check piece/test-block scope")

    mean_y = sum_y / n
    ss_res = sum_y2 - 2 * sum_y_yp + sum_yp2
    ss_tot = sum_y2 - n * mean_y ** 2
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / n))
    print(f"lead={lead_tag}: n={n} R2={r2:.4f} RMSE={rmse:.4f}", flush=True)

    raw_yt = np.concatenate(raw_yt_chunks)
    raw_yp = np.concatenate(raw_yp_chunks)
    raw_piece_lens = np.array([len(c) for c in raw_yt_chunks], dtype=np.int64)
    raw_piece_ids = np.array(raw_piece_ids, dtype=np.int64)
    print(f"raw pixel arrays: {raw_yt.nbytes / 1e6:.1f} MB (truth) + "
          f"{raw_yp.nbytes / 1e6:.1f} MB (pred)", flush=True)

    out = args.out or str(REPO / f"docs/pixel_metrics_diffusion_lead{lead_tag}.npz")
    np.savez_compressed(
        out, conf=conf_total, conf_bins=conf_bins,
        truth_hist=truth_hist, pred_hist=pred_hist, hist_bins=hist_bins,
        n=n, sum_y=sum_y, sum_y2=sum_y2, sum_yp=sum_yp, sum_yp2=sum_yp2,
        sum_y_yp=sum_y_yp, r2=r2, rmse=rmse, lead_h=args.lead,
        raw_yt=raw_yt, raw_yp=raw_yp,
        raw_piece_ids=raw_piece_ids, raw_piece_lens=raw_piece_lens,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
