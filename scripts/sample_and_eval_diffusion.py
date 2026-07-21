"""Generate forecasts from the trained diffusion model via full ancestral
DDPM sampling (see latentsky/diffusion.py:GaussianDiffusion.ddpm_sample --
DDIM was tried first and found to be numerically unstable at high noise
levels; ddpm_sample fixes that), and compute pixel-level R^2/RMSE on the
SAME held-out test blocks used by patchsplit_twohead_olr_fss_lead3h
(diffusion_lead3h.yaml deliberately uses the identical
patch_split_seed/train_frac/val_frac/patch_size, so this is the exact same
test population -- the persistence/climatology/random numbers already in
docs/baselines_lead3h_newsplits.csv for "patchsplit_test_blocks" apply
directly here too).

Samples at PATCH resolution (128x128, matching training exactly) rather
than the full domain -- avoids a train/inference resolution mismatch and is
~28x cheaper per sample, which matters a lot since each DDPM draw costs
`diffusion_timesteps` (1000) sequential forward passes.

A single DDPM draw is one random sample from a distribution, not a point
forecast -- `--ensemble-size` independent draws are averaged per frame to
get a stable point estimate comparable to the deterministic regression
models (standard practice for scoring generative forecasts).

Full ancestral sampling is expensive (1000 steps/draw), so this only
evaluates a bounded number of pieces/frames -- not the same exhaustive pass
used for the regression models. Scope is explicit via CLI args and always
logged so results aren't silently mistaken for a full-coverage eval.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch

from latentsky.data import load_piece_arrays
from latentsky.diffusion import GaussianDiffusion
from latentsky.models import ConditionalDiffusionUNet
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.registry import load_run

from latentsky.registry import REPO_ROOT as REPO


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float64)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2, dtype=np.float64)
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, dtype=np.float64)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="diffusion_lead3h")
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--frames-per-piece", type=int, default=2)
    p.add_argument("--blocks-per-piece", type=int, default=1,
                    help="Cap on how many test blocks to sample per piece (speed).")
    p.add_argument("--max-pieces", type=int, default=8)
    p.add_argument("--out-csv", default="docs/diffusion_vs_baselines_lead3h.csv")
    args = p.parse_args()
    print(f"scope: max_pieces={args.max_pieces} blocks_per_piece={args.blocks_per_piece} "
          f"frames_per_piece={args.frames_per_piece} ensemble_size={args.ensemble_size} "
          f"(NOT full coverage -- see script docstring)", flush=True)

    run = load_run(args.run_name)
    cfg = run.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model = ConditionalDiffusionUNet(
        cond_ch=cfg.in_channels, base_ch=cfg.base_ch, depth=cfg.depth,
        dropout_p=cfg.dropout_p, time_emb_dim=cfg.diffusion_time_emb_dim,
    )
    state = torch.load(run.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    diffusion = GaussianDiffusion(cfg.diffusion_timesteps, cfg.diffusion_schedule, device=device)
    scale = cfg.diffusion_target_scale
    P = cfg.patch_size

    probe = np.load(f"{cfg.pieces_dir}/{cfg.input_vars[0]}_piece00.npy", mmap_mode="r")
    _, H, W = probe.shape
    nrows, ncols = compute_patch_grid(H, W, P)
    _, _, test_blocks = make_patch_split(
        cfg.n_total_pieces, nrows, ncols, cfg.patch_split_seed,
        cfg.patch_split_train_frac, cfg.patch_split_val_frac,
    )
    print(f"grid={nrows}x{ncols}  total test blocks={len(test_blocks)}", flush=True)

    truths, preds = [], []
    n_pieces_done = 0
    for pid in range(cfg.n_total_pieces):
        blocks = sorted(blocks_for_piece(test_blocks, pid))
        if not blocks:
            continue
        if n_pieces_done >= args.max_pieces:
            break
        n_pieces_done += 1
        blocks = blocks[: args.blocks_per_piece]

        X, Y = load_piece_arrays(cfg.pieces_dir, pid, cfg.input_vars, cfg.target_prefix, cfg.delta_index)
        n_frames = min(args.frames_per_piece, X.shape[0])
        frame_idx = np.linspace(0, X.shape[0] - 1, n_frames).round().astype(int)

        for (r, c) in blocks:
            i, j = r * P, c * P
            for fi in frame_idx:
                cond = torch.from_numpy(X[fi:fi + 1, :, i:i + P, j:j + P]).to(device)
                draws = []
                for _ in range(args.ensemble_size):
                    x0 = diffusion.ddpm_sample(model, cond)
                    draws.append(torch.expm1(x0 * scale))
                rain_pred = torch.clamp(torch.stack(draws).mean(dim=0), min=0.0).cpu().numpy()[0, 0]
                rain_truth = Y[fi, 0, i:i + P, j:j + P]
                truths.append(rain_truth.ravel())
                preds.append(rain_pred.ravel())
                print(f"piece {pid} block ({r},{c}) frame {fi}: "
                      f"truth mean={rain_truth.mean():.4f} pred mean={rain_pred.mean():.4f}", flush=True)
        del X, Y

    t = np.concatenate(truths)
    p_ = np.concatenate(preds)
    row = {"predictor_set": "diffusion", "eval_set": "patchsplit_test_blocks",
           "R2_all": r2(t, p_), "RMSE": rmse(t, p_), "n_pixels": len(t),
           "n_pieces": n_pieces_done, "ensemble_size": args.ensemble_size}
    print("\n" + str(row))

    out = REPO / args.out_csv
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
