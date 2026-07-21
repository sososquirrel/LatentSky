"""Residual/refinement ('SDEdit'-style warm-started) diffusion sampling:
instead of starting the reverse DDPM chain from pure Gaussian noise
(sample_and_eval_diffusion.py / GaussianDiffusion.ddpm_sample -- found to
badly overpredict on held-out test blocks, R2=-486 vs random's -1.1), start
it from the patch-split two-head REGRESSION model's own point prediction,
forward-noised out to a chosen t_start, then run the same reverse loop for
only t_start+1 steps (GaussianDiffusion.ddpm_refine_sample). No retraining
needed -- reuses the existing diffusion_lead3h checkpoint as-is.

Sweeps t_start so we can see the tradeoff directly: t_start=0 is just the
regression prediction with a single denoise step (should ~= regression
alone); larger t_start lets diffusion move further from the regression
starting point (more "generative" freedom) at the cost of re-opening the
door to the compounding drift that broke pure ddpm_sample (t_start=999).

Same held-out test blocks as sample_and_eval_diffusion.py
(patchsplit_test_blocks, from diffusion_lead3h's config, which deliberately
matches patchsplit_twohead_olr_fss_lead3h's split) -- so this is directly
comparable to that run's R2=-485.9, RMSE=4.35, and to the cached
persistence/climatology/random numbers in docs/baselines_lead3h_newsplits.csv.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch

from latentsky.data import load_piece_arrays
from latentsky.diffusion import GaussianDiffusion
from latentsky.inference import _load_model
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
    p.add_argument("--diffusion-run", default="diffusion_lead3h")
    p.add_argument("--regression-run", default="patchsplit_twohead_olr_fss_lead3h")
    p.add_argument("--t-starts", type=int, nargs="+", default=[50, 150, 400])
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--frames-per-piece", type=int, default=3)
    p.add_argument("--blocks-per-piece", type=int, default=2)
    p.add_argument("--max-pieces", type=int, default=12)
    p.add_argument("--out-csv", default="docs/diffusion_refine_vs_baselines_lead3h.csv")
    args = p.parse_args()
    print(f"t_starts={args.t_starts} ensemble_size={args.ensemble_size} "
          f"scope: max_pieces={args.max_pieces} blocks_per_piece={args.blocks_per_piece} "
          f"frames_per_piece={args.frames_per_piece}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    diff_run = load_run(args.diffusion_run)
    dcfg = diff_run.config
    diff_model = ConditionalDiffusionUNet(
        cond_ch=dcfg.in_channels, base_ch=dcfg.base_ch, depth=dcfg.depth,
        dropout_p=dcfg.dropout_p, time_emb_dim=dcfg.diffusion_time_emb_dim,
    )
    diff_model.load_state_dict(torch.load(diff_run.checkpoint_path, map_location=device))
    diff_model = diff_model.to(device).eval()
    diffusion = GaussianDiffusion(dcfg.diffusion_timesteps, dcfg.diffusion_schedule, device=device)
    scale = dcfg.diffusion_target_scale
    P = dcfg.patch_size

    reg_run = load_run(args.regression_run)
    reg_model = _load_model(reg_run, device)

    probe = np.load(f"{dcfg.pieces_dir}/{dcfg.input_vars[0]}_piece00.npy", mmap_mode="r")
    _, H, W = probe.shape
    nrows, ncols = compute_patch_grid(H, W, P)
    _, _, test_blocks = make_patch_split(
        dcfg.n_total_pieces, nrows, ncols, dcfg.patch_split_seed,
        dcfg.patch_split_train_frac, dcfg.patch_split_val_frac,
    )
    print(f"grid={nrows}x{ncols}  total test blocks={len(test_blocks)}", flush=True)

    truths = []
    reg_preds = []
    refine_preds = {t: [] for t in args.t_starts}
    n_pieces_done = 0
    for pid in range(dcfg.n_total_pieces):
        blocks = sorted(blocks_for_piece(test_blocks, pid))
        if not blocks:
            continue
        if n_pieces_done >= args.max_pieces:
            break
        n_pieces_done += 1
        blocks = blocks[: args.blocks_per_piece]

        X, Y = load_piece_arrays(dcfg.pieces_dir, pid, dcfg.input_vars, dcfg.target_prefix, dcfg.delta_index)
        n_frames = min(args.frames_per_piece, X.shape[0])
        frame_idx = np.linspace(0, X.shape[0] - 1, n_frames).round().astype(int)

        for (r, c) in blocks:
            i, j = r * P, c * P
            for fi in frame_idx:
                cond = torch.from_numpy(X[fi:fi + 1, :, i:i + P, j:j + P]).to(device)
                rain_truth = Y[fi, 0, i:i + P, j:j + P]

                with torch.no_grad():
                    out = reg_model(cond)
                    reg_rain = out[0] if isinstance(out, tuple) else out
                    reg_rain = torch.clamp(reg_rain, min=0.0)
                reg_pred_np = reg_rain.cpu().numpy()[0, 0]
                x0_init = torch.log1p(reg_rain) / scale

                truths.append(rain_truth.ravel())
                reg_preds.append(reg_pred_np.ravel())

                for t_start in args.t_starts:
                    draws = []
                    for _ in range(args.ensemble_size):
                        x0 = diffusion.ddpm_refine_sample(diff_model, cond, x0_init, t_start)
                        draws.append(torch.expm1(x0 * scale))
                    refine_rain = torch.clamp(torch.stack(draws).mean(dim=0), min=0.0).cpu().numpy()[0, 0]
                    refine_preds[t_start].append(refine_rain.ravel())

                msg = f"piece {pid} block ({r},{c}) frame {fi}: truth={rain_truth.mean():.4f} reg={reg_pred_np.mean():.4f}"
                for t_start in args.t_starts:
                    msg += f" refine[{t_start}]={refine_preds[t_start][-1].mean():.4f}"
                print(msg, flush=True)
        del X, Y

    t = np.concatenate(truths)
    rows = []
    reg_flat = np.concatenate(reg_preds)
    rows.append({"predictor_set": "regression_only", "eval_set": "patchsplit_test_blocks",
                 "R2_all": r2(t, reg_flat), "RMSE": rmse(t, reg_flat), "n_pixels": len(t)})
    for t_start in args.t_starts:
        p_ = np.concatenate(refine_preds[t_start])
        rows.append({"predictor_set": f"diffusion_refine_tstart{t_start}", "eval_set": "patchsplit_test_blocks",
                     "R2_all": r2(t, p_), "RMSE": rmse(t, p_), "n_pixels": len(t),
                     "n_pieces": n_pieces_done, "ensemble_size": args.ensemble_size})
    print("\n" + "\n".join(str(row) for row in rows))

    out = REPO / args.out_csv
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
