"""Pixel-level R^2/RMSE at lead=3h for the new clean-model retrains
(twohead_olr_fss_lead3h, patchsplit_twohead_olr_fss_lead3h, and -- once it
finishes -- twohead_olr_fss_lead3h_finetuned), alongside the original
production LatentSky and persistence/climatology/random baselines already
cached from earlier in this session.

`patchsplit_twohead_olr_fss_lead3h` needs special handling: its test set is
NOT a set of whole days (unlike every other config here) -- it's individual
non-overlapping spatial patches scattered across all 40 pieces (see
latentsky/patch_split.py). Evaluating it fairly means running inference on
every piece but restricting the R^2/RMSE computation to just the pixels
inside that piece's test blocks.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch

from latentsky.inference import predict_piece
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.registry import load_run

from latentsky.registry import REPO_ROOT as REPO


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float64)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2, dtype=np.float64)
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, dtype=np.float64)))


def eval_piece_mode_run(name: str, max_frames: int) -> dict:
    run = load_run(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    truths, preds = [], []
    for pid in run.config.test_piece_ids:
        print(f"  [{name}] piece {pid} ...", flush=True)
        _, Y, Yp = predict_piece(run, pid, max_frames=max_frames, batch_size=1, device=device)
        truths.append(Y.ravel())
        preds.append(Yp.ravel())
    t = np.concatenate(truths)
    p = np.concatenate(preds)
    return {"name": name, "R2_all": r2(t, p), "RMSE": rmse(t, p), "n_pixels": len(t)}


def eval_patch_mode_run(name: str, max_frames: int) -> dict:
    run = load_run(name)
    cfg = run.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_path = f"{cfg.pieces_dir}/{cfg.input_vars[0]}_piece00.npy"
    _, H, W = np.load(probe_path, mmap_mode="r").shape
    nrows, ncols = compute_patch_grid(H, W, cfg.patch_size)
    _, _, test_blocks = make_patch_split(
        cfg.n_total_pieces, nrows, ncols, cfg.patch_split_seed,
        cfg.patch_split_train_frac, cfg.patch_split_val_frac,
    )

    truths, preds = [], []
    for pid in range(cfg.n_total_pieces):
        blocks = blocks_for_piece(test_blocks, pid)
        if not blocks:
            continue
        print(f"  [{name}] piece {pid}: {len(blocks)} test blocks ...", flush=True)
        _, Y, Yp = predict_piece(run, pid, max_frames=max_frames, batch_size=1, device=device)
        # mask to just the test-block pixels
        mask = np.zeros((H, W), dtype=bool)
        for r, c in blocks:
            i, j = r * cfg.patch_size, c * cfg.patch_size
            mask[i:i + cfg.patch_size, j:j + cfg.patch_size] = True
        truths.append(Y[:, 0, mask].ravel())
        preds.append(Yp[:, 0, mask].ravel())
    t = np.concatenate(truths)
    p = np.concatenate(preds)
    return {"name": name, "R2_all": r2(t, p), "RMSE": rmse(t, p), "n_pixels": len(t)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--out-csv", default="docs/clean_models_vs_baselines_lead3h.csv")
    args = p.parse_args()

    rows = []
    print("=== twohead_olr_fss_lead3h (piece-mode, from-scratch, overfit run) ===")
    rows.append(eval_piece_mode_run("twohead_olr_fss_lead3h", args.max_frames))

    print("=== patchsplit_twohead_olr_fss_lead3h (patch-mode) ===")
    rows.append(eval_patch_mode_run("patchsplit_twohead_olr_fss_lead3h", args.max_frames))

    try:
        print("=== twohead_olr_fss_lead3h_finetuned (pretrain-initialized) ===")
        rows.append(eval_piece_mode_run("twohead_olr_fss_lead3h_finetuned", args.max_frames))
    except Exception as e:
        print(f"  skipping finetuned run (not ready yet?): {e}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = REPO / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
