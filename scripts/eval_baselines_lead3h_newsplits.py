"""Persistence/climatology/random baselines at lead=3h, evaluated on the
SAME held-out data as each new two-head model -- fixes the apples-to-oranges
comparison in the first pass (persistence was on [2,12,22,32], the new
models were on a completely different held-out set).

Two held-out definitions, matching the two new models' actual test sets:
  - "piece" mode (twohead_olr_fss_lead3h / _finetuned): whole test pieces
    [1, 13, 23, 32].
  - "patch" mode (patchsplit_twohead_olr_fss_lead3h): scattered non-
    overlapping test blocks across all 40 pieces (latentsky.patch_split).

Uses /glade/derecho/scratch/sabramian/latentsky_pieces (the new models'
pieces_dir) for rain_total, not the old tzhou default -- baselines only
need the raw rain field, so this is just for consistency, not because the
physical data differs.
"""
from __future__ import annotations
import gc
import pathlib
import numpy as np
import pandas as pd

from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split

REPO = pathlib.Path(__file__).resolve().parents[1]
PIECES_DIR = pathlib.Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
DELTA_3H = 13  # lead_steps_15=12 -> delta_index=13
N_TOTAL = 40


def load_rain(pid: int) -> np.ndarray:
    return np.load(PIECES_DIR / f"rain_total_piece{pid:02d}.npy", mmap_mode="r").astype(np.float32)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float64)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2, dtype=np.float64)
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, dtype=np.float64)))


def eval_piece_mode(test_pieces: list[int]) -> pd.DataFrame:
    print(f"[piece-mode] test pieces = {test_pieces}", flush=True)
    clim_sum, clim_n = None, 0
    for pid in test_pieces:
        Y = load_rain(pid)
        clim_sum = Y.sum(axis=0) if clim_sum is None else clim_sum + Y.sum(axis=0)
        clim_n += Y.shape[0]
    clim_mean = clim_sum / clim_n

    rand_pieces = {
        pid: load_rain(test_pieces[(i + 2) % len(test_pieces)])
        for i, pid in enumerate(test_pieces)
    }
    accum = {"persistence": ([], []), "climatology": ([], []), "random": ([], [])}
    for pid in test_pieces:
        Y = load_rain(pid)
        Y_rand = rand_pieces[pid]
        T = Y.shape[0] - DELTA_3H - 1
        truth = np.ascontiguousarray(Y[DELTA_3H + 1: DELTA_3H + 1 + T])
        pers = np.ascontiguousarray(Y[0:T])
        clim = np.broadcast_to(clim_mean[None], truth.shape).copy()
        rand = np.ascontiguousarray(Y_rand[0:T])
        for label, pred in [("persistence", pers), ("climatology", clim), ("random", rand)]:
            accum[label][0].append(truth.ravel())
            accum[label][1].append(pred.ravel())

    rows = []
    for label, (ts, ps) in accum.items():
        t, p = np.concatenate(ts), np.concatenate(ps)
        rows.append({"predictor_set": label, "eval_set": "piece_1_13_23_32",
                     "R2_all": r2(t, p), "RMSE": rmse(t, p)})
    return pd.DataFrame(rows)


def eval_patch_mode(patch_size: int, seed: int, train_frac: float, val_frac: float) -> pd.DataFrame:
    probe = np.load(PIECES_DIR / "rain_total_piece00.npy", mmap_mode="r")
    _, H, W = probe.shape
    nrows, ncols = compute_patch_grid(H, W, patch_size)
    _, _, test_blocks = make_patch_split(N_TOTAL, nrows, ncols, seed, train_frac, val_frac)
    print(f"[patch-mode] grid={nrows}x{ncols}  total test blocks={len(test_blocks)}", flush=True)

    # climatology mean over ALL 40 pieces (matches patch-split's own train
    # scope using all days) -- computed once.
    clim_sum, clim_n = None, 0
    for pid in range(N_TOTAL):
        Y = load_rain(pid)
        clim_sum = Y.sum(axis=0) if clim_sum is None else clim_sum + Y.sum(axis=0)
        clim_n += Y.shape[0]
        del Y
        gc.collect()
    clim_mean = clim_sum / clim_n

    accum = {"persistence": ([], []), "climatology": ([], []), "random": ([], [])}
    for pid in range(N_TOTAL):
        blocks = blocks_for_piece(test_blocks, pid)
        if not blocks:
            continue
        mask = np.zeros((H, W), dtype=bool)
        for r, c in blocks:
            i, j = r * patch_size, c * patch_size
            mask[i:i + patch_size, j:j + patch_size] = True

        Y = load_rain(pid)
        Y_rand = load_rain((pid + 20) % N_TOTAL)  # ~20-piece offset, decorrelated
        T = Y.shape[0] - DELTA_3H - 1
        truth = np.ascontiguousarray(Y[DELTA_3H + 1: DELTA_3H + 1 + T])
        pers = np.ascontiguousarray(Y[0:T])
        clim = np.broadcast_to(clim_mean[None], truth.shape).copy()
        rand = np.ascontiguousarray(Y_rand[0:T])

        print(f"  [patch-mode] piece {pid}: {len(blocks)} test blocks, T={T}", flush=True)
        for label, pred in [("persistence", pers), ("climatology", clim), ("random", rand)]:
            accum[label][0].append(truth[:, mask].ravel())
            accum[label][1].append(pred[:, mask].ravel())

        del Y, Y_rand, truth, pers, clim, rand
        gc.collect()

    rows = []
    for label, (ts, ps) in accum.items():
        t, p = np.concatenate(ts), np.concatenate(ps)
        rows.append({"predictor_set": label, "eval_set": "patchsplit_test_blocks",
                     "R2_all": r2(t, p), "RMSE": rmse(t, p)})
    return pd.DataFrame(rows)


def main():
    df1 = eval_piece_mode([1, 13, 23, 32])
    df2 = eval_patch_mode(patch_size=128, seed=1919810, train_frac=0.7, val_frac=0.15)
    df = pd.concat([df1, df2], ignore_index=True)
    print("\n" + df.to_string(index=False))
    out = REPO / "docs/baselines_lead3h_newsplits.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
