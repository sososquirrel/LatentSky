"""Precompute log1p-normalized rain input files for all 40 DYAMOND pieces.

The existing 9 atmospheric variables are already z-scored in the piece files.
Rain (rain_total) is raw (mm/15min, right-skewed). We apply log1p then z-score
using statistics computed over training pieces only (excluding test/val).

Output: rain_log1p_piece{id:02d}.npy  alongside the existing piece files.
"""
from __future__ import annotations
import pathlib
import numpy as np

TZHOU_DIR  = pathlib.Path(
    "/glade/derecho/scratch/tzhou/splitted_1440x720_9vars_40pieces_v2"
)
# Write rain_log1p to our own directory (tzhou's dir is read-only).
# All tzhou files are already symlinked there.
PIECES_DIR = pathlib.Path(
    "/glade/derecho/scratch/sabramian/latentsky_pieces"
)
N_PIECES   = 40
TEST_IDS   = {2, 12, 22, 32}
VAL_IDS    = {6, 16, 26, 36}
TRAIN_IDS  = [i for i in range(N_PIECES) if i not in TEST_IDS and i not in VAL_IDS]

# --- compute log1p mean/std over TRAINING pieces only ---
print("computing log1p statistics over training pieces ...", flush=True)
total_sum  = 0.0
total_sum2 = 0.0
total_n    = 0
for pid in TRAIN_IDS:
    arr = np.load(TZHOU_DIR / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")
    log_arr = np.log1p(arr.astype(np.float64))
    total_sum  += log_arr.sum()
    total_sum2 += (log_arr ** 2).sum()
    total_n    += log_arr.size

mean_log = total_sum  / total_n
std_log  = np.sqrt(total_sum2 / total_n - mean_log ** 2)
print(f"  log1p mean={mean_log:.4f}  std={std_log:.4f}  (over {len(TRAIN_IDS)} train pieces)", flush=True)

# --- write normalized files for all 40 pieces ---
print("\nwriting rain_log1p files ...", flush=True)
for pid in range(N_PIECES):
    out = PIECES_DIR / f"rain_log1p_piece{pid:02d}.npy"
    if out.exists():
        print(f"  piece {pid:02d}: already exists, skipping", flush=True)
        continue
    arr = np.load(TZHOU_DIR / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")
    normed = ((np.log1p(arr.astype(np.float32)) - mean_log) / std_log).astype(np.float32)
    np.save(out, normed)
    print(f"  piece {pid:02d}: {normed.shape}  mean={normed.mean():.3f}  std={normed.std():.3f}", flush=True)

print("\ndone.", flush=True)
