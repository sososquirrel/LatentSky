"""Full held-out test set systems table for the PERSISTENCE null at one lead:
truth at t+delta vs rain at t+1 (the aligned current-rain field every model
receives as input) over every test block of every day (patch split seed
1919810). Model-free, CPU, ~3 min. Lead via argv (default 3). Writes
data/eval_method/systems_alltest_persistence_lead{L}h.csv.gz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from latentsky.metrics import cell_area_km2, match_systems
from latentsky.training import blocks_for_piece, make_patch_split

PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUTDIR = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
P, THR, IOU_MIN = 128, 0.1, 0.1


def main():
    lead = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    delta = int(round(lead * 4)) + 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    area_full = cell_area_km2(320, 1440)
    _, _, test_b = make_patch_split(40, 2, 11, 1919810, 0.7, 0.15)

    rows = []
    for pid in range(40):
        blocks = sorted(blocks_for_piece(test_b, pid))
        if not blocks:
            continue
        ytab = np.load(PIECES / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")
        for t in range(96 - delta):
            yt = np.asarray(ytab[t + delta], dtype=np.float32)
            yp = np.asarray(ytab[t + 1], dtype=np.float32)
            for r, c in blocks:
                sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
                dfr = match_systems(yt[sl], yp[sl], THR,
                                    area_full[r * P:(r + 1) * P], iou_min=IOU_MIN)
                dfr.insert(0, "frame", t)
                dfr.insert(0, "block_col", c)
                dfr.insert(0, "block_row", r)
                dfr.insert(0, "piece", pid)
                rows.append(dfr)
        print(f"piece {pid} done", flush=True)

    out = pd.concat(rows, ignore_index=True)
    fn = OUTDIR / f"systems_alltest_persistence_lead{lead:g}h.csv.gz"
    out.to_csv(fn, index=False, compression="gzip")
    print("DONE", len(out), "systems:", out.role.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
