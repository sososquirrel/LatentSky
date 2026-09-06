"""Full held-out test set systems table at lead 3 h: truth vs two-head over
every test block of every day (patch split seed 1919810 — 132 blocks x 83
frames). CPU-friendly (batched two-head forwards); writes
data/eval_method/systems_alltest_lead3h.csv.gz. Meant for a CPU PBS job."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from latentsky.metrics import cell_area_km2, match_systems
from latentsky.models import UNet2DTwoHead
from latentsky.training import blocks_for_piece, make_patch_split

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUTDIR = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")

INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, THR, IOU_MIN = 128, 0.1, 0.1


def main():
    lead = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    DELTA = int(round(lead * 4)) + 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("lead:", lead, "delta:", DELTA, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(16)
    print("device:", device, flush=True)

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / f"unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead{lead:g}h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()

    area_full = cell_area_km2(320, 1440)
    _, _, test_b = make_patch_split(40, 2, 11, 1919810, 0.7, 0.15)

    rows = []
    for pid in range(40):
        blocks = sorted(blocks_for_piece(test_b, pid))
        if not blocks:
            continue
        ytab = np.load(PIECES / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")
        vtabs = {v: np.load(PIECES / f"{v}_piece{pid:02d}.npy", mmap_mode="r")
                 for v in INPUT_VARS}
        for t in range(96 - DELTA):
            x = np.stack([np.asarray(vtabs[v][t], dtype=np.float32)
                          for v in INPUT_VARS], 0)
            patches = np.stack([x[:, r * P:(r + 1) * P, c * P:(c + 1) * P]
                                for r, c in blocks], 0)
            with torch.no_grad():
                pred, _ = twohead(torch.from_numpy(patches).to(device))
            pred = np.clip(pred.cpu().numpy()[:, 0], 0, None)
            yt = np.asarray(ytab[t + DELTA], dtype=np.float32)
            for bi, (r, c) in enumerate(blocks):
                ytp = yt[r * P:(r + 1) * P, c * P:(c + 1) * P]
                arow = area_full[r * P:(r + 1) * P]
                dfr = match_systems(ytp, pred[bi], THR, arow, iou_min=IOU_MIN)
                dfr.insert(0, "frame", t)
                dfr.insert(0, "block_col", c)
                dfr.insert(0, "block_row", r)
                dfr.insert(0, "piece", pid)
                rows.append(dfr)
        print(f"piece {pid}: {len(blocks)} test blocks done", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUTDIR / f"systems_alltest_lead{lead:g}h.csv.gz", index=False,
               compression="gzip")
    n = out.role.value_counts().to_dict()
    print("DONE", len(out), "systems:", n, flush=True)


if __name__ == "__main__":
    main()
