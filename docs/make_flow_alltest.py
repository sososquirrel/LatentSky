"""Full held-out test set systems table for the FLOW at one lead: truth vs one
flow draw over every test block of every day (patch split seed 1919810).
Per-draw scoring; GPU job (~45 min on an A100 per lead). Lead via argv
(default 3). Writes data/eval_method/systems_alltest_flow_lead{L}h.csv.gz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from latentsky.metrics import cell_area_km2, match_systems
from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow
from latentsky.training import blocks_for_piece, make_patch_split

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUTDIR = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")

INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, THR, IOU_MIN, SCALE, NSTEPS = 128, 0.1, 0.1, 2.0, 32
SIGMA = {0.25: 0.1018, 0.5: 0.1193, 1: 0.1457, 2: 0.1713, 3: 0.1876, 6: 0.204}


def main():
    lead = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    delta = int(round(lead * 4)) + 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "lead:", lead, flush=True)

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / f"unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead{lead:g}h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()
    net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / f"flow_models/flow_matching_lead{lead:g}h.checkpoint.pt",
        map_location=device, weights_only=True))
    net = net.to(device).eval()
    rf = RectifiedFlow(sigma=SIGMA[lead])

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
        for t in range(96 - delta):
            x = np.stack([np.asarray(vtabs[v][t], dtype=np.float32) for v in INPUT_VARS], 0)
            patches = np.stack([x[:, r * P:(r + 1) * P, c * P:(c + 1) * P]
                                for r, c in blocks], 0)
            cond = torch.from_numpy(patches).to(device)
            with torch.no_grad():
                det, _ = twohead(cond)
                zf = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
                g = torch.Generator(device=device).manual_seed(pid * 1000 + t)
                z = rf.sample(net, cond, zf, n_steps=NSTEPS, generator=g)
            mm = np.clip(np.expm1(SCALE * np.clip(z.cpu().numpy()[:, 0], 0, 2.5)), 0, None)
            yt = np.asarray(ytab[t + delta], dtype=np.float32)
            for bi, (r, c) in enumerate(blocks):
                dfr = match_systems(yt[r * P:(r + 1) * P, c * P:(c + 1) * P], mm[bi],
                                    THR, area_full[r * P:(r + 1) * P], iou_min=IOU_MIN)
                dfr.insert(0, "frame", t)
                dfr.insert(0, "block_col", c)
                dfr.insert(0, "block_row", r)
                dfr.insert(0, "piece", pid)
                rows.append(dfr)
        print(f"piece {pid} done", flush=True)

    out = pd.concat(rows, ignore_index=True)
    fn = OUTDIR / f"systems_alltest_flow_lead{lead:g}h.csv.gz"
    out.to_csv(fn, index=False, compression="gzip")
    print("DONE", len(out), "systems:", out.role.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
