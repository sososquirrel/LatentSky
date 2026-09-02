"""Lead-dependent t_start: match the noise budget to the forecast's own error.

Rule: t*(lead) = smallest t with (1 - abar_t) >= abar_t * MSE_z(lead), where
MSE_z is the two-head forecast's error variance in z = log1p(rain)/2 space —
the noise injected at t* then exceeds the warm start's distance to truth, so
the chain can bridge it. Verifies the rule by measuring the small-scale power
ratio of single draws at t*(lead) for three leads. See HYPOTHESIS.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from latentsky.models import ConditionalDiffusionUNet, GaussianDiffusion, UNet2DTwoHead

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run import (  # noqa: E402
    FRAMES, INPUT_VARS, PIECES, RUNS, SCALE, band_power, mosaic, refine, tiles,
    H_USE, W_USE, PIECE,
)

OUT = Path(__file__).parent / "out"
LEADS = [0.25, 0.5, 1, 2, 3, 6]
VERIFY_LEADS = [0.25, 1, 6]


def load_frame_lead(t, delta):
    chans = []
    for v in INPUT_VARS:
        a = np.load(PIECES / f"{v}_piece{PIECE:02d}.npy", mmap_mode="r")
        chans.append(np.asarray(a[t], dtype=np.float32))
    x = np.stack(chans, axis=0)
    y = np.asarray(
        np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")[t + delta],
        dtype=np.float32,
    )
    return x[:, :H_USE, :W_USE], y[:H_USE, :W_USE]


def main():
    device = torch.device("cuda")
    gd = GaussianDiffusion(1000, "cosine", device=device)
    abar = gd.alphas_cumprod.cpu().numpy()

    rows = []
    for lead in LEADS:
        delta = int(round(lead * 4)) + 1
        twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
        twohead.load_state_dict(torch.load(
            RUNS / f"unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead{lead:g}h/checkpoint.pt",
            map_location=device, weights_only=True))
        twohead = twohead.to(device).eval()

        frames = [t for t in FRAMES if t + delta < 96]
        mses, det_bands, truth_bands, truths, conds, dets = [], [], [], [], [], []
        with torch.no_grad():
            for t in frames:
                x, y = load_frame_lead(t, delta)
                cond = tiles(x).to(device)
                det, _ = twohead(cond)
                det_mm = mosaic(np.clip(det.cpu().numpy()[:, 0], 0, None))
                z_det = np.log1p(det_mm) / SCALE
                z_truth = np.log1p(np.clip(y, 0, None)) / SCALE
                mses.append(float(np.mean((z_det - z_truth) ** 2)))
                det_bands.append(band_power(det_mm))
                truth_bands.append(band_power(y))
                truths.append(y); conds.append(cond); dets.append(det)
        mse_z = float(np.mean(mses))
        tstar = int(np.argmax((1 - abar) >= abar * mse_z))
        row = dict(lead_h=lead, mse_z=mse_z, t_star=tstar,
                   abar_star=float(abar[tstar]),
                   det_band_ratio=float(np.mean(det_bands) / np.mean(truth_bands)))

        if lead in VERIFY_LEADS:
            diff = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
            diff.load_state_dict(torch.load(
                RUNS / f"gen_models/10var/diffusion_lead{lead:g}h/checkpoint.pt",
                map_location=device, weights_only=True))
            diff = diff.to(device).eval()
            bands, wets = [], []
            with torch.no_grad():
                for fi, (cond, det, y) in enumerate(zip(conds, dets, truths)):
                    x0_init = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
                    z = refine(gd, diff, cond, x0_init, tstar, seed=555 + fi)
                    mm = mosaic(np.clip(
                        np.expm1(SCALE * np.clip(z.cpu().numpy()[:, 0], 0.0, 2.5)), 0, None))
                    bands.append(band_power(mm))
                    wets.append(float((mm > 0.1).mean()))
            row["draw_band_ratio_at_tstar"] = float(np.mean(bands) / np.mean(truth_bands))
            row["draw_wet_frac"] = float(np.mean(wets))
            row["truth_wet_frac"] = float(np.mean([(y > 0.1).mean() for y in truths]))
        rows.append(row)
        print(row, flush=True)
        del twohead, conds, dets, truths
        torch.cuda.empty_cache()

    keys = ["lead_h", "mse_z", "t_star", "abar_star", "det_band_ratio",
            "draw_band_ratio_at_tstar", "draw_wet_frac", "truth_wet_frac"]
    with open(OUT / "tstart_by_lead.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("done")


def mosaic_inv(y):
    """Truth field is already a (H_USE, W_USE) mosaic; identity for clarity."""
    return y


if __name__ == "__main__":
    main()
