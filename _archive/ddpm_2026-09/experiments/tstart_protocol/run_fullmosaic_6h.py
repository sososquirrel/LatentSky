"""Full-mosaic single-chain sampling at lead 6 h: does the 0.80 band-ratio
saturation come from independent per-tile chains (decoherence) or from the
model itself? One reverse chain over the whole 256x1408 field (the network is
fully convolutional; 256 and 1408 are divisible by 2^5). See HYPOTHESIS.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from latentsky.models import ConditionalDiffusionUNet, GaussianDiffusion, UNet2DTwoHead

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run import RUNS, SCALE, band_power, mosaic, refine, tiles  # noqa: E402
from run_tstart_by_lead import load_frame_lead  # noqa: E402

CLIP = (0.0, 2.5)
DELTA = 25  # 6 h
FRAMES = [8, 24, 40, 56]
T_STARTS = [500, 750]


def main():
    device = torch.device("cuda")
    gd = GaussianDiffusion(1000, "cosine", device=device)

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead6h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()
    diff = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    diff.load_state_dict(torch.load(
        RUNS / "gen_models/10var/diffusion_lead6h/checkpoint.pt",
        map_location=device, weights_only=True))
    diff = diff.to(device).eval()

    truth_bands, res = [], {ts: {"band": [], "wet": []} for ts in T_STARTS}
    with torch.no_grad():
        for fi, t in enumerate(FRAMES):
            x, y = load_frame_lead(t, DELTA)
            truth_bands.append(band_power(y))
            # two-head forecast per tile (as trained), then reassemble the mosaic
            cond_tiles = tiles(x).to(device)
            det, _ = twohead(cond_tiles)
            det_mosaic = mosaic(np.clip(det.cpu().numpy()[:, 0], 0, None))
            x0_init = torch.from_numpy(
                np.log1p(det_mosaic)[None, None] / SCALE).float().to(device)
            cond_full = torch.from_numpy(x[None]).to(device)
            for ts in T_STARTS:
                z = refine(gd, diff, cond_full, x0_init, ts, seed=42 + fi)
                mm = np.clip(np.expm1(SCALE * np.clip(z.cpu().numpy()[0, 0], 0.0, 2.5)), 0, None)
                res[ts]["band"].append(band_power(mm))
                res[ts]["wet"].append(float((mm > 0.1).mean()))
                print(f"frame {t} ts={ts}: band_ratio(frame)="
                      f"{res[ts]['band'][-1] / truth_bands[-1]:.3f}", flush=True)

    tb = float(np.mean(truth_bands))
    for ts in T_STARTS:
        print(f"FULL-MOSAIC lead 6h t_start={ts}: band_ratio="
              f"{np.mean(res[ts]['band']) / tb:.3f} wet={np.mean(res[ts]['wet']):.4f} "
              f"(tiled reference: 0.80 at both)", flush=True)


if __name__ == "__main__":
    main()
