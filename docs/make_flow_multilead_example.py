"""Multi-lead gallery for docs/model_reference.html: persistence / two-head /
flow draw / truth at leads 1, 2, 3, 6 h, same initialization frame (piece 8,
frame 64, block (1,4)), shared log color scale. CPU login-safe (~2 min:
4x(1 two-head forward + 32 flow steps))."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"

INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
SCALE = 2.0
LEADS = [1, 2, 3, 6]
SIGMA = {1: 0.1457, 2: 0.1713, 3: 0.1876, 6: 0.204}
NSTEPS = 32

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def main():
    torch.set_num_threads(4)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    vtabs = {v: np.load(PIECES / f"{v}_piece{PIECE:02d}.npy", mmap_mode="r")
             for v in INPUT_VARS}
    ytab = np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")
    x = np.stack([np.asarray(vtabs[v][FRAME], dtype=np.float32)[sl]
                  for v in INPUT_VARS], 0)
    cond = torch.from_numpy(x[None])
    persist = np.asarray(ytab[FRAME + 1], dtype=np.float32)[sl]

    rows = {}
    with torch.no_grad():
        for L in LEADS:
            delta = L * 4 + 1
            truth = np.asarray(ytab[FRAME + delta], dtype=np.float32)[sl]
            twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
            twohead.load_state_dict(torch.load(
                RUNS / f"unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead{L:g}h/checkpoint.pt",
                map_location="cpu", weights_only=True))
            twohead.eval()
            det, _ = twohead(cond)
            det_mm = np.clip(det.numpy()[0, 0], 0, None)
            net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5,
                                           time_emb_dim=128)
            net.load_state_dict(torch.load(
                RUNS / f"flow_models/flow_matching_lead{L:g}h.checkpoint.pt",
                map_location="cpu", weights_only=True))
            net.eval()
            rf = RectifiedFlow(sigma=SIGMA[L])
            zf = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
            g = torch.Generator().manual_seed(7)
            z = rf.sample(net, cond, zf, n_steps=NSTEPS, generator=g)
            flow_mm = np.clip(np.expm1(SCALE * np.clip(z.numpy()[0, 0], 0, 2.5)),
                              0, None)
            rows[L] = (det_mm, flow_mm, truth)
            print(f"lead {L}h: det_max={det_mm.max():.1f} flow_max={flow_mm.max():.1f} "
                  f"truth_max={truth.max():.1f}", flush=True)

    vmax = max(persist.max(), max(max(a.max() for a in v) for v in rows.values()), 0.5)
    norm = LogNorm(vmin=0.1, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")
    COLS = {"persistence": "#7a7a7a", "two-head": "#2166ac",
            "flow": "#b05e0a", "truth": "#1a1a1a"}

    fig, axes = plt.subplots(len(LEADS), 4, figsize=(9.6, 2.42 * len(LEADS)))
    for i, L in enumerate(LEADS):
        det_mm, flow_mm, truth = rows[L]
        fields = [("persistence", persist), ("two-head", det_mm),
                  ("flow draw", flow_mm), ("truth", truth)]
        for j, (name, f) in enumerate(fields):
            ax = axes[i, j]
            im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                           interpolation="nearest")
            color = COLS[name.split()[0] if name != "flow draw" else "flow"]
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(color); sp.set_linewidth(2.0)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=9, color=color)
            if j == 0:
                ax.set_ylabel(f"+{L} h", fontsize=10)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015)
    cb.set_label("rain (mm)", fontsize=8)
    fig.suptitle("one initialization (piece 8, frame 64), four leads — "
                 "persistence, deterministic, flow draw, truth", fontsize=10, y=0.995)
    fig.savefig(OUT / "flow_multilead_gallery.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("saved flow_multilead_gallery.png")


if __name__ == "__main__":
    main()
