"""The flow-matching method as one step-by-step figure on real data
(flow_process.png), replacing most of the prose in model_reference.html §4:

  1  one training pair: forecast -> source x0 = z_fc + sigma*eps, target x1
  2  the straight line x_t = (1-t) x0 + t x1 at t = 0.25 / 0.5 / 0.75
  3  the regression: velocity target x1 - x0, the trained network's v-hat,
     and the residual
  4  sampling: the real 32-step Heun integration from a FRESH draw, vs truth

Standard example case (piece 8, frame 64, block (1,4), +3 h). CPU ~1 min."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from latentsky.models import VelocityUNet, UNet2DTwoHead
from latentsky.layout import rain_from_z
from latentsky.models.flow import T_EMB_RANGE, RectifiedFlow

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"
INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
SCALE, SIGMA, NSTEPS = 2.0, 0.1876, 32

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
matplotlib.rcParams["savefig.bbox"] = None


def main():
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME], dtype=np.float32)[sl]
                  for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13], dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = VelocityUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
        map_location="cpu", weights_only=True))
    net.eval()
    rf = RectifiedFlow(sigma=SIGMA)


    with torch.no_grad():
        det_t, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        z1 = torch.from_numpy(np.log1p(truth)[None, None] / SCALE)
        g = torch.Generator().manual_seed(7)
        x0 = rf.source(zf, generator=g)
        xts = {t: ((1 - t) * x0 + t * z1).numpy()[0, 0] for t in (0.25, 0.5, 0.75)}
        v_target = (z1 - x0).numpy()[0, 0]
        x_half = (0.5 * x0 + 0.5 * z1)
        v_pred = net(x_half, cond, torch.full((1,), 0.5) * (T_EMB_RANGE - 1)
                     ).numpy()[0, 0]
        # sampling from a FRESH draw
        g2 = torch.Generator().manual_seed(11)
        xcur = rf.source(zf, generator=g2)
        snaps = {0.0: xcur.numpy()[0, 0].copy()}
        dt = 1.0 / NSTEPS
        for i in range(NSTEPS):
            t0v = torch.full((1,), i * dt)
            v0 = net(xcur, cond, t0v * (T_EMB_RANGE - 1))
            x_pred = xcur + dt * v0
            if i + 1 < NSTEPS:
                t1v = torch.full((1,), (i + 1) * dt)
                v1 = net(x_pred, cond, t1v * (T_EMB_RANGE - 1))
                xcur = xcur + dt * 0.5 * (v0 + v1)
            else:
                xcur = x_pred
            for mark in (0.5, 1.0):
                if abs((i + 1) * dt - mark) < dt / 2:
                    snaps[mark] = xcur.numpy()[0, 0].copy()

    zf0, x00 = zf.numpy()[0, 0], x0.numpy()[0, 0]
    vmax = max(truth.max(), 1.0)
    rnorm = LogNorm(vmin=0.1, vmax=vmax)
    rcmap = plt.get_cmap("Blues").copy(); rcmap.set_under("#ffffff")
    vlim = float(np.abs(v_target).max()) * 0.85

    ROWS = [
        ("STEP 1 · ONE TRAINING PAIR   —   source  x₀ = z_forecast + σ·ε   (σ = 0.188, measured),   target  x₁ = z_truth",
         [("two-head forecast", rain_from_z(zf0), "rain"),
          ("source  x₀  (t = 0)", rain_from_z(x00), "rain"),
          ("target  x₁  (t = 1)", truth, "rain"), None]),
        ("STEP 2 · THE STRAIGHT LINE   —   x_t = (1 − t)·x₀ + t·x₁ ;  training draws one random t per sample",
         [(f"x_t at t = {t:g}", rain_from_z(xts[t]), "rain") for t in (0.25, 0.5, 0.75)]
         + [None]),
        ("STEP 3 · THE REGRESSION   —   min ‖v̂(x_t, c, t) − (x₁ − x₀)‖²   →   v̂ → E[x₁ − x₀ | x_t, c]",
         [("target  v = x₁ − x₀", v_target, "v"),
          ("network  v̂(x_t, c, t = 0.5)", v_pred, "v"),
          ("residual  v − v̂", v_target - v_pred, "v"), None]),
        ("STEP 4 · SAMPLING   —   integrate  dx/dt = v̂(x, c, t)  from a FRESH draw x₀ (32 Heun steps; t = correction progress, not weather time)",
         [("fresh x₀  (t = 0)", rain_from_z(snaps[0.0]), "rain"),
          ("mid-ride  (t = 0.5)", rain_from_z(snaps[0.5]), "rain"),
          ("draw  x(1)", rain_from_z(snaps[1.0]), "rain"),
          ("truth  x₁ (reference)", truth, "rain")]),
    ]

    fig = plt.figure(figsize=(11.8, 13.6))
    gs = fig.add_gridspec(4, 4, left=0.015, right=0.90, top=0.965, bottom=0.01,
                          hspace=0.33, wspace=0.06)
    ims = {}
    for ri, (header, panels) in enumerate(ROWS):
        y = {0: 0.985, 1: 0.737, 2: 0.489, 3: 0.241}[ri]
        fig.text(0.02, y, header, fontsize=9.5, family="monospace",
                 color="#1c2026", va="top")
        for ci, pan in enumerate(panels):
            if pan is None:
                continue
            ttl, f, kind = pan
            ax = fig.add_subplot(gs[ri, ci])
            if kind == "rain":
                im = ax.imshow(f, origin="lower", norm=rnorm, cmap=rcmap,
                               interpolation="nearest")
                ims["rain"] = im
            else:
                im = ax.imshow(f, origin="lower", cmap="RdBu_r",
                               vmin=-vlim, vmax=vlim, interpolation="nearest")
                ims["v"] = im
            ax.set_title(ttl, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
    cax1 = fig.add_axes([0.92, 0.55, 0.015, 0.33])
    fig.colorbar(ims["rain"], cax=cax1).set_label("rain (mm)", fontsize=8)
    cax2 = fig.add_axes([0.92, 0.12, 0.015, 0.33])
    fig.colorbar(ims["v"], cax=cax2).set_label("velocity (z units), + = add rain",
                                               fontsize=8)
    fig.savefig(OUT / "flow_process.png", dpi=125)
    plt.close(fig)
    res = float(np.sqrt(np.mean((v_target - v_pred) ** 2)))
    tgt = float(np.sqrt(np.mean(v_target ** 2)))
    print(f"saved flow_process.png  (v rms {tgt:.3f}, residual {res:.3f}; "
          f"draw max {rain_from_z(snaps[1.0]).max():.1f} vs truth {truth.max():.1f})")


if __name__ == "__main__":
    main()
