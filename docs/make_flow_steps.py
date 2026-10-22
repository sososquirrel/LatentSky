"""Six small strips (flow_step1..6.png) telling the flow-matching method as a
picture book on the standard +3 h example. Shared scales, fixed canvases.

  1  the problem: forecast vs truth
  2  sprinkle noise on the forecast (source x0)
  3  the crossfade: x_t at t = 0.25 / 0.5 / 0.75
  4  the one question: direction to truth (target v, network v-hat)
  5  forecast time: ride 32 small steps from fresh noise
  6  two different sprinkles, two plausible truths (draw A, draw B, truth)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from latentsky.models import VelocityUNet, UNet2DTwoHead
from latentsky.layout import (  # noqa: F401  (re-exported)
    EXAMPLE_BLOCK, EXAMPLE_FRAME, EXAMPLE_PIECE, INPUT_VARS, LOG_RAIN_SCALE, P, PIECES, RUNS,
    rain_from_z)
from latentsky.models.flow import T_EMB_RANGE, RectifiedFlow, flow_config  # noqa: F401

REPO = Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "docs" / "assets_model_reference"
P, PIECE, FRAME, BLOCK = P, EXAMPLE_PIECE, EXAMPLE_FRAME, EXAMPLE_BLOCK
SCALE = LOG_RAIN_SCALE
_CFG = flow_config(3)
SIGMA, NSTEPS = _CFG["sigma"], _CFG["n_steps"]

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
matplotlib.rcParams["savefig.bbox"] = None


def sample_traj(rf, net, cond, zf, seed):
    """Fields at flow times 0, 0.5 and 1 of one draw (numpy, block only)."""
    snaps = rf.sample_path(net, cond, zf, n_steps=NSTEPS, snap_at=(0.0, 0.5, 1.0),
                           generator=torch.Generator().manual_seed(seed))
    return {m: v.numpy()[0, 0].copy() for m, v in snaps.items()}


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
        x_half = 0.5 * x0 + 0.5 * z1
        v_pred = net(x_half, cond,
                     torch.full((1,), 0.5) * (T_EMB_RANGE - 1)).numpy()[0, 0]
        trajA = sample_traj(rf, net, cond, zf, 11)
        trajB = sample_traj(rf, net, cond, zf, 12)

    zf0, x00 = zf.numpy()[0, 0], x0.numpy()[0, 0]
    vmax = max(truth.max(), 1.0)
    rnorm = LogNorm(vmin=0.1, vmax=vmax)
    rcmap = plt.get_cmap("Blues").copy(); rcmap.set_under("#ffffff")
    vlim = float(np.abs(v_target).max()) * 0.85

    def strip(panels, fn):
        n = len(panels)
        fig = plt.figure(figsize=(2.35 * n + 0.25, 2.55))
        gs = fig.add_gridspec(1, n, left=0.01, right=0.99, top=0.86,
                              bottom=0.02, wspace=0.05)
        for k, (ttl, f, kind) in enumerate(panels):
            ax = fig.add_subplot(gs[0, k])
            if kind == "rain":
                ax.imshow(f, origin="lower", norm=rnorm, cmap=rcmap,
                          interpolation="nearest")
            else:
                ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=-vlim,
                          vmax=vlim, interpolation="nearest")
            ax.set_title(ttl, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(OUT / fn, dpi=135)
        plt.close(fig)
        print("saved", fn)

    strip([("our forecast for +3 h", rain_from_z(zf0), "rain"),
           ("what actually happened", truth, "rain")], "flow_step1.png")
    strip([("the forecast", rain_from_z(zf0), "rain"),
           ("+ a sprinkle of noise = x₀", rain_from_z(x00), "rain")],
          "flow_step2.png")
    strip([("25 % of the way", rain_from_z(xts[0.25]), "rain"),
           ("50 %", rain_from_z(xts[0.5]), "rain"),
           ("75 %", rain_from_z(xts[0.75]), "rain")], "flow_step3.png")
    strip([("the right answer:  truth − x₀", v_target, "v"),
           ("the network's answer", v_pred, "v")], "flow_step4.png")
    strip([("start: fresh noise on the forecast", rain_from_z(trajA[0.0]), "rain"),
           ("after 16 of 32 steps", rain_from_z(trajA[0.5]), "rain"),
           ("after 32 steps: the draw", rain_from_z(trajA[1.0]), "rain")],
          "flow_step5.png")
    strip([("draw A (one sprinkle)", rain_from_z(trajA[1.0]), "rain"),
           ("draw B (another sprinkle)", rain_from_z(trajB[1.0]), "rain"),
           ("the truth", truth, "rain")], "flow_step6.png")
    print("draw maxima:", round(float(rain_from_z(trajA[1.0]).max()), 1),
          round(float(rain_from_z(trajB[1.0]).max()), 1),
          "truth", round(float(truth.max()), 1))


if __name__ == "__main__":
    main()
