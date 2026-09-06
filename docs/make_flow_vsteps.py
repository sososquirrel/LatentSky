"""flow_step5b.png: the network's velocity answer v-hat along the ACTUAL
32-step ride of the step-5 draw (seed 11) - five evaluations at t = 0, 1/4,
1/2, 3/4 and the last step, on one shared scale, plus the mean of all 32
answers, which by construction equals the total correction x(1) - x0.
Standard +3 h example. CPU ~30 s."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
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
    cond = torch.from_numpy(x[None])
    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
        map_location="cpu", weights_only=True))
    net.eval()
    rf = RectifiedFlow(sigma=SIGMA)

    with torch.no_grad():
        det_t, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        g = torch.Generator().manual_seed(11)   # the SAME ride as step 5
        xcur = rf.source(zf, generator=g)
        x_start = xcur.numpy()[0, 0].copy()
        vs = []
        dt = 1.0 / NSTEPS
        for i in range(NSTEPS):
            t0 = torch.full((1,), i * dt)
            v0 = net(xcur, cond, t0 * (T_EMB_RANGE - 1))
            vs.append(v0.numpy()[0, 0].copy())
            x_pred = xcur + dt * v0
            if i + 1 < NSTEPS:
                t1 = torch.full((1,), (i + 1) * dt)
                v1 = net(x_pred, cond, t1 * (T_EMB_RANGE - 1))
                xcur = xcur + dt * 0.5 * (v0 + v1)
            else:
                xcur = x_pred
        x_end = xcur.numpy()[0, 0]

    mean_v = np.mean(vs, axis=0)
    diff = vs[31] - vs[0]
    vlim = 0.5
    dlim = float(np.abs(diff).max()) * 0.7
    panels = [(f"answer at step 1 (t = 0)", vs[0], vlim),
              (f"answer at step 17 (t = 0.5)", vs[16], vlim),
              (f"answer at step 32 (t = 0.97)", vs[31], vlim),
              ("change: step 32 − step 1\n(own scale ±%.2f)" % dlim, diff, dlim),
              ("mean of all 32 answers\n≈ x(1) − x₀", mean_v, vlim)]
    n = len(panels)
    fig = plt.figure(figsize=(2.35 * n + 0.7, 2.85))
    gs = fig.add_gridspec(1, n, left=0.008, right=0.93, top=0.80, bottom=0.02,
                          wspace=0.05)
    for k, (ttl, f, lim) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")
        ax.set_title(ttl, fontsize=8.5,
                     fontweight="bold" if k == n - 1 else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            im0 = im
    cax = fig.add_axes([0.945, 0.1, 0.011, 0.66])
    fig.colorbar(im0, cax=cax).set_label("v̂ (z units)", fontsize=7)
    fig.savefig(OUT / "flow_step5b.png", dpi=135)
    plt.close(fig)

    # numbers for the caption: step-to-step change and identity check
    d01 = float(np.sqrt(np.mean((vs[8] - vs[0]) ** 2)))
    rms0 = float(np.sqrt(np.mean(vs[0] ** 2)))
    ident = float(np.sqrt(np.mean((mean_v - (x_end - x_start)) ** 2)))
    print(f"saved flow_step5b.png; rms v(0) {rms0:.3f}, rms[v(8)-v(0)] {d01:.3f}, "
          f"|mean_v - (x1-x0)| rms {ident:.4f}")


if __name__ == "__main__":
    main()
