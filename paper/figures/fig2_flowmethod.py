"""Paper Fig 2 — the rectified flow on one example (+3 h, standard block).

Top row, the result:      (a) forecast  (b) truth  (c) draw A  (d) draw B
Bottom row, the mechanism: (e) x0 = forecast + sigma*eps  (f) crossfade t=0.5
                           (g) target v = truth - x0      (h) network v-hat

Caption sentence: the smooth conditional-mean forecast becomes sharp sample
realizations; the network's one job is the constant crossfade speed, and it
answers it. Reuses make_flow_steps' computation (CPU, ~2 min).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from paperfig import W2

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "docs"))
from make_flow_steps import (  # noqa: E402
    BLOCK, FRAME, INPUT_VARS, P, PIECE, PIECES, RUNS, SCALE, SIGMA,
    sample_traj)

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead  # noqa: E402
from latentsky.models.flow import T_EMB_RANGE, RectifiedFlow  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.rcParams["savefig.bbox"] = "tight"  # make_flow_steps unsets it

OUT = REPO / "paper" / "figures"


def compute():
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME],
                             dtype=np.float32)[sl] for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13],
                       dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])
    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5,
                                   time_emb_dim=128)
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
        x_half = (0.5 * x0 + 0.5 * z1).numpy()[0, 0]
        v_target = (z1 - x0).numpy()[0, 0]
        v_pred = net(0.5 * x0 + 0.5 * z1, cond,
                     torch.full((1,), 0.5) * (T_EMB_RANGE - 1)).numpy()[0, 0]
        trajA = sample_traj(rf, net, cond, zf, 11)
        trajB = sample_traj(rf, net, cond, zf, 12)
    return dict(zf=zf.numpy()[0, 0], x0=x0.numpy()[0, 0], truth=truth,
                x_half=x_half, v_target=v_target, v_pred=v_pred,
                drawA=trajA[1.0], drawB=trajB[1.0])


def main():
    d = compute()

    def to_rain(z):
        return np.clip(np.expm1(SCALE * np.clip(z, 0, 2.5)), 0, None)

    vmax = max(d["truth"].max(), 1.0)
    rnorm = LogNorm(vmin=0.1, vmax=vmax)
    rcmap = plt.get_cmap("Blues").copy()
    rcmap.set_under("#ffffff")
    vlim = float(np.abs(d["v_target"]).max()) * 0.85

    panels = [
        ("(a)", "forecast (two-head)", to_rain(d["zf"]), "rain"),
        ("(b)", "truth", d["truth"], "rain"),
        ("(c)", "flow draw A", to_rain(d["drawA"]), "rain"),
        ("(d)", "flow draw B", to_rain(d["drawB"]), "rain"),
        ("(e)", "x₀ = forecast + σ·ε", to_rain(d["x0"]), "rain"),
        ("(f)", "crossfade x₀→truth, t = 0.5", to_rain(d["x_half"]), "rain"),
        ("(g)", "target v = truth − x₀", d["v_target"], "v"),
        ("(h)", "network v̂ at t = 0.5", d["v_pred"], "v"),
    ]

    fig = plt.figure(figsize=(W2, W2 * 0.62))
    gs = fig.add_gridspec(2, 4, left=0.005, right=0.995, top=0.97,
                          bottom=0.17, hspace=0.14, wspace=0.04)
    im_rain = im_v = None
    axes = []
    for k, (lab, name, f, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        axes.append(ax)
        if kind == "rain":
            im_rain = ax.imshow(f, origin="lower", norm=rnorm, cmap=rcmap,
                                interpolation="nearest")
        else:
            im_v = ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=-vlim,
                             vmax=vlim, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.03, 0.97, f"{lab} {name}", transform=ax.transAxes,
                fontsize=7.5, ha="left", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    cax1 = fig.add_axes([0.06, 0.075, 0.5, 0.022])
    cb1 = fig.colorbar(im_rain, cax=cax1, orientation="horizontal")
    cb1.set_label("rain (mm, log scale)", fontsize=8)
    cb1.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb1.outline.set_linewidth(0.5)
    cax2 = fig.add_axes([0.65, 0.075, 0.3, 0.022])
    cb2 = fig.colorbar(im_v, cax=cax2, orientation="horizontal")
    cb2.set_label("velocity v̂ (z units)", fontsize=8)
    cb2.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb2.outline.set_linewidth(0.5)

    fig.savefig(OUT / "fig2_flowmethod.pdf", dpi=300)
    print("saved fig2_flowmethod.pdf")


if __name__ == "__main__":
    main()
