"""Visualize the adopted 6 h flow on the standard example block (piece 8,
block (1,4), frame 64): rain now / two-head +6 h / three flow draws / truth,
plus the three-way error maps. Short GPU job; PNGs into
docs/assets_model_reference/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, SymLogNorm

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"
sys.path.insert(0, str(REPO / "docs"))
from make_model_reference_assets import INPUT_VARS, load_frame  # noqa: E402

P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
DELTA6 = 25
SCALE, SIGMA, NSTEPS = 2.0, 0.204, 32


def main():
    device = torch.device("cuda")
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]

    import make_model_reference_assets as mra
    mra.DELTA = DELTA6  # load_frame uses module-level DELTA
    x, y, y0 = load_frame(PIECE, FRAME)
    cond = torch.from_numpy(x[:, sl[0], sl[1]][None]).to(device)
    truth, persist = y[sl], y0[sl]

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead6h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()
    net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead6h.checkpoint.pt",
        map_location=device, weights_only=True))
    net = net.to(device).eval()
    rf = RectifiedFlow(sigma=SIGMA)

    with torch.no_grad():
        det, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
        draws = []
        for s in range(3):
            g = torch.Generator(device=device).manual_seed(10 + s)
            z = rf.sample(net, cond, zf, n_steps=NSTEPS, generator=g)
            draws.append(np.clip(np.expm1(SCALE * np.clip(z.cpu().numpy()[0, 0], 0, 2.5)), 0, None))
    det_mm = np.clip(det.cpu().numpy()[0, 0], 0, None)

    plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
    vmax = max(truth.max(), det_mm.max(), max(d.max() for d in draws), 0.5)
    norm = LogNorm(vmin=0.1, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")

    def panel(ax, f, title, color="0.2"):
        im = ax.imshow(f, norm=norm, cmap=cmap, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=8, color=color)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(color); sp.set_linewidth(2.0)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    fig, axes = plt.subplots(1, 6, figsize=(13.8, 2.4))
    panel(axes[0], persist, "rain now (persistence)", "#7a7a7a")
    panel(axes[1], det_mm, "two-head (+6 h)", "#2166ac")
    for j in range(3):
        panel(axes[2 + j], draws[j], f"flow draw {j + 1} (+6 h)", "#b2182b")
    im = panel(axes[5], truth, "truth (+6 h)", "#1a1a1a")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("rain (mm)", fontsize=8)
    fig.savefig(OUT / "flow6h_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    e_det, e_flow, e_ref = truth - det_mm, truth - draws[0], draws[0] - det_mm
    lim = max(np.abs(e_det).max(), np.abs(e_flow).max(), np.abs(e_ref).max(), 0.5)
    snorm = SymLogNorm(linthresh=0.1, vmin=-lim, vmax=lim)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1))
    for ax, e, title, color in [
        (axes[0], e_det, "truth − two-head (+6 h)", "#2166ac"),
        (axes[1], e_flow, "truth − flow draw 1 (+6 h)", "#b2182b"),
        (axes[2], e_ref, "flow draw 1 − two-head (+6 h)", "#1a1a1a"),
    ]:
        im = ax.imshow(e, origin="lower", cmap="RdBu", norm=snorm, interpolation="nearest")
        ax.set_title(title, fontsize=8, color=color)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(color); sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Δ rain (mm)", fontsize=8)
    fig.savefig(OUT / "flow6h_errors.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(dict(truth_max=float(truth.max()), det_max=float(det_mm.max()),
               draw_max=[float(d.max()) for d in draws],
               truth_wet=float((truth > 0.1).mean()),
               det_wet=float((det_mm > 0.1).mean()),
               draw_wet=[float((d > 0.1).mean()) for d in draws]))


if __name__ == "__main__":
    main()
