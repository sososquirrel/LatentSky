"""Generate the real-example panels for docs/model_reference.html.

Picks the rainiest held-out test block (patch split, seed 1919810) at lead 1 h,
runs the canonical two-head forecast, then the warm-started reverse diffusion
chain (t_start=100), saving the intermediate states. Two PNGs into
docs/assets_model_reference/. Run on a GPU node (101 denoiser passes).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from latentsky.models import ConditionalDiffusionUNet, GaussianDiffusion, UNet2DTwoHead

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"

INPUT_VARS = [
    "precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
    "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
    "umeridional_850hPa", "rain_log1p",
]
P = 128
LEAD_STEPS = 12  # 3 h
DELTA = LEAD_STEPS + 1  # off-by-one correction, see model_reference §1
SCALE = 2.0
T_START = 100
CLIP = (0.0, 2.5)
SNAP_TS = [100, 60, 30, 10]  # states to save along the reverse chain


def patch_test_blocks():
    # same math as _archive/latentsky/patch_split.py (seed 1919810, 70/15/15)
    import random

    H, W = 320, 1440
    nrows, ncols = H // P, W // P
    blocks = [(p, r, c) for p in range(40) for r in range(nrows) for c in range(ncols)]
    rng = random.Random(1919810)
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = int(round(n * 0.7))
    n_val = int(round(n * 0.15))
    return set(blocks[n_train + n_val:])


def pick_patch(test_blocks):
    """Rainiest (truth wet fraction) test block over a stride of frames of a
    few pieces. Deterministic."""
    best = None
    for pid in (1, 8, 20, 33):
        y = np.load(PIECES / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")
        blocks = [(r, c) for (p, r, c) in test_blocks if p == pid]
        for t in range(0, 96 - DELTA, 8):
            yt = np.asarray(y[t + DELTA], dtype=np.float32)
            for r, c in blocks:
                patch = yt[r * P:(r + 1) * P, c * P:(c + 1) * P]
                wet = float((patch > 0.1).mean())
                score = wet * float(patch.max() > 1.0)
                if best is None or score > best[0]:
                    best = (score, pid, t, r, c)
    return best[1:]


def load_frame(pid, t):
    chans = []
    for v in INPUT_VARS:
        a = np.load(PIECES / f"{v}_piece{pid:02d}.npy", mmap_mode="r")
        chans.append(np.asarray(a[t], dtype=np.float32))
    x = np.stack(chans, axis=0)
    y = np.asarray(
        np.load(PIECES / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")[t + DELTA],
        dtype=np.float32,
    )
    y0 = np.asarray(
        np.load(PIECES / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")[t + 1],
        dtype=np.float32,
    )  # rain simultaneous with the inputs (the +1 raw offset): the persistence field
    return x, y, y0


def main():
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    test_blocks = patch_test_blocks()
    pid, t, r, c = pick_patch(test_blocks)
    print(f"chosen: piece {pid}, frame {t}, block ({r},{c})")

    x, y, y0 = load_frame(pid, t)
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    cond = torch.from_numpy(x[:, sl[0], sl[1]][None]).to(device)
    truth = y[sl]
    persist = y0[sl]

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()

    diff = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    diff.load_state_dict(torch.load(
        RUNS / "gen_models/10var/diffusion_lead3h/checkpoint.pt",
        map_location=device, weights_only=True))
    diff = diff.to(device).eval()
    gd = GaussianDiffusion(1000, "cosine", device=device)

    with torch.no_grad():
        det_rain, _ = twohead(cond)
        x0_init = torch.log1p(torch.clamp(det_rain, min=0.0)) / SCALE

        # one reverse chain with snapshots (same math as ddpm_refine_sample)
        torch.manual_seed(0)
        tb = torch.full((1,), T_START, device=device, dtype=torch.long)
        xstate = gd.q_sample(x0_init, tb, torch.randn_like(x0_init))
        snaps = {}
        snaps_x0 = {}
        for tt in reversed(range(T_START + 1)):
            if tt in SNAP_TS:
                snaps[tt] = xstate.clone()
            tb = torch.full((1,), tt, device=device, dtype=torch.long)
            eps = diff(xstate, cond, tb)
            ab = gd.alphas_cumprod[tt]
            x0p = torch.clamp((xstate - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab), *CLIP)
            if tt in SNAP_TS:
                snaps_x0[tt] = x0p.clone()
            mean = gd.posterior_mean_coef1[tt] * x0p + gd.posterior_mean_coef2[tt] * xstate
            if tt > 0:
                xstate = mean + torch.sqrt(gd.posterior_variance[tt]) * torch.randn_like(xstate)
            else:
                xstate = mean
        draw0 = xstate.clone()

        # two more draws for the ensemble mean
        draws = [draw0]
        for seed in (1, 2):
            torch.manual_seed(seed)
            draws.append(gd.ddpm_refine_sample(diff, cond, x0_init, T_START, CLIP, 1.0))

    def to_mm(z):
        return np.clip(np.expm1(SCALE * np.clip(z.cpu().numpy()[0, 0], 0.0, 2.5)), 0, None)

    det_mm = np.clip(det_rain.cpu().numpy()[0, 0], 0, None)
    final_mm = to_mm(draw0)
    ens_mm = np.mean([to_mm(d) for d in draws], axis=0)
    snap_mm = {tt: to_mm(v) for tt, v in snaps.items()}

    vmax = max(truth.max(), det_mm.max(), final_mm.max(), 0.5)
    norm = LogNorm(vmin=0.1, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#f7f9fb")
    plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))

    def panel(ax, field, title):
        im = ax.imshow(field, norm=norm, cmap=cmap, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    # figure 1: context
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panel(axes[0], persist, "rain now (persistence)")
    panel(axes[1], det_mm, "two-head forecast (+3 h)")
    im = panel(axes[2], truth, "truth (+3 h)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("rain (mm)", fontsize=8)
    fig.savefig(OUT / "example_context.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # figure 2: denoising trajectory — row 1 the chain state x_t, row 2 the
    # model's running x0 estimate, both anchored at the two-head forecast.
    snap_x0_mm = {tt: to_mm(v) for tt, v in snaps_x0.items()}
    ts_sorted = sorted(snap_mm, reverse=True)
    ncol = len(ts_sorted) + 2
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.4))
    panel(axes[0, 0], det_mm, "two-head forecast\n(warm-start input)")
    for j, tt in enumerate(ts_sorted, start=1):
        panel(axes[0, j], snap_mm[tt], f"state $x_t$, t={tt}")
    im = panel(axes[0, ncol - 1], final_mm, "draw at t=0")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "model's running\nestimate of the rain\n" r"$\hat{x}_0(t)\ \rightarrow$",
                    ha="center", va="center", fontsize=8, transform=axes[1, 0].transAxes)
    for j, tt in enumerate(ts_sorted, start=1):
        panel(axes[1, j], snap_x0_mm[tt], f"$\\hat{{x}}_0$ at t={tt}")
    im = panel(axes[1, ncol - 1], to_mm(draws[1]), "draw 2 (t=0)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015)
    cb.set_label("rain (mm)", fontsize=8)
    fig.savefig(OUT / "example_denoise.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # figure 3: final side-by-side comparison — three individual draws, never
    # the mean (FRAME ensemble scoring rule)
    dmm = [to_mm(d) for d in draws]
    fig, axes = plt.subplots(1, 6, figsize=(13.8, 2.4))
    panel(axes[0], persist, "rain now (persistence)")
    panel(axes[1], det_mm, "two-head (+3 h)")
    for j in range(3):
        panel(axes[2 + j], dmm[j], f"diffusion draw {j + 1} (+3 h)")
    im = panel(axes[5], truth, "truth (+3 h)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("rain (mm)", fontsize=8)
    fig.savefig(OUT / "example_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(OUT / "example_fields.npz", persist=persist, truth=truth,
                        det=det_mm, final=final_mm, ens=ens_mm,
                        **{f"state_{tt}": v for tt, v in snap_mm.items()})

    info = dict(piece=pid, frame=t, block=(r, c),
                truth_max=float(truth.max()), det_max=float(det_mm.max()),
                final_max=float(final_mm.max()), vmax=float(vmax))
    (OUT / "example_info.txt").write_text(repr(info) + "\n")
    print(info)


if __name__ == "__main__":
    main()
