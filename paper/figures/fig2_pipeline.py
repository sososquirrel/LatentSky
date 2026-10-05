"""Paper Fig 2 — the forecast pipeline in one schematic (after Sophie's sketch).

Left to right: the ten input fields at time t -> the two-head U-Net (encoder,
bottleneck, decoder, dashed skip connections) -> its rain forecast at
t + dt -> the flow: the forecast plus noise is carried to a sharp field by
repeated steps of one velocity network, and different noise gives different
draws. All rain maps are real fields of the standard example block at +3 h
on one log colour scale.

Two stages: `compute()` runs the two models once (CPU, a few minutes) and
caches the fields in data/paper_figs/fig2_pipeline.npz; `draw()` only reads
the cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "docs"))
sys.path.insert(0, str(REPO / "paper" / "figures"))
from paperfig import COLOR, W2  # noqa: E402

OUT = REPO / "paper" / "figures"
CACHE = REPO / "data" / "paper_figs" / "fig2_pipeline.npz"
SNAP_T = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (11, 12, 13)


def compute():
    import torch
    from make_flow_steps import (BLOCK, FRAME, INPUT_VARS, NSTEPS, P, PIECE,
                                 PIECES, RUNS, SCALE, SIGMA)

    from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
    from latentsky.models.flow import T_EMB_RANGE, RectifiedFlow

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

    def to_rain(z):
        return np.clip(np.expm1(SCALE * np.clip(z, 0, 2.5)), 0, None)

    out = dict(inputs=x, truth=truth, input_vars=np.array(INPUT_VARS))
    dt = 1.0 / NSTEPS
    with torch.no_grad():
        det_t, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        out["forecast"] = to_rain(zf.numpy()[0, 0])
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            xc = rf.source(zf, generator=g)
            snaps = {0.0: xc.numpy()[0, 0].copy()}
            # Heun, as in the paper: last step plain Euler
            for i in range(NSTEPS):
                t0 = torch.full((1,), i * dt)
                v0 = net(xc, cond, t0 * (T_EMB_RANGE - 1))
                xp = xc + dt * v0
                if i + 1 < NSTEPS:
                    t1 = torch.full((1,), (i + 1) * dt)
                    v1 = net(xp, cond, t1 * (T_EMB_RANGE - 1))
                    xc = xc + dt * 0.5 * (v0 + v1)
                else:
                    xc = xp
                for m in SNAP_T[1:]:
                    if abs((i + 1) * dt - m) < dt / 2:
                        snaps[m] = xc.numpy()[0, 0].copy()
            if seed == SEEDS[0]:
                out["path"] = np.stack([to_rain(snaps[m]) for m in SNAP_T])
            out[f"draw{seed}"] = to_rain(snaps[1.0])
            print("seed", seed, "done", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print("cached", CACHE)


CMAPS = {"precipw": "BrBG", "mse_500hPa": "BrBG", "mse_850hPa": "BrBG",
         "t2m": "plasma", "olrtoa": "gray", "uzonal_500hPa": "RdBu_r",
         "uzonal_850hPa": "RdBu_r", "umeridional_500hPa": "RdBu_r",
         "umeridional_850hPa": "RdBu_r", "rain_log1p": "Blues"}
BLUE, ORANGE = COLOR["two-head"], COLOR["flow"]
BLUE_BG, ORANGE_BG, INK, GRAY = "#dfe9f4", "#fdf0e6", "0.2", "0.45"


def draw():
    d = np.load(CACHE, allow_pickle=False)
    norm = LogNorm(0.1, max(float(d["truth"].max()), 1.0))
    rcmap = plt.get_cmap("Blues").copy()
    rcmap.set_under("#ffffff")
    rcmap.set_bad("#ffffff")

    fig, ax = plt.subplots(figsize=(W2, W2 * 0.53))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 53)
    ax.axis("off")

    def arrow(p0, p1, color=INK, lw=0.9, ls="-", ms=7, rad=0.0, style="-|>"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                     color=color, lw=lw, ls=ls, zorder=6,
                                     shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={rad}"))

    def rain(f, x0, y0, s, title=None, ec="0.25", lw=0.8, tag=None):
        ax.imshow(f, origin="lower", norm=norm, cmap=rcmap, aspect="auto",
                  extent=(x0, x0 + s, y0, y0 + s), interpolation="nearest",
                  zorder=3)
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec=ec, lw=lw, zorder=4))
        if title:
            ax.text(x0 + s / 2, y0 + s + 0.7, title, ha="center", va="bottom",
                    fontsize=6.5)
        if tag:
            ax.text(x0 + 0.4, y0 + s - 0.4, tag, ha="left", va="top", fontsize=5.5,
                    zorder=5, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))

    def frame(x0, y0, w, h, ec, fc):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9, zorder=1))

    # ---------------- inputs: ten fields as a stack of sheets -----------------
    S_IN, n = 9.5, len(d["input_vars"])
    for k, v in enumerate(d["input_vars"]):
        f = d["inputs"][k]
        lo, hi = np.percentile(f, [2, 98])
        x0, y0 = 0.6 + 0.48 * k, 40.2 - 0.95 * k - S_IN
        last = k == n - 1
        ax.imshow(f, origin="lower", cmap=CMAPS[str(v)], vmin=lo, vmax=hi, aspect="auto",
                  extent=(x0, x0 + S_IN, y0, y0 + S_IN), interpolation="nearest",
                  zorder=3 + 0.01 * k)
        ax.add_patch(Rectangle((x0, y0), S_IN, S_IN, fill=False,
                               ec=BLUE if last else "0.35", lw=1.2 if last else 0.5,
                               zorder=3.005 + 0.01 * k))
    ax.text(7.5, 42.2, "ten fields at time $t$", ha="center", va="bottom", fontsize=6.5)
    ax.text(7.5, 19.6, "large-scale state\nand current rain", ha="center", va="top",
            fontsize=6, color=GRAY, linespacing=1.15)
    in_right = (0.6 + 0.48 * (n - 1) + S_IN, 40.2 - 0.95 * (n - 1) - S_IN / 2)

    # ---------------- the U-Net ----------------------------------------------
    frame(17.0, 4.5, 43.5, 47.0, "0.6", "#fafafa")
    ax.text(38.75, 6.0, r"U-Net:  fields at $t$  $\rightarrow$  rain at $t + \Delta t$",
            ha="center", va="bottom", fontsize=7.5)
    W = 2.3
    xe = [19.5, 23.0, 26.5, 30.0, 33.5]
    xd = [55.7, 52.2, 48.7, 45.2, 41.7]
    yc = [36.5, 32.0, 28.0, 24.5, 21.5]
    hh = [17.0, 13.0, 9.5, 6.8, 4.6]
    res = ["128²", "64²", "32²", "16²", "8²"]
    for i in range(5):
        for x in (xe[i], xd[i]):
            ax.add_patch(Rectangle((x, yc[i] - hh[i] / 2), W, hh[i], fc=BLUE_BG, ec=BLUE,
                                   lw=0.8, zorder=3))
        top = yc[i] + hh[i] / 2
        arrow((xe[i] + W + 0.3, top - 0.9), (xd[i] - 0.3, top - 0.9), color="0.5",
              lw=0.7, ls=(0, (4, 3)), ms=6)
        ax.text(xe[i] + W / 2, yc[i] - hh[i] / 2 - 0.5, res[i], ha="center", va="top",
                fontsize=5, color=GRAY)
        if i < 4:
            arrow((xe[i] + W, yc[i] - hh[i] / 2 + 1.0), (xe[i + 1], yc[i + 1]), ms=5, lw=0.7)
            arrow((xd[i + 1] + W, yc[i + 1]), (xd[i], yc[i] - hh[i] / 2 + 1.0), ms=5, lw=0.7)
    # bottleneck
    xb0, xb1, yb = 36.4, 41.1, 13.2
    ax.add_patch(Rectangle((xb0, yb), xb1 - xb0, 2.6, fc="#b9cfe6", ec=BLUE, lw=0.8, zorder=3))
    ax.text((xb0 + xb1) / 2, yb - 0.5, "bottleneck 4²", ha="center", va="top", fontsize=5,
            color=GRAY)
    arrow((xe[4] + W, yc[4] - 1.2), (xb0, yb + 2.0), ms=5, lw=0.7)
    arrow((xb1, yb + 2.0), (xd[4], yc[4] - 1.2), ms=5, lw=0.7)
    ax.text(38.75, yc[0] + hh[0] / 2 + 0.2, "skip connections", ha="center", va="bottom",
            fontsize=5.5, color="0.5")
    ax.text(xe[2] - 1.2, 16.6, "encoder", ha="center", fontsize=6, color=BLUE)
    ax.text(xd[2] + W + 1.2, 16.6, "decoder", ha="center", fontsize=6, color=BLUE)
    arrow(in_right, (xe[0], in_right[1]))

    # ---------------- U-Net forecast, then the flow ---------------------------
    S, YC = 7.6, 36.5
    y0 = YC - S / 2
    frame(71.2, 4.5, 28.4, 47.0, ORANGE, ORANGE_BG)
    ax.text(85.4, 6.0, "flow matching", ha="center", va="bottom", fontsize=7.5,
            color=ORANGE)
    xf, x0s, xm, xdr = 62.2, 72.6, 82.0, 91.2
    rain(d["forecast"], xf, y0, S, "U-Net forecast", ec=BLUE, lw=1.1)
    arrow((xd[0] + W, YC), (xf, YC))
    rain(d["path"][0], x0s, y0, S, "$x_0$")
    rain(d["path"][2], xm, y0, S, "$x_{0.5}$")
    arrow((xf + S, YC), (x0s, YC), color=ORANGE)
    ax.text(x0s + S / 2, y0 - 0.6, r"forecast $+\,\sigma\varepsilon$", ha="center",
            va="top", fontsize=5.5, color=ORANGE)
    arrow((x0s + S, YC), (xm, YC), color=ORANGE)
    # three draws, three noise seeds
    ys = [YC + S + 0.9 - S / 2, YC - S / 2, YC - S - 0.9 - S / 2]
    for y, seed, tag in zip(ys, SEEDS, "ABC"):
        rain(d[f"draw{seed}"], xdr, y, S, tag=f"draw {tag}")
        arrow((xm + S, YC), (xdr, y + S / 2), color=ORANGE, lw=0.8, ms=6)
    ax.text(xdr + S / 2, ys[0] + S + 0.5, "$x_1$", ha="center", va="bottom", fontsize=6.5)
    # the velocity network, applied at every step
    bx0, bx1, by0, by1 = 72.6, 89.6, 12.0, 19.6
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                                boxstyle="round,pad=0,rounding_size=0.8", fc="white",
                                ec=ORANGE, lw=1.0, zorder=3))
    ax.text((bx0 + bx1) / 2, (by0 + by1) / 2 + 1.4, r"velocity network $\hat{v}(x_t, t, c)$",
            ha="center", va="center", fontsize=6, color=ORANGE)
    ax.text((bx0 + bx1) / 2, (by0 + by1) / 2 - 1.5,
            r"$x_{t+\delta} = x_t + \hat{v}\,\delta$,  32 steps", ha="center", va="center",
            fontsize=6)
    xl = (x0s + S + xm) / 2
    arrow((xl - 0.9, YC - 0.6), (xl - 0.9, by1), color=ORANGE, lw=0.8, ms=6)
    arrow((xl + 0.9, by1), (xl + 0.9, YC - 0.6), color=ORANGE, lw=0.8, ms=6)
    ax.text(xl - 1.5, 26.5, "$x_t$", ha="right", va="center", fontsize=6.5)
    ax.text(xl + 1.5, 26.5, r"$\hat{v}$", ha="left", va="center", fontsize=6.5, color=ORANGE)
    # conditioning: the same ten fields
    yl = 2.2
    ax.plot([7.5, 7.5, (bx0 + bx1) / 2], [14.6, yl, yl], color="0.45", lw=0.8,
            ls=(0, (4, 3)), zorder=6)
    arrow(((bx0 + bx1) / 2, yl), ((bx0 + bx1) / 2, by0), color="0.45", lw=0.8,
          ls=(0, (4, 3)), ms=6)
    ax.text(40, yl + 0.5, "conditioning $c$: the same ten fields", ha="center", va="bottom",
            fontsize=6, color=GRAY)
    # truth, for reference
    rain(d["truth"], xdr, 12.0, S, tag="truth")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "fig2_pipeline.pdf")
    print("saved fig2_pipeline.pdf")


if __name__ == "__main__":
    if "--compute" in sys.argv or not CACHE.exists():
        compute()
    draw()
