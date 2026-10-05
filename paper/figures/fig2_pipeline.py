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
from paperfig import W2  # noqa: E402

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
# muted versions of the paper's two model colours (blue = U-Net, warm = flow)
BLUE, ORANGE = "#35526e", "#a8553a"
BLUE_BG, ORANGE_BG, INK, GRAY = "#eef1f4", "#f7eeea", "0.2", "0.45"


def draw():
    d = np.load(CACHE, allow_pickle=False)
    norm = LogNorm(0.1, max(float(d["truth"].max()), 1.0))
    rcmap = plt.get_cmap("Blues").copy()
    rcmap.set_under("#ffffff")
    rcmap.set_bad("#ffffff")

    # the paper's text font is Computer Modern: use it here too
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(10.5, 55.5)
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
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=fc, ec=ec, lw=0.5, zorder=1))

    def panel(x, letter, title):
        ax.text(x, 52.4, letter, ha="left", va="bottom", fontsize=9, fontfamily="cmb10")
        ax.text(x + 2.6, 52.4, title, ha="left", va="bottom", fontsize=8)

    YC = 38.0   # the level of the U-Net's input and output

    # ---------------- inputs: nine state fields stacked, the rain at t below ---
    S_IN = 9.0
    names = [str(v) for v in d["input_vars"]]
    state = [k for k, v in enumerate(names) if v != "rain_log1p"]
    for j, k in enumerate(state):
        f = d["inputs"][k]
        lo, hi = np.percentile(f, [2, 98])
        x0, y0 = 0.6 + 0.5 * j, 50.6 - 0.8 * j - S_IN
        ax.imshow(f, origin="lower", cmap=CMAPS[names[k]], vmin=lo, vmax=hi, aspect="auto",
                  extent=(x0, x0 + S_IN, y0, y0 + S_IN), interpolation="nearest",
                  zorder=3 + 0.01 * j)
        ax.add_patch(Rectangle((x0, y0), S_IN, S_IN, fill=False, ec="0.35", lw=0.5,
                               zorder=3.005 + 0.01 * j))
    ax.text(7.2, 39.0 - 0.8 * 8 - 0.2, "state at time $t$", ha="center", va="top", fontsize=6.5)
    xr, yr = 2.6, 16.0
    f = d["inputs"][names.index("rain_log1p")]
    ax.imshow(f, origin="lower", cmap="Blues", vmin=0, vmax=np.percentile(f, 99.5), aspect="auto",
              extent=(xr, xr + S_IN, yr, yr + S_IN), interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((xr, yr), S_IN, S_IN, fill=False, ec=BLUE, lw=1.2, zorder=4))
    ax.text(xr + S_IN / 2, yr - 0.6, "rain at time $t$", ha="center", va="top", fontsize=6.5)
    # both enter the U-Net at one point
    xj = 15.6
    st_right = 0.6 + 0.5 * (len(state) - 1) + S_IN
    ax.plot([st_right, xj], [YC, YC], color=INK, lw=0.9, zorder=6)
    ax.plot([xr + S_IN, xj, xj], [yr + S_IN / 2, yr + S_IN / 2, YC], color=INK, lw=0.9, zorder=6)

    # ---------------- the U-Net ----------------------------------------------
    frame(18.5, 11.5, 35.0, 40.0, BLUE, BLUE_BG)
    panel(18.5, "B.", r"U-Net: fields at $t$ $\rightarrow$ rain at $t + \Delta t$")
    panel(0.4, "A.", "Inputs")
    W, DX, DY = 2.0, 2.75, 3.7
    xe = [21.0 + DX * i for i in range(5)]
    xd = [49.0 - DX * i for i in range(5)]
    yc = [YC - DY * i for i in range(5)]
    hh = [12.0, 9.2, 6.9, 5.0, 3.5]
    res = ["$128^2$", "$64^2$", "$32^2$", "$16^2$", "$8^2$"]
    xbm, ybm, wb, hb = 36.0, YC - DY * 5, 2.8, 2.2      # bottleneck, on the same staircase
    for i in range(5):
        for x in (xe[i], xd[i]):
            ax.add_patch(Rectangle((x, yc[i] - hh[i] / 2), W, hh[i], fc="white", ec=BLUE,
                                   lw=0.8, zorder=3))
        top = yc[i] + hh[i] / 2
        ax.plot([xe[i] + W + 0.3, xd[i] - 0.3], [top - 0.8] * 2, color=BLUE, lw=0.5,
                ls=(0, (4, 3)), zorder=4)
        ax.text(xe[i] + W / 2, yc[i] - hh[i] / 2 - 0.5, res[i], ha="center", va="top",
                fontsize=5, color=GRAY)
    ax.add_patch(Rectangle((xbm - wb / 2, ybm - hb / 2), wb, hb, fc=BLUE, ec=BLUE,
                           lw=0.8, alpha=0.55, zorder=3))
    ax.text(xbm, ybm - hb / 2 - 0.5, "bottleneck $4^2$", ha="center", va="top", fontsize=5,
            color=GRAY)
    ax.text(36.0, yc[0] + hh[0] / 2 + 0.2, "skip connections", ha="center", va="bottom",
            fontsize=5.5, color=BLUE)
    ax.text(xe[1], 19.0, "encoder", ha="center", fontsize=6, color=BLUE)
    ax.text(xd[1] + W, 19.0, "decoder", ha="center", fontsize=6, color=BLUE)
    arrow((xj, YC), (xe[0], YC))

    # ---------------- U-Net forecast, then the flow ---------------------------
    S = 7.6
    y0 = YC - S / 2
    frame(67.2, 11.5, 32.4, 40.0, ORANGE, "none")
    panel(67.2, "C.", "Flow matching: forecast + noise $\\rightarrow$ draws")
    xf, x0s, xdr = 58.2, 68.6, 91.2
    rain(d["forecast"], xf, y0, S, "U-Net forecast", ec=BLUE, lw=1.1)
    arrow((xd[0] + W, YC), (xf, YC))
    rain(d["path"][0], x0s, y0, S, "$x_0$")
    ax.text(x0s + S / 2, y0 - 0.6, r"forecast $+\,\sigma\varepsilon$", ha="center",
            va="top", fontsize=5.5, color=ORANGE)
    arrow((xf + S, YC), (x0s, YC), color=ORANGE)
    # the velocity network: a second U-Net, drawn like the one in b, applied 32 times
    bx0, bx1, bh = 78.3, 88.1, 8.4
    by0, by1 = YC - bh / 2, YC + bh / 2
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, bh, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=ORANGE_BG, ec=ORANGE, lw=0.7, zorder=3))
    mw, mdx, mdy = 0.52, 0.80, 0.85
    mh = [4.2, 3.2, 2.4, 1.7, 1.2]
    xm0 = bx0 + 0.75
    xme = [xm0 + mdx * i for i in range(5)]
    xmd = [bx1 - 0.75 - mw - mdx * i for i in range(5)]
    ymc = [YC + 1.5 - mdy * i for i in range(5)]
    for i in range(5):
        for x in (xme[i], xmd[i]):
            ax.add_patch(Rectangle((x, ymc[i] - mh[i] / 2), mw, mh[i], fc="white", ec=ORANGE,
                                   lw=0.6, zorder=4))
        ax.plot([xme[i] + mw + 0.12, xmd[i] - 0.12], [ymc[i] + mh[i] / 2 - 0.3] * 2, color=ORANGE,
                lw=0.35, ls=(0, (3, 2)), zorder=4)
    xmb = (xme[4] + mw + xmd[4]) / 2
    ax.add_patch(Rectangle((xmb - 0.5, ymc[4] - mdy - 0.45), 1.0, 0.9, fc=ORANGE, ec=ORANGE,
                           lw=0.6, alpha=0.55, zorder=4))
    ax.text((bx0 + bx1) / 2, by1 + 0.5, "velocity network", ha="center", va="bottom",
            fontsize=6.5, color=ORANGE)
    # in: the state x_t; out: the velocity
    arrow((x0s + S, YC), (bx0, YC), color=ORANGE)
    ax.text((x0s + S + bx0) / 2, YC + 0.6, "$x_t$", ha="center", va="bottom", fontsize=6.5)
    # the loop: x_t <- x_t + v delta, 32 times
    arrow((bx1 - 1.6, by1 + 3.3), (bx0 + 1.6, by1 + 3.3), color=ORANGE, lw=0.8, ms=6, rad=0.5)
    ax.text((bx0 + bx1) / 2, by1 + 6.0, r"$x_t \leftarrow x_t + \hat{v}\,\delta$,  32 times",
            ha="center", va="bottom", fontsize=6.5)
    # three draws, three noise seeds
    ys = [YC + S + 0.9 - S / 2, YC - S / 2, YC - S - 0.9 - S / 2]
    # one line out of the network, split at right angles into the three draws
    xbus = (bx1 + xdr) / 2 - 0.1
    ax.plot([bx1, xbus], [YC, YC], color=ORANGE, lw=0.8, zorder=6, solid_capstyle="butt")
    ax.plot([xbus, xbus], [ys[2] + S / 2, ys[0] + S / 2], color=ORANGE, lw=0.8, zorder=6,
            solid_capstyle="projecting")
    for y, seed, tag in zip(ys, SEEDS, "ABC"):
        rain(d[f"draw{seed}"], xdr, y, S, tag=f"draw {tag}, $x_1$")
        arrow((xbus, y + S / 2), (xdr, y + S / 2), color=ORANGE, lw=0.8, ms=5)

    # what the flow does to the distribution: three broad modes -> three sharp ones
    from matplotlib.colors import LinearSegmentedColormap
    DENS = LinearSegmentedColormap.from_list("dens", ["#fbf6f3", "#e3b9a6", ORANGE, "#6e2f1c"])
    g = np.linspace(-2.2, 2.2, 160)
    X, Y = np.meshgrid(g, g)
    mus = [(-0.95, -0.55), (0.95, -0.55), (0.0, 1.05)]

    def mix(sig):
        return sum(np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / (2 * sig ** 2)) for mx, my in mus)

    SD, yd = 7.6, 14.4
    xa, xb = 68.6, 91.2
    def to_ax(xq, u, v):   # sketch coordinates (-2.2..2.2) -> axis coordinates
        return xq + (u + 2.2) / 4.4 * SD, yd + (v + 2.2) / 4.4 * SD

    for xq, sig, lab in ((xa, 0.58, "broad"), (xb, 0.2, "sharp")):
        p = mix(sig)
        ax.imshow(p / p.max(), origin="lower", cmap=DENS, vmin=0, vmax=1.15, aspect="auto",
                  extent=(xq, xq + SD, yd, yd + SD), interpolation="bilinear", zorder=3)
        for q in np.linspace(0, SD, 7)[1:-1]:   # grid
            ax.plot([xq + q, xq + q], [yd, yd + SD], color="0.35", lw=0.25, alpha=0.5, zorder=3.5)
            ax.plot([xq, xq + SD], [yd + q, yd + q], color="0.35", lw=0.25, alpha=0.5, zorder=3.5)
        ax.add_patch(Rectangle((xq, yd), SD, SD, fill=False, ec="0.25", lw=0.5, zorder=5))
        ax.text(xq + SD / 2, yd + SD + 0.6, lab, ha="center", va="bottom", fontsize=6.5)
    src = to_ax(xa, 0.45, 0.25)                                 # one point of the broad cloud
    # three draws: one falls near one peak, the two others near another
    ends = [to_ax(xb, mus[2][0] + 0.02, mus[2][1] - 0.14),
            to_ax(xb, mus[1][0] - 0.13, mus[1][1] + 0.12),
            to_ax(xb, mus[1][0] + 0.10, mus[1][1] - 0.12)]
    for e, rad in zip(ends, (-0.30, -0.16, 0.10)):
        arrow(src, e, color=INK, lw=0.6, ms=5, rad=rad)
        ax.plot(*e, "o", ms=2.4, mfc=INK, mec="white", mew=0.4, zorder=7)
    ax.plot(*src, "o", ms=3.0, mfc=INK, mec="white", mew=0.4, zorder=7)
    ax.text((xa + SD + xb) / 2, yd - 0.5, "one forecast, three draws", ha="center", va="top",
            fontsize=5.5, color=GRAY)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "fig2_pipeline.pdf")
    print("saved fig2_pipeline.pdf")


if __name__ == "__main__":
    if "--compute" in sys.argv or not CACHE.exists():
        compute()
    draw()
