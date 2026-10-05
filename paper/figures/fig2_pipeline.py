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
    ax.set_xlim(-1.6, 100)
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

    YM = 31.5   # the level of the main row: inputs join -> U-Net -> forecast -> flow

    def unet(x0, yc, col, bg, title, n_in, n_out):
        """One U-Net glyph; the two networks are drawn at the same size (comparable parameters)."""
        gw, gh = 18.7, 18.4
        ax.add_patch(FancyBboxPatch((x0, yc - gh / 2), gw, gh, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=bg, ec=col, lw=0.8, zorder=3))
        bw, dx, dy = 1.0, 1.45, 1.75
        hs = [8.5, 6.6, 5.0, 3.6, 2.5]
        xe_ = [x0 + 1.2 + dx * i for i in range(5)]
        xd_ = [x0 + gw - 1.2 - bw - dx * i for i in range(5)]
        ycs = [yc + 2.6 - dy * i for i in range(5)]
        for i in range(5):
            for x in (xe_[i], xd_[i]):
                ax.add_patch(Rectangle((x, ycs[i] - hs[i] / 2), bw, hs[i], fc="white", ec=col,
                                       lw=0.7, zorder=4))
            ax.plot([xe_[i] + bw + 0.2, xd_[i] - 0.2], [ycs[i] + hs[i] / 2 - 0.45] * 2, color=col,
                    lw=0.4, ls=(0, (3, 2)), zorder=4)
        xb_ = (xe_[4] + bw + xd_[4]) / 2
        ax.add_patch(Rectangle((xb_ - 0.65, ycs[4] - dy - 0.65), 1.3, 1.3, fc=col, ec=col, lw=0.6,
                               alpha=0.55, zorder=4))
        # sizes as resolution x channels: what goes in, what comes out, and the latent
        for xq, nch in ((xe_[0] + bw / 2 + 1.5, n_in), (xd_[0] + bw / 2 - 1.5, n_out)):
            ax.text(xq, ycs[0] + hs[0] / 2 + 0.45, rf"$128^2\!\times\!{nch}$", ha="center",
                    va="bottom", fontsize=5, color=col)
        ax.text(xb_, ycs[4] - dy - 1.0, r"latent $4^2\!\times\!2048$", ha="center", va="top",
                fontsize=5, color=col)
        ax.text(x0 + gw / 2, yc + gh / 2 + 0.5, title, ha="center", va="bottom", fontsize=6.5,
                color=col)
        return x0 + gw, yc + gh / 2

    # ---------------- inputs: nine state fields stacked, the rain at t below ---
    S_IN = 9.0
    names = [str(v) for v in d["input_vars"]]
    state = [k for k, v in enumerate(names) if v != "rain_log1p"]
    for j, k in enumerate(state):
        f = d["inputs"][k]
        lo, hi = np.percentile(f, [2, 98])
        x0, y0 = 0.6 + 0.5 * j, 49.6 - 0.8 * j - S_IN
        ax.imshow(f, origin="lower", cmap=CMAPS[names[k]], vmin=lo, vmax=hi, aspect="auto",
                  extent=(x0, x0 + S_IN, y0, y0 + S_IN), interpolation="nearest",
                  zorder=3 + 0.01 * j)
        ax.add_patch(Rectangle((x0, y0), S_IN, S_IN, fill=False, ec="0.35", lw=0.5,
                               zorder=3.005 + 0.01 * j))
    ax.text(7.2, 38.0 - 0.8 * 8 - 0.2, "state at time $t$", ha="center", va="top", fontsize=6.5)
    xr, yr = 2.6, 16.0
    f = d["inputs"][names.index("rain_log1p")]
    ax.imshow(f, origin="lower", cmap="Blues", vmin=0, vmax=np.percentile(f, 99.5), aspect="auto",
              extent=(xr, xr + S_IN, yr, yr + S_IN), interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((xr, yr), S_IN, S_IN, fill=False, ec=BLUE, lw=1.2, zorder=4))
    ax.text(xr + S_IN / 2, yr - 0.6, "rain at time $t$", ha="center", va="top", fontsize=6.5)
    panel(-1.2, "A.", "Inputs")
    # the two inputs join and enter the U-Net
    xj, y_st = 15.6, 37.0
    st_right = 0.6 + 0.5 * (len(state) - 1) + S_IN
    ax.plot([st_right, xj, xj, xr + S_IN], [y_st, y_st, yr + S_IN / 2, yr + S_IN / 2], color=INK,
            lw=0.9, zorder=6)

    # ---------------- the U-Net and its forecast ------------------------------
    S = 7.6
    y0 = YM - S / 2
    xu = 18.6
    xu1, _ = unet(xu, YM, BLUE, BLUE_BG, "U-Net", 10, 2)
    panel(xu, "B.", r"U-Net: rain at $t + \Delta t$")
    arrow((xj, YM), (xu, YM))
    xf = xu1 + 3.0
    rain(d["forecast"], xf, y0, S, "U-Net forecast", ec=BLUE, lw=1.1)
    arrow((xu1, YM), (xf, YM))
    frame(-1.2, 11.5, xf + S + 2.2, 40.0, "0.55", "none")   # A and B together, as tall as C

    # ---------------- flow matching -------------------------------------------
    xc0 = xf + S + 2.5
    frame(xc0, 11.5, 99.6 - xc0, 40.0, ORANGE, "none")
    panel(xc0, "C.", "Flow matching: forecast + noise $\\rightarrow$ draws")
    x0s = xc0 + 1.6
    rain(d["path"][0], x0s, y0, S, "$x_0$")
    ax.text(x0s + S / 2, y0 - 0.6, r"forecast $+\,\sigma\varepsilon$", ha="center",
            va="top", fontsize=5.5, color=ORANGE)
    arrow((xf + S, YM), (x0s, YM), color=ORANGE)
    # the velocity network: a second U-Net of the same size, applied 32 times
    xv = x0s + S + 3.0
    xv1, yv1 = unet(xv, YM, ORANGE, ORANGE_BG, "velocity network", 11, 1)
    arrow((x0s + S, YM), (xv, YM), color=ORANGE)
    ax.text((x0s + S + xv) / 2, YM + 0.6, "$x_t$", ha="center", va="bottom", fontsize=6.5)
    arrow((xv1 - 3.5, yv1 + 3.3), (xv + 3.5, yv1 + 3.3), color=ORANGE, lw=0.8, ms=6, rad=0.4)
    ax.text((xv + xv1) / 2, yv1 + 6.6, r"$x_t \leftarrow x_t + \hat{v}\,\delta$,  32 times",
            ha="center", va="bottom", fontsize=6.5)
    # three draws, three noise seeds: one line out of the network, split at right angles
    xdr = 91.2
    ys = [YM + S + 0.9 - S / 2, YM - S / 2, YM - S - 0.9 - S / 2]
    xbus = (xv1 + xdr) / 2
    ax.plot([xv1, xbus], [YM, YM], color=ORANGE, lw=0.8, zorder=6, solid_capstyle="butt")
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

    SD, yd = 6.4, 12.9
    mid = (x0s + xv1) / 2 - 0.6      # the pair centred under x_0 and the velocity network
    xa, xb = mid - 6.5 - SD, mid + 6.5

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
        ax.text(xq + SD / 2, yd + SD + 0.5, lab, ha="center", va="bottom", fontsize=6.5)
    src = to_ax(xa, 0.45, 0.25)                                 # one point of the broad cloud
    # three draws: one falls near one peak, the two others near another
    ends = [to_ax(xb, mus[2][0] + 0.02, mus[2][1] - 0.14),
            to_ax(xb, mus[1][0] - 0.13, mus[1][1] + 0.12),
            to_ax(xb, mus[1][0] + 0.10, mus[1][1] - 0.12)]
    for e, rad in zip(ends, (-0.30, -0.16, 0.10)):
        arrow(src, e, color=INK, lw=0.6, ms=5, rad=rad)
        ax.plot(*e, "o", ms=2.4, mfc=INK, mec="white", mew=0.4, zorder=7)
    ax.plot(*src, "o", ms=3.0, mfc=INK, mec="white", mew=0.4, zorder=7)
    ax.text(xb + SD + 1.0, yd + SD / 2, "one forecast,\nthree draws", ha="left", va="center",
            fontsize=5.5, color=GRAY, linespacing=1.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "fig2_pipeline.pdf")
    print("saved fig2_pipeline.pdf")


if __name__ == "__main__":
    if "--compute" in sys.argv or not CACHE.exists():
        compute()
    draw()
