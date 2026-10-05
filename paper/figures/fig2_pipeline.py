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

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.555))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55.5)
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
        ax.text(x, 52.4, letter, ha="left", va="bottom", fontsize=8, fontweight="bold")
        ax.text(x + 2.0, 52.4, title, ha="left", va="bottom", fontsize=7)

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
    frame(17.0, 4.5, 43.5, 47.0, "0.55", "none")
    panel(17.0, "b", r"U-Net: fields at $t$ $\rightarrow$ rain at $t + \Delta t$")
    panel(0.4, "a", "Inputs")
    W, DX, DY = 2.3, 3.5, 5.0
    xe = [19.5 + DX * i for i in range(5)]
    xd = [55.7 - DX * i for i in range(5)]
    yc = [YC - DY * i for i in range(5)]
    hh = [15.0, 11.5, 8.6, 6.2, 4.2]
    res = ["128²", "64²", "32²", "16²", "8²"]
    xbm, ybm, wb, hb = 38.75, YC - DY * 5, 3.6, 2.8      # bottleneck, on the same staircase
    for i in range(5):
        for x in (xe[i], xd[i]):
            ax.add_patch(Rectangle((x, yc[i] - hh[i] / 2), W, hh[i], fc=BLUE_BG, ec=BLUE,
                                   lw=0.8, zorder=3))
        top = yc[i] + hh[i] / 2
        arrow((xe[i] + W + 0.3, top - 0.8), (xd[i] - 0.3, top - 0.8), color="0.5",
              lw=0.7, ls=(0, (4, 3)), ms=6)
        ax.text(xe[i] + W / 2, yc[i] - hh[i] / 2 - 0.5, res[i], ha="center", va="top",
                fontsize=5, color=GRAY)
        if i < 4:   # centre to centre, one slope everywhere
            arrow((xe[i] + W, yc[i]), (xe[i + 1], yc[i + 1]), ms=5, lw=0.7)
            arrow((xd[i + 1] + W, yc[i + 1]), (xd[i], yc[i]), ms=5, lw=0.7)
    ax.add_patch(Rectangle((xbm - wb / 2, ybm - hb / 2), wb, hb, fc="#b9cfe6", ec=BLUE,
                           lw=0.8, zorder=3))
    ax.text(xbm, ybm - hb / 2 - 0.5, "bottleneck 4²", ha="center", va="top", fontsize=5,
            color=GRAY)
    arrow((xe[4] + W, yc[4]), (xbm - wb / 2, ybm), ms=5, lw=0.7)
    arrow((xbm + wb / 2, ybm), (xd[4], yc[4]), ms=5, lw=0.7)
    ax.text(38.75, yc[0] + hh[0] / 2 + 0.2, "skip connections", ha="center", va="bottom",
            fontsize=5.5, color="0.5")
    ax.text(xe[1], 17.5, "encoder", ha="center", fontsize=6, color=BLUE)
    ax.text(xd[1] + W, 17.5, "decoder", ha="center", fontsize=6, color=BLUE)
    arrow((xj, YC), (xe[0], YC))

    # ---------------- U-Net forecast, then the flow ---------------------------
    S = 7.6
    y0 = YC - S / 2
    frame(71.2, 4.5, 28.4, 47.0, ORANGE, "none")
    panel(71.2, "c", "Flow matching: forecast + noise $\\rightarrow$ draws")
    xf, x0s, xdr = 62.2, 73.0, 91.2
    rain(d["forecast"], xf, y0, S, "U-Net forecast", ec=BLUE, lw=1.1)
    arrow((xd[0] + W, YC), (xf, YC))
    rain(d["path"][0], x0s, y0, S, "$x_0$")
    ax.text(x0s + S / 2, y0 - 0.6, r"forecast $+\,\sigma\varepsilon$", ha="center",
            va="top", fontsize=5.5, color=ORANGE)
    arrow((xf + S, YC), (x0s, YC), color=ORANGE)
    # three draws, three noise seeds
    ys = [YC + S + 0.9 - S / 2, YC - S / 2, YC - S - 0.9 - S / 2]
    for y, seed, tag in zip(ys, SEEDS, "ABC"):
        rain(d[f"draw{seed}"], xdr, y, S, tag=f"draw {tag}, $x_1$")
        arrow((x0s + S, YC), (xdr, y + S / 2), color=ORANGE, lw=0.8, ms=6)
    ax.text((x0s + S + xdr) / 2 - 0.3, YC + 4.6, r"$\hat{v}(x_t, t, c)$" + "\n32 steps",
            ha="center", va="bottom", fontsize=5.5, color=ORANGE, linespacing=1.2)

    # what the flow does to the distribution: three broad modes -> three sharp ones
    g = np.linspace(-2.2, 2.2, 160)
    X, Y = np.meshgrid(g, g)
    mus = [(-0.95, -0.55), (0.95, -0.55), (0.0, 1.05)]

    def mix(sig):
        return sum(np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / (2 * sig ** 2)) for mx, my in mus)

    SD, yd = 8.6, 10.6
    xa, xb = 73.0, 90.2
    def to_ax(xq, u, v):   # sketch coordinates (-2.2..2.2) -> axis coordinates
        return xq + (u + 2.2) / 4.4 * SD, yd + (v + 2.2) / 4.4 * SD

    for xq, sig, lab in ((xa, 0.58, "broad"), (xb, 0.2, "sharp")):
        p = mix(sig)
        ax.imshow(p / p.max(), origin="lower", cmap="Oranges", vmin=0, vmax=1.15, aspect="auto",
                  extent=(xq, xq + SD, yd, yd + SD), interpolation="bilinear", zorder=3)
        for q in np.linspace(0, SD, 7)[1:-1]:   # grid
            ax.plot([xq + q, xq + q], [yd, yd + SD], color="0.35", lw=0.25, alpha=0.5, zorder=3.5)
            ax.plot([xq, xq + SD], [yd + q, yd + q], color="0.35", lw=0.25, alpha=0.5, zorder=3.5)
        ax.add_patch(Rectangle((xq, yd), SD, SD, fill=False, ec="0.25", lw=0.5, zorder=5))
        ax.text(xq + SD / 2, yd + SD + 0.6, lab, ha="center", va="bottom", fontsize=6.5)
    src = to_ax(xa, 0.15, -0.1)                                  # one point of the broad cloud
    ends = [to_ax(xb, mx + ox, my + oy) for (mx, my), (ox, oy) in
            zip(mus, [(0.22, 0.2), (-0.24, 0.16), (0.2, -0.24)])]   # one draw near each peak
    for e, rad in zip(ends, (0.10, 0.34, -0.25)):
        arrow(src, e, color=INK, lw=0.6, ms=5, rad=rad)
        ax.plot(*e, "o", ms=2.6, mfc=INK, mec="white", mew=0.4, zorder=7)
    ax.plot(*src, "o", ms=3.0, mfc=INK, mec="white", mew=0.4, zorder=7)
    ax.text(85.4, yd + SD + 3.4, "distribution of the possible outcomes", ha="center",
            va="bottom", fontsize=6, color=GRAY)
    ax.text((xa + SD + xb) / 2, yd - 0.5, "one forecast, three draws", ha="center", va="top",
            fontsize=5.5, color=GRAY)
    # conditioning: the same ten fields
    yl = 2.2
    xc_ = (xa + SD + xb) / 2
    ax.plot([xj, xj, xc_], [yr + S_IN / 2, yl, yl], color="0.45", lw=0.6,
            ls=(0, (4, 3)), zorder=6)
    arrow((xc_, yl), (xc_, yd - 2.0), color="0.45", lw=0.6, ls=(0, (4, 3)), ms=5)
    ax.text(42, yl + 0.5, "conditioning $c$: the same ten fields", ha="center", va="bottom",
            fontsize=6, color=GRAY)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "fig2_pipeline.pdf")
    print("saved fig2_pipeline.pdf")


if __name__ == "__main__":
    if "--compute" in sys.argv or not CACHE.exists():
        compute()
    draw()
