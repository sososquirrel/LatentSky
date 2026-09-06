"""Frames for the interactive lead-time explorer in model_reference.html §1:
the standard example block (piece 8, frame 64, block (1,4)).

  lead_inputs.png       — compact 2x5 grid of the ten standardized input channels
  lead_now.png          — rain at the initialization time (shared log scale)
  lead_{L}h.png         — true rain at +L for the six trained leads
  leaddiff_{L}h.png     — target − now (SymLog, positive = red), shared limits

All rain panels share one LogNorm and all diff panels one SymLogNorm so the
click-through reads as an animation. CPU, seconds."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

REPO = Path("/glade/u/home/sabramian/LatentSky")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"
INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
CMAPS = {"precipw": "BrBG", "t2m": "inferno", "olrtoa": "gray",
         "mse_500hPa": "BrBG", "mse_850hPa": "BrBG",
         "uzonal_500hPa": "RdBu_r", "uzonal_850hPa": "RdBu_r",
         "umeridional_500hPa": "RdBu_r", "umeridional_850hPa": "RdBu_r",
         "rain_log1p": "Blues"}
P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
LEADS = [0.25, 0.5, 1, 2, 3, 6]

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
matplotlib.rcParams["savefig.bbox"] = None  # fixed canvas for the explorer frames


def main():
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    ytab = np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")
    now = np.asarray(ytab[FRAME], dtype=np.float32)[sl]
    targets = {L: np.asarray(ytab[FRAME + int(round(L * 4)) + 1],
                             dtype=np.float32)[sl] for L in LEADS}

    # --- inputs grid: 3x3 met channels + rain_log1p centered beneath
    fig = plt.figure(figsize=(6.2, 8.9))
    gs = fig.add_gridspec(4, 3, hspace=0.25, wspace=0.06,
                          left=0.02, right=0.98, top=0.93, bottom=0.02)

    def draw(ax, v, title_kw=None):
        f = np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME], dtype=np.float32)[sl]
        lim = float(np.abs(f).max())
        kw = (dict(vmin=-lim, vmax=lim) if CMAPS[v] in ("RdBu_r", "BrBG")
              else {})
        ax.imshow(f, origin="lower", cmap=CMAPS[v], interpolation="nearest", **kw)
        ax.set_title(v, fontsize=8, family="monospace", **(title_kw or {}))
        ax.set_xticks([]); ax.set_yticks([])

    met = [v for v in INPUT_VARS if v != "rain_log1p"]
    for k, v in enumerate(met):
        draw(fig.add_subplot(gs[k // 3, k % 3]), v)
    ax = fig.add_subplot(gs[3, 1])
    draw(ax, "rain_log1p", title_kw=dict(color="#2166ac", fontweight="bold"))
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#2166ac"); sp.set_linewidth(2)
    fig.suptitle("input: the ten channels at t = 0\n"
                 "(bottom: the current rain — the nested persistence input)",
                 fontsize=9.5, y=0.985)
    fig.savefig(OUT / "lead_inputs.png", dpi=130)
    plt.close(fig)

    # --- rain panels, shared scale
    vmax = max(float(now.max()), max(float(t.max()) for t in targets.values()))
    norm = LogNorm(vmin=0.1, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")

    def panel(f, title, fn, pnorm, pcmap, cblabel):
        """One cell, sized to match a single input-grid box (1.907 inch map
        at the same dpi), fixed canvas: title, map, horizontal colorbar.
        Identical pixel dimensions across frames - no layout shift."""
        fig = plt.figure(figsize=(2.0, 2.95))
        ax = fig.add_axes([0.023, 0.265, 0.9535, 0.6464])
        im = ax.imshow(f, origin="lower", norm=pnorm, cmap=pcmap,
                       interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        cax = fig.add_axes([0.1, 0.12, 0.8, 0.045])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(cblabel, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.savefig(OUT / fn, dpi=130)
        plt.close(fig)

    def rain_panel(f, title, fn):
        panel(f, title, fn, norm, cmap, "rain (mm)")

    rain_panel(now, "rain now (t = 0)", "lead_now.png")
    for L in LEADS:
        wet = float((targets[L] > 0.1).mean() * 100)
        rain_panel(targets[L], f"truth +{L:g} h · wet {wet:.0f} %",
                   f"lead_{L:g}h.png")

    # --- diff panels, shared symlog scale, positive = red
    dlim = max(float(np.abs(targets[L] - now).max()) for L in LEADS)
    snorm = SymLogNorm(linthresh=0.1, vmin=-dlim, vmax=dlim)
    for L in LEADS:
        d = targets[L] - now
        frac = float(np.abs(d).sum() / max(now.sum(), 1e-6))
        panel(d, f"Δ: (+{L:g} h) − now", f"leaddiff_{L:g}h.png",
              snorm, "RdBu_r", "Δ rain (mm), red = appears")
        print(f"+{L:g}h: |diff|/now volume ratio {frac:.2f}")
    print("saved lead explorer frames")


if __name__ == "__main__":
    main()
