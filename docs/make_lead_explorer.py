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
    inputs = {v: np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                    mmap_mode="r")[FRAME], dtype=np.float32)[sl]
              for v in INPUT_VARS}

    vmax = max(float(now.max()), max(float(t.max()) for t in targets.values()))
    rnorm = LogNorm(vmin=0.1, vmax=vmax)
    rcmap = plt.get_cmap("Blues").copy(); rcmap.set_under("#ffffff")
    dlim = max(float(np.abs(targets[L] - now).max()) for L in LEADS)
    snorm = SymLogNorm(linthresh=0.1, vmin=-dlim, vmax=dlim)

    for L in LEADS:
        fig = plt.figure(figsize=(13.2, 5.6))
        # background zones: inputs (cool) vs output (warm)
        fig.patches.append(plt.Rectangle((0.005, 0.015), 0.755, 0.955,
                                         transform=fig.transFigure,
                                         color="#edf2f7", zorder=-2))
        fig.patches.append(plt.Rectangle((0.775, 0.015), 0.22, 0.955,
                                         transform=fig.transFigure,
                                         color="#fbeedd", zorder=-2))
        gs_in = fig.add_gridspec(2, 5, left=0.018, right=0.75,
                                 top=0.86, bottom=0.06, hspace=0.28, wspace=0.12)
        gs_out = fig.add_gridspec(2, 1, left=0.80, right=0.972,
                                  top=0.86, bottom=0.06, hspace=0.28)
        for k, v in enumerate(INPUT_VARS):
            ax = fig.add_subplot(gs_in[k // 5, k % 5])
            f = inputs[v]
            lim = float(np.abs(f).max())
            kw = (dict(vmin=-lim, vmax=lim) if CMAPS[v] in ("RdBu_r", "BrBG")
                  else {})
            im = ax.imshow(f, origin="lower", cmap=CMAPS[v],
                           interpolation="nearest", **kw)
            ax.set_title(v, fontsize=7.5, family="monospace")
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=5)
            if v == "rain_log1p":
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color("#2166ac")
                    sp.set_linewidth(2)
        ax_t = fig.add_subplot(gs_out[0])
        im_r = ax_t.imshow(targets[L], origin="lower", norm=rnorm, cmap=rcmap,
                           interpolation="nearest")
        wet = float((targets[L] > 0.1).mean() * 100)
        ax_t.set_title(f"true rain +{L:g} h · wet {wet:.0f} %", fontsize=8.5)
        ax_t.set_xticks([]); ax_t.set_yticks([])
        cbt = fig.colorbar(im_r, ax=ax_t, fraction=0.046, pad=0.03)
        cbt.set_label("mm", fontsize=6); cbt.ax.tick_params(labelsize=5)
        ax_d = fig.add_subplot(gs_out[1])
        im_d = ax_d.imshow(targets[L] - now, origin="lower", norm=snorm,
                           cmap="RdBu_r", interpolation="nearest")
        ax_d.set_title(f"Δ: (+{L:g} h) − now", fontsize=8.5)
        ax_d.set_xticks([]); ax_d.set_yticks([])
        cbd = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.03)
        cbd.set_label("Δ mm", fontsize=6); cbd.ax.tick_params(labelsize=5)

        fig.text(0.385, 0.935, "INPUT — the ten channels at t = 0",
                 ha="center", fontsize=10.5)
        fig.text(0.885, 0.935, f"OUTPUT — target at +{L:g} h",
                 ha="center", fontsize=10.5, color="#b05e0a")
        fig.savefig(OUT / f"lead_frame_{L:g}h.png", dpi=125)
        plt.close(fig)
        print(f"saved lead_frame_{L:g}h.png")


if __name__ == "__main__":
    main()
