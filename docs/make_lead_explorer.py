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


def main():
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    ytab = np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")
    now = np.asarray(ytab[FRAME], dtype=np.float32)[sl]
    targets = {L: np.asarray(ytab[FRAME + int(round(L * 4)) + 1],
                             dtype=np.float32)[sl] for L in LEADS}

    # --- inputs grid (portrait, sits left of the lead panel)
    fig, axs = plt.subplots(4, 3, figsize=(6.2, 8.6))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, v in zip(axs.ravel(), INPUT_VARS):
        f = np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME], dtype=np.float32)[sl]
        lim = float(np.abs(f).max())
        kw = (dict(vmin=-lim, vmax=lim) if CMAPS[v] in ("RdBu_r", "BrBG")
              else {})
        ax.axis("on")
        ax.imshow(f, origin="lower", cmap=CMAPS[v], interpolation="nearest", **kw)
        ax.set_title(v, fontsize=8, family="monospace")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("input: the ten channels at t = 0", fontsize=10, y=0.995)
    fig.savefig(OUT / "lead_inputs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- rain panels, shared scale
    vmax = max(float(now.max()), max(float(t.max()) for t in targets.values()))
    norm = LogNorm(vmin=0.1, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")

    def rain_panel(f, title, fn):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label("rain (mm)",
                                                                 fontsize=8)
        fig.savefig(OUT / fn, dpi=130, bbox_inches="tight")
        plt.close(fig)

    rain_panel(now, "rain now (t = 0) — the persistence forecast", "lead_now.png")
    for L in LEADS:
        wet = float((targets[L] > 0.1).mean() * 100)
        rain_panel(targets[L], f"true rain at +{L:g} h   (wet {wet:.0f} %)",
                   f"lead_{L:g}h.png")

    # --- diff panels, shared symlog scale, positive = red
    dlim = max(float(np.abs(targets[L] - now).max()) for L in LEADS)
    snorm = SymLogNorm(linthresh=0.1, vmin=-dlim, vmax=dlim)
    for L in LEADS:
        d = targets[L] - now
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = ax.imshow(d, origin="lower", norm=snorm, cmap="RdBu_r",
                       interpolation="nearest")
        frac = float(np.abs(d).sum() / max(now.sum(), 1e-6))
        ax.set_title(f"Δ rain: (+{L:g} h) − now   (red = appears/intensifies)",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label("Δ rain (mm)",
                                                                 fontsize=8)
        fig.savefig(OUT / f"leaddiff_{L:g}h.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"+{L:g}h: |diff|/now volume ratio {frac:.2f}")
    print("saved lead explorer frames")


if __name__ == "__main__":
    main()
