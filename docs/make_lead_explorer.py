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

    def panel(f, title, fn, pnorm, pcmap, cblabel):
        """Fixed canvas (no tight bbox): every frame has identical pixel
        dimensions, so the page's click-through never shifts layout."""
        fig = plt.figure(figsize=(5.4, 5.0))
        ax = fig.add_axes([0.02, 0.02, 0.80, 0.86])
        im = ax.imshow(f, origin="lower", norm=pnorm, cmap=pcmap,
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        cax = fig.add_axes([0.85, 0.08, 0.035, 0.74])
        fig.colorbar(im, cax=cax).set_label(cblabel, fontsize=8)
        fig.savefig(OUT / fn, dpi=130)
        plt.close(fig)

    def rain_panel(f, title, fn):
        panel(f, title, fn, norm, cmap, "rain (mm)")

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
        frac = float(np.abs(d).sum() / max(now.sum(), 1e-6))
        panel(d, f"Δ rain: (+{L:g} h) − now   (red = appears/intensifies)",
              f"leaddiff_{L:g}h.png", snorm, "RdBu_r", "Δ rain (mm)")
        print(f"+{L:g}h: |diff|/now volume ratio {frac:.2f}")
    print("saved lead explorer frames")


if __name__ == "__main__":
    main()
