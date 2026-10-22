"""Paper Fig 2 companions — the U-Net's input stack and output pair.

fig2a_inputs.pdf : the ten input channels on the example block (2 x 5),
                   rain_log1p last with a blue frame (the nested persistence
                   input); colormaps by family as on the page (moisture/MSE
                   brown-teal, temperature thermal, OLR grayscale, winds
                   red-blue, rain Blues); one thin colorbar under each map.
fig2c_outputs.pdf: the two-head's two outputs for this input at +3 h — the
                   rain field (mm) and the wet/dry probability. CPU ~30 s.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

REPO = Path("/glade/u/home/sabramian/LatentSky")
plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
from latentsky.layout import EXAMPLE_BLOCK as BLOCK  # noqa: E402
from latentsky.layout import EXAMPLE_FRAME as FRAME  # noqa: E402
from latentsky.layout import EXAMPLE_PIECE as PIECE  # noqa: E402
from latentsky.layout import INPUT_VARS, P, PIECES, RUNS  # noqa: E402

from latentsky.models import UNet2DTwoHead  # noqa: E402

matplotlib.rcParams["savefig.bbox"] = "tight"
OUT = REPO / "paper_figures" / "extra"
CM = 1 / 2.54
W2 = 17.5 * CM
DET = "#2166ac"

CMAPS = {
    "precipw": "BrBG", "mse_500hPa": "BrBG", "mse_850hPa": "BrBG",
    "t2m": "plasma", "olrtoa": "gray",
    "uzonal_500hPa": "RdBu_r", "uzonal_850hPa": "RdBu_r",
    "umeridional_500hPa": "RdBu_r", "umeridional_850hPa": "RdBu_r",
    "rain_log1p": "Blues",
}
LABELS = {
    "precipw": "precipw", "t2m": "t2m", "olrtoa": "olrtoa",
    "mse_500hPa": "mse 500 hPa", "mse_850hPa": "mse 850 hPa",
    "uzonal_500hPa": "u 500 hPa", "uzonal_850hPa": "u 850 hPa",
    "umeridional_500hPa": "v 500 hPa", "umeridional_850hPa": "v 850 hPa",
    "rain_log1p": "rain (current)",
}


def load_block():
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME],
                             dtype=np.float32)[sl] for v in INPUT_VARS], 0)
    return x


def fig_inputs(x):
    fig, axs = plt.subplots(2, 5, figsize=(W2, W2 * 0.52))
    for k, (var, ax) in enumerate(zip(INPUT_VARS, axs.ravel())):
        f = x[k]
        cmap = CMAPS[var]
        if cmap == "RdBu_r":
            m = float(np.percentile(np.abs(f), 99))
            vmin, vmax = -m, m
        else:
            vmin = float(np.percentile(f, 1))
            vmax = float(np.percentile(f, 99))
        im = ax.imshow(f, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 1.04, LABELS[var], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7)
        if var == "rain_log1p":
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(DET)
                s.set_linewidth(2.0)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.055, pad=0.05, aspect=22)
        cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.5)
        cb.outline.set_linewidth(0.4)
        cb.locator = matplotlib.ticker.MaxNLocator(4)
        cb.update_ticks()
    fig.subplots_adjust(wspace=0.1, hspace=0.28)
    fig.savefig(OUT / "fig2a_inputs.pdf", dpi=300)
    print("saved fig2a_inputs.pdf")


def predict(x):
    """The +3 h U-Net's two outputs for one block: rain (mm) and wet/dry logits."""
    cond = torch.from_numpy(x[None])
    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    with torch.no_grad():
        rain_t, logit_t = twohead(cond)
    rain = np.clip(rain_t.numpy()[0, 0], 0, None)
    logit = logit_t.numpy()[0, 0]
    return rain, logit


def fig_outputs(x):
    rain, logit = predict(x)

    fig, axs = plt.subplots(1, 3, figsize=(W2 * 0.63, W2 * 0.27))
    im0 = axs[0].imshow(rain, origin="lower", cmap="Blues",
                        norm=LogNorm(0.1, max(rain.max(), 1.0)),
                        interpolation="nearest")
    axs[0].text(0.5, 1.04, "rain (mm)", transform=axs[0].transAxes,
                ha="center", va="bottom", fontsize=7)
    vlim = float(np.percentile(np.abs(logit), 99))
    im1 = axs[1].imshow(logit, origin="lower", cmap="RdBu_r", vmin=-vlim,
                        vmax=vlim, interpolation="nearest")
    axs[1].text(0.5, 1.04, "wet/dry (logits)", transform=axs[1].transAxes,
                ha="center", va="bottom", fontsize=7)
    from matplotlib.colors import ListedColormap
    mask = (rain > 0.1).astype(float)
    im2 = axs[2].imshow(mask, origin="lower",
                        cmap=ListedColormap(["#ffffff", DET]), vmin=0,
                        vmax=1, interpolation="nearest")
    axs[2].text(0.5, 1.04, "precip mask (rain > 0.1 mm)",
                transform=axs[2].transAxes, ha="center", va="bottom",
                fontsize=7)
    for ax, im in zip(axs, (im0, im1, im2)):
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.055, pad=0.05, aspect=22)
        cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.5)
        cb.outline.set_linewidth(0.4)
    cb.set_ticks([0.25, 0.75])
    cb.set_ticklabels(["dry", "wet"])
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(OUT / "fig2c_outputs.pdf", dpi=300)
    print("saved fig2c_outputs.pdf")


if __name__ == "__main__":
    xb = load_block()
    fig_inputs(xb)
    fig_outputs(xb)
