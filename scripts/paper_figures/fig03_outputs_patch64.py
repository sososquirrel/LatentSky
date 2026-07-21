"""Figure: predicted precip at every available lead, GENUINE 64x64 px
(16x16 deg) training-patch-sized crop -- each lead's model ran a single
forward pass on exactly this 64x64 input, nothing cropped after the fact
(contrast with fig03_outputs.py, which infers on the full domain then
crops the output for display).

Same vmin/vmax (one shared colorbar) across all leads.

Reads the cache written by run_inference_patch64.py.
"""
from __future__ import annotations
import pathlib
import numpy as np
import cmocean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
CACHE = OUT_DIR / "_inference_cache_patch64.npz"

LAT_EXTENT = (-40.0, 40.0)
LON_EXTENT = (0.0, 360.0)


def grid_latlon_full(ny, nx):
    lat = LAT_EXTENT[0] + (np.arange(ny) + 0.5) * (LAT_EXTENT[1] - LAT_EXTENT[0]) / ny
    lon = LON_EXTENT[0] + (np.arange(nx) + 0.5) * (LON_EXTENT[1] - LON_EXTENT[0]) / nx
    return lat, lon


def main():
    d = np.load(CACHE, allow_pickle=True)
    leads = d["leads"]
    leads = leads[leads != 6.0]  # drop 6h

    r0, r1 = d["patch_lat_idx_in_full_domain"]
    c0, c1 = d["patch_lon_idx_in_full_domain"]
    lat_full, lon_full = grid_latlon_full(320, 1440)
    lat_c, lon_c = lat_full[r0:r1], lon_full[c0:c1]

    preds = [d[f"pred_{lead:g}h"] for lead in leads]

    vmin_log = 0.1
    vmax = float(max(max(p.max() for p in preds), vmin_log * 10))
    norm = LogNorm(vmin=vmin_log, vmax=vmax)
    levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 20)

    n = len(leads)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows),
                            squeeze=False, constrained_layout=True)

    im = None
    for j, (lead, pred) in enumerate(zip(leads, preds)):
        r, c = divmod(j, ncols)
        ax = axs[r, c]
        im = ax.contourf(lon_c, lat_c, pred, levels=levels, cmap=cmocean.cm.rain,
                         norm=norm, extend="both")
        ax.set_title(f"lead = {lead:g} h", fontsize=13)
        ax.set_xlabel("lon (°E)", fontsize=10)
        ax.set_ylabel("lat (°N)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.4, lw=0.5, color="0.3")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r, c].set_visible(False)

    cbar = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.5,
                        pad=0.02, aspect=40)
    cbar.set_label("predicted precip (mm, log scale) — SAME scale every panel", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Predicted precip vs lead — GENUINE 64×64 px (16°×16°) training-patch-sized crop, "
        f"piece {int(d['piece_id'])}, frame {int(d['frame'])}\n"
        f"lat {lat_c[0]:.1f}°-{lat_c[-1]:.1f}°N, lon {lon_c[0]:.1f}°-{lon_c[-1]:.1f}°E",
        fontsize=14,
    )

    out = OUT_DIR / "fig03_outputs_patch64.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
