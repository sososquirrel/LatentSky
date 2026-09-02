"""Figure: predicted precip at every available lead (0h..6h), same input
snapshot, West Africa crop. Same vmin/vmax (one shared colorbar) across
ALL leads, so the pattern fading over time is directly comparable.

Note: inference was run on the FULL 320x1440 domain (correct convolutional
context, no patch-boundary seams) then cropped to West Africa for display
-- not a literal 64x64 training patch.

Reads the cache written by run_inference_cache.py.
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
CACHE = OUT_DIR / "_inference_cache.npz"

LAT_EXTENT = (-40.0, 40.0)
LON_EXTENT = (0.0, 360.0)


def grid_latlon(ny, nx):
    lat = LAT_EXTENT[0] + (np.arange(ny) + 0.5) * (LAT_EXTENT[1] - LAT_EXTENT[0]) / ny
    lon = LON_EXTENT[0] + (np.arange(nx) + 0.5) * (LON_EXTENT[1] - LON_EXTENT[0]) / nx
    return lat, lon


def main():
    d = np.load(CACHE, allow_pickle=True)
    leads = d["leads"]
    ny, nx = d["X_full"].shape[1], d["X_full"].shape[2]
    lat, lon = grid_latlon(ny, nx)

    r0, r1 = int(d["wa_lat_idx"][0]), int(d["wa_lat_idx"][1])
    c0, c1 = int(d["wa_lon_idx"][0]), int(d["wa_lon_idx"][1])
    lat_c, lon_c = lat[r0:r1], lon[c0:c1]

    preds = [d[f"pred_{lead:g}h"][r0:r1, c0:c1] for lead in leads]

    # Shared color scale across every lead, so intensity fading over time
    # is directly comparable (not each panel auto-scaled to its own max).
    vmin_log = 0.1
    vmax = float(max(max(p.max() for p in preds), vmin_log * 10))
    norm = LogNorm(vmin=vmin_log, vmax=vmax)
    levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 20)

    n = len(leads)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.2 * nrows),
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
        f"Predicted precip vs lead — piece {int(d['piece_id'])}, frame {int(d['frame'])}, "
        f"West Africa crop (lat {lat_c[0]:.1f}°-{lat_c[-1]:.1f}°N, "
        f"lon {lon_c[0]:.1f}°-{lon_c[-1]:.1f}°E)",
        fontsize=14,
    )

    out = OUT_DIR / "fig03_outputs.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
