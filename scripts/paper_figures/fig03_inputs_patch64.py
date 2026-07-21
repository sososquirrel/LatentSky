"""Figure: model inputs, all 9 channels, a GENUINE 64x64 px (16x16 deg)
training-patch-sized crop -- sliced BEFORE inference, not cropped from a
full-domain forward pass (contrast with fig03_inputs.py).

Reads the cache written by run_inference_patch64.py.
"""
from __future__ import annotations
import pathlib
import numpy as np
import cmocean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
CACHE = OUT_DIR / "_inference_cache_patch64.npz"

LAT_EXTENT = (-40.0, 40.0)
LON_EXTENT = (0.0, 360.0)

DENORM = {
    "precipw": (31.78785514831543, 15.428478240966797),
    "olrtoa": (263.7668762207031, 40.1941032409668),
}


def grid_latlon_full(ny, nx):
    lat = LAT_EXTENT[0] + (np.arange(ny) + 0.5) * (LAT_EXTENT[1] - LAT_EXTENT[0]) / ny
    lon = LON_EXTENT[0] + (np.arange(nx) + 0.5) * (LON_EXTENT[1] - LON_EXTENT[0]) / nx
    return lat, lon


def panel(ax, cax, lon, lat, data, cmap, title, symmetric=False):
    if symmetric:
        m = float(max(abs(data.min()), abs(data.max()), 1e-6))
        levels = np.linspace(-m, m, 21)
        im = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend="both")
        lo, hi = -m, m
    else:
        im = ax.contourf(lon, lat, data, levels=20, cmap=cmap)
        lo, hi = im.get_clim()

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("lon (°E)", fontsize=10)
    ax.set_ylabel("lat (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.4, lw=0.5, color="0.3")

    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    ticks = np.linspace(lo, hi, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:.0f}" if abs(v) >= 10 else f"{v:.2g}" for v in ticks])
    cbar.ax.tick_params(labelsize=9)
    return im


def main():
    d = np.load(CACHE, allow_pickle=True)
    X = d["X_patch"]  # (C, 64, 64) -- genuine patch, nothing cropped after inference
    input_vars = [str(v) for v in d["input_vars"]]

    r0, r1 = d["patch_lat_idx_in_full_domain"]
    c0, c1 = d["patch_lon_idx_in_full_domain"]
    lat_full, lon_full = grid_latlon_full(320, 1440)
    lat_c, lon_c = lat_full[r0:r1], lon_full[c0:c1]
    assert X.shape[1] == len(lat_c) and X.shape[2] == len(lon_c), \
        "patch shape doesn't match recorded lat/lon indices"

    n = len(input_vars)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(6.5 * ncols, 6.0 * nrows))
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[1, 0.07] * nrows,
                          hspace=0.5, wspace=0.4)

    for i, var in enumerate(input_vars):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[2 * r, c])
        cax = fig.add_subplot(gs[2 * r + 1, c])
        data = X[i].copy()
        title = var
        cmap = cmocean.cm.thermal
        symmetric = False
        if "uzonal" in var or "umeridional" in var:
            cmap, symmetric = "RdBu_r", True
        elif var == "olrtoa":
            cmap = "gray_r"
        elif var == "precipw":
            cmap = cmocean.cm.rain
        if var in DENORM:
            mean, std = DENORM[var]
            data = data * std + mean
        else:
            title = f"{var}  (z-score, not de-normalized)"
        panel(ax, cax, lon_c, lat_c, data, cmap, title, symmetric=symmetric)

    fig.suptitle(
        f"Model inputs — GENUINE 64×64 px (16°×16°) training-patch-sized crop, "
        f"piece {int(d['piece_id'])}, frame {int(d['frame'])}\n"
        f"lat {lat_c[0]:.1f}°-{lat_c[-1]:.1f}°N, lon {lon_c[0]:.1f}°-{lon_c[-1]:.1f}°E "
        f"— sliced BEFORE inference (contrast with fig03_inputs.png)",
        fontsize=13,
    )

    out = OUT_DIR / "fig03_inputs_patch64.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
