"""Figure: method illustration over West Africa, lead=3h, 9var (mse+uv).

2 rows x 2 cols:
  (a) truth precip           (b) prediction precip
  (c) truth + detected systems outlined   (d) prediction + detected systems outlined

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

from latentsky.scale_analysis import label_systems, cell_area_km2

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
CACHE = OUT_DIR / "_inference_cache.npz"

LAT_EXTENT = (-40.0, 40.0)
LON_EXTENT = (0.0, 360.0)
THRESHOLD = 0.1


def grid_latlon(ny, nx):
    lat = LAT_EXTENT[0] + (np.arange(ny) + 0.5) * (LAT_EXTENT[1] - LAT_EXTENT[0]) / ny
    lon = LON_EXTENT[0] + (np.arange(nx) + 0.5) * (LON_EXTENT[1] - LON_EXTENT[0]) / nx
    return lat, lon


def main():
    d = np.load(CACHE)
    yt = d["yt_wa"]
    yp = d["yp_wa"]
    r0, r1 = int(d["wa_lat_idx"][0]), int(d["wa_lat_idx"][1])
    c0, c1 = int(d["wa_lon_idx"][0]), int(d["wa_lon_idx"][1])

    lat_full, lon_full = grid_latlon(320, 1440)
    lat_c, lon_c = lat_full[r0:r1], lon_full[c0:c1]
    area_row_full = cell_area_km2(320, 1440, LAT_EXTENT, LON_EXTENT)
    area_row = area_row_full[r0:r1]

    labels_t, obj_t = label_systems(yt, THRESHOLD, area_row)
    labels_p, obj_p = label_systems(yp, THRESHOLD, area_row)
    print(f"truth systems: {len(obj_t)}   pred systems: {len(obj_p)}", flush=True)

    vmin_log = 0.1
    vmax = float(max(yt.max(), yp.max(), vmin_log * 10))
    norm = LogNorm(vmin=vmin_log, vmax=vmax)
    levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 20)

    fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    def _base(ax, data, title):
        im = ax.contourf(lon_c, lat_c, data, levels=levels, cmap=cmocean.cm.rain,
                         norm=norm, extend="both")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("lon (°E)", fontsize=9)
        ax.set_ylabel("lat (°N)", fontsize=9)
        ax.tick_params(labelsize=8)
        return im

    im = _base(axs[0, 0], yt, "(a) truth — precip, lead=3h")
    _base(axs[0, 1], yp, "(b) prediction — precip, lead=3h")
    fig.colorbar(im, ax=axs[0, :].tolist(), orientation="vertical",
                fraction=0.025, pad=0.02, label="precip (mm, log)")

    _base(axs[1, 0], yt, "(c) truth + detected systems")
    axs[1, 0].contour(lon_c, lat_c, labels_t > 0, levels=[0.5], colors="red", linewidths=1.2)

    _base(axs[1, 1], yp, "(d) prediction + detected systems")
    axs[1, 1].contour(lon_c, lat_c, labels_p > 0, levels=[0.5], colors="red", linewidths=1.2)

    fig.suptitle(
        f"Method illustration — West Africa, lead=3h  "
        f"(threshold={THRESHOLD} mm, {len(obj_t)} truth / {len(obj_p)} pred systems detected)",
        fontsize=12,
    )

    out = OUT_DIR / "fig05_west_africa_method.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
