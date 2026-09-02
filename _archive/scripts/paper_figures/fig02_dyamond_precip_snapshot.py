"""Figure: large snapshot of DYAMOND precipitation over the full tropics
domain (lat -40..40, lon 0..360), SAME time/piece/frame as fig01.

rain_total is stored raw (mm/15min) in the piece files, no inversion needed
(unlike precipw, which is z-scored).
"""
from __future__ import annotations
import pathlib
import numpy as np
import cartopy.crs as ccrs
import cmocean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
PIECES_DIR = pathlib.Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
PIECE_ID = 2
FRAME = 60  # same snapshot used for figs 01, 02, 03 (must satisfy FRAME+delta_index<96 for lead6h)

LAT_EXTENT = (-40.0, 40.0)
LON_EXTENT = (0.0, 360.0)

# Illustrative patch-size grid: 16x16deg cells (matching the 64x64px
# training patch size at 0.25deg/px). Simple reference grid only, not an
# exact rendering of the true tiling/leftover-strip edge case.
PATCH_DEG = 16.0


def grid_latlon(ny, nx):
    lat = LAT_EXTENT[0] + (np.arange(ny) + 0.5) * (LAT_EXTENT[1] - LAT_EXTENT[0]) / ny
    lon = LON_EXTENT[0] + (np.arange(nx) + 0.5) * (LON_EXTENT[1] - LON_EXTENT[0]) / nx
    return lat, lon


def main():
    rain = np.load(PIECES_DIR / f"rain_total_piece{PIECE_ID:02d}.npy", mmap_mode="r")[FRAME]
    rain = np.asarray(rain)
    ny, nx = rain.shape
    lat, lon = grid_latlon(ny, nx)

    vmin_log = 0.1
    vmax = float(max(rain.max(), vmin_log * 10))
    norm = LogNorm(vmin=vmin_log, vmax=vmax)
    levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 25)

    # Standard, familiar Africa/Atlantic-centered view.
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(16, 6), subplot_kw={"projection": proj},
                           constrained_layout=True)
    im = ax.contourf(lon, lat, rain, levels=levels, cmap=cmocean.cm.rain,
                     norm=norm, extend="both", transform=proj)
    ax.coastlines(resolution="110m", linewidth=0.6, color="0.2")
    ax.set_extent([-180.0, 180.0, LAT_EXTENT[0], LAT_EXTENT[1]], crs=proj)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    # Thick reference grid, 16x16deg cells, full domain, no marked leftover
    # region. Horizontal lines need many interpolated points, not just the
    # 2 endpoints -- a 2-point line is drawn as a naive straight connection
    # in transformed space and doesn't trace across the map's wraparound.
    lat_edges = np.arange(LAT_EXTENT[0], LAT_EXTENT[1] + 1e-6, PATCH_DEG)
    lon_edges = np.arange(LON_EXTENT[0], LON_EXTENT[1] + 1e-6, PATCH_DEG)
    lon_line = np.linspace(LON_EXTENT[0], LON_EXTENT[1], 400)
    for le in lat_edges:
        ax.plot(lon_line, np.full_like(lon_line, le), transform=proj,
                color="k", lw=1.4, alpha=0.9, zorder=7)
    for lo in lon_edges:
        ax.plot([lo, lo], [LAT_EXTENT[0], LAT_EXTENT[1]], transform=proj,
                color="k", lw=1.4, alpha=0.9, zorder=7)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("precipitation rate (mm / 15min, log scale)", fontsize=11)
    ax.set_title(f"DYAMOND precipitation — tropics ±40°  "
                 f"(piece {PIECE_ID}, frame {FRAME})  "
                 f"— grid: ~16°×16° patches", fontsize=13)

    out = OUT_DIR / "fig02_dyamond_precip_snapshot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
