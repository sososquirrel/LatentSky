"""Paper Fig 7 (WB2 maps) — GraphCast pair recognition at +6 h, restyled.

A. truth belt / B. GraphCast belt (each full textwidth, same aspect and
coastline weight as the data-section domain map), C./D. the 128-px zoom pair
(equal square panels spanning close to full width, centered), legend row
below. Computation identical to wb2_benchmark/pairs_graphcast.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "wb2_benchmark"))
from pairs_graphcast import CACHE, IOU_MIN, THR, get_landmask, lut  # noqa: E402

sys.path.insert(0, str(REPO / "docs"))
from make_data_overview_assets import NC, NR  # noqa: E402

from latentsky.metrics import cell_area_km2, label_systems, match_systems  # noqa: E402

OUT = REPO / "paper" / "figures"
CM = 1 / 2.54
W2 = 17.5 * CM


def main():
    ds = xr.open_dataset(CACHE / "wb2_6h_graphcast_2020-08-15.nc")
    truth = ds.truth.values[0] * 1000.0
    fc = ds.forecast.values[0] * 1000.0
    ny, nx = truth.shape
    area_row = cell_area_km2(ny, nx, lat_extent=(float(ds.lat[0]) - 0.125,
                                                 float(ds.lat[-1]) + 0.125))
    lt, tt = label_systems(truth, THR, area_row)
    lp, tp = label_systems(fc, THR, area_row)
    mm = match_systems(truth, fc, THR, area_row, iou_min=IOU_MIN)
    lut_t, lut_p = lut(tt), lut(tp)
    matches = mm[mm.role == "match"]
    misses = mm[mm.role == "miss"]
    fas = mm[mm.role == "false_alarm"]

    # matched pairs share a colour across the two maps; missed grey, false alarm red
    MISS, FA = (0.30, 0.30, 0.30), (0.69, 0.227, 0.18)
    pair_colors = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) for h in
                   ("#1f78b4", "#33a02c", "#e6ab02", "#7570b3", "#e7298a", "#66a61e",
                    "#a6761d", "#1b9e77", "#d95f02", "#386cb0", "#f0027f", "#bf5b17",
                    "#8da0cb", "#fc8d62", "#a6d854", "#ffd92f")]
    order = matches.sort_values("A_t", ascending=False)
    color_of_t, color_of_p = {}, {}
    for k, (_, rr) in enumerate(order.iterrows()):
        col = pair_colors[k % len(pair_colors)]
        color_of_t[lut_t[(round(rr["row"], 6), round(rr["col"], 6))]] = col
        color_of_p[lut_p[(round(rr["row_p"], 6), round(rr["col_p"], 6))]] = col
    miss_ids = {lut_t[(round(r["row"], 6), round(r["col"], 6))]
                for _, r in misses.iterrows()}
    fa_ids = {lut_p[(round(r["row"], 6), round(r["col"], 6))]
              for _, r in fas.iterrows()}

    def overlay(lab, color_of, other_ids, other_col):
        edge = np.zeros(lab.shape, bool)
        edge[1:, :] |= lab[1:, :] != lab[:-1, :]
        edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
        edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        edge &= lab > 0
        rgba = np.zeros((*lab.shape, 4), np.float32)
        for lid, col in color_of.items():
            m = lab == lid
            rgba[m] = (*col[:3], 0.55)
            rgba[m & edge] = (*col[:3], 1.0)
        for lid in other_ids:
            m = lab == lid
            rgba[m] = (*other_col, 0.45)
            rgba[m & edge] = (*other_col, 1.0)
        return rgba

    ov_t = overlay(lt, color_of_t, miss_ids, MISS)
    ov_p = overlay(lp, color_of_p, fa_ids, FA)

    # richest 128-px block, merged-miss windows excluded (as on the page)
    P = 128
    vol2 = truth * area_row[:, None]
    missed_mask = (np.isin(lt, list(miss_ids)) if miss_ids
                   else np.zeros_like(lt, bool))
    best, i0, j0 = (-1, -1.0), 0, 0
    for ii in range(0, truth.shape[0] - P + 1, 16):
        for jj in range(0, truth.shape[1] - P + 1, 16):
            w = lt[ii:ii + P, jj:jj + P]
            wm = missed_mask[ii:ii + P, jj:jj + P]
            if wm.any():
                ids, cnts = np.unique(w[wm], return_counts=True)
                if cnts.max() > 400:
                    continue
            nsys = len(np.unique(w)) - (1 if (w == 0).any() else 0)
            score = (nsys, float(vol2[ii:ii + P, jj:jj + P].sum()))
            if score > best:
                best, i0, j0 = score, ii, jj
    zoom = np.s_[i0:i0 + P, j0:j0 + P]
    lon0, lon1 = float(ds.lon[j0]), float(ds.lon[j0 + P - 1])

    norm = LogNorm(vmin=THR, vmax=max(truth.max(), fc.max()))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#ffffff")
    lsm = get_landmask(ds.lat.values, ds.lon.values)

    # ---- layout: as Fig. 1, the belts exactly textwidth with latitude labels to the left
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})
    LABW = 0.85 * CM
    belt_h = W2 * ny / nx
    zoom_w = 0.285 * W2      # the panel size of Fig. 3
    TITLE, GAP, TICK, LEG = 0.5 * CM, 0.35 * CM, 0.45 * CM, 0.7 * CM
    FW = LABW + W2
    FH = 2 * (TITLE + belt_h) + TICK + zoom_w + TITLE + LEG + 2 * GAP
    fig = plt.figure(figsize=(FW, FH))
    lon, lat = ds.lon.values, ds.lat.values

    def title(ax, letter, text):
        ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=9, fontfamily="cmb10",
                ha="left", va="bottom")
        ax.text(0.045 if ax.get_position().width > 0.6 else 0.12, 1.02, text,
                transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom")

    def lon_index(v):
        return int(np.argmin(np.abs(((lon - v) + 180) % 360 - 180)))

    def lat_index(v):
        return int(np.argmin(np.abs(lat - v)))

    y_cursor = FH
    axes = []
    labels = ["Truth (ERA5), valid 2020-08-15 06 UTC", "GraphCast, +6 h"]
    for k, (f, ov) in enumerate([(truth, ov_t), (fc, ov_p)]):
        y_cursor -= TITLE + belt_h
        ax = fig.add_axes([LABW / FW, y_cursor / FH, W2 / FW, belt_h / FH])
        ax.imshow(f, origin="lower", norm=norm, cmap=cmap, interpolation="nearest",
                  aspect="auto")
        ax.imshow(ov, origin="lower", interpolation="nearest", aspect="auto")
        ax.contour(lsm, levels=[0.5], colors=["0.35"], linewidths=0.5)
        ax.add_patch(Rectangle((j0, i0), P, P, fill=False, ec="0.15", lw=1.0, ls="--"))
        xt = [lon_index(v) for v in (0, 90, 180, 270)] + [nx - 1]
        ax.set_xticks(xt)
        ax.set_xticklabels([r"$0^{\circ}$", r"$90^{\circ}$E", r"$180^{\circ}$", r"$90^{\circ}$W",
                            r"$0^{\circ}$"] if k == 1 else [""] * 5)
        ax.set_yticks([lat_index(v) for v in (-40, -20, 0, 20, 40)])
        ax.set_yticklabels([r"$40^{\circ}$S", r"$20^{\circ}$S", r"$0^{\circ}$", r"$20^{\circ}$N",
                            r"$40^{\circ}$N"])
        ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.5)
        for rr_ in range(NR + 1):           # the block grid of Fig. 1 (lines only: the WB2
            ax.axhline(rr_ * P, color="0.15", lw=0.6)   # matching covers the whole belt)
        for cc_ in range(NC + 1):
            ax.axvline(cc_ * P, color="0.15", lw=0.6)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        if k == 1:
            xl = ax.get_xticklabels()
            xl[0].set_ha("left")
            xl[-1].set_ha("right")
        title(ax, "AB"[k] + ".", labels[k])
        axes.append(ax)
        y_cursor -= GAP
    y_cursor -= TICK

    # zoom pair, centred, equal squares
    y_cursor -= TITLE + zoom_w
    x_left = (LABW + (W2 - 2 * zoom_w - 1.0 * CM) / 2) / FW
    zoom_labels = ["Truth, one 128-pixel block", "GraphCast, same block"]
    print(f"zoom block: {lon0:.0f} to {lon1:.0f} E")
    for k, (f, ov) in enumerate([(truth, ov_t), (fc, ov_p)]):
        ax = fig.add_axes([x_left + k * (zoom_w + 1.0 * CM) / FW, y_cursor / FH, zoom_w / FW,
                           zoom_w / FH])
        ax.imshow(f[zoom], origin="lower", norm=norm, cmap=cmap, interpolation="nearest",
                  aspect="auto")
        ax.imshow(ov[zoom], origin="lower", interpolation="nearest", aspect="auto")
        ax.contour(lsm[zoom], levels=[0.5], colors=["0.35"], linewidths=0.5)
        for v in (32, 64, 96):                  # a light grid every 8 degrees, as in Fig. 3
            ax.axhline(v - 0.5, color="0.55", lw=0.3, alpha=0.6, zorder=1.2)
            ax.axvline(v - 0.5, color="0.55", lw=0.3, alpha=0.6, zorder=1.2)
        if k == 0:
            ybar = 10
            latbar = -40 + 0.25 * (i0 + ybar)
            nbar = 500.0 / (27.83 * np.cos(np.deg2rad(latbar)))
            ax.plot([6, 6 + nbar], [ybar, ybar], "-", color="0.1", lw=2.2, solid_capstyle="butt")
            ax.text(6 + nbar / 2, ybar + 4, "500 km", fontsize=7, ha="center", color="0.1")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        title(ax, "CD"[k] + ".", zoom_labels[k])

    from matplotlib.patches import Patch
    handles = [
        Patch(fc=(*pair_colors[0], 0.55), ec=pair_colors[0], lw=1.0,
              label=f"matched pair: same colour in both maps ({len(matches)})"),
        Patch(fc=(*MISS, 0.45), ec=MISS, lw=1.0, label=f"missed: truth system not found ({len(misses)})"),
        Patch(fc=(*FA, 0.45), ec=FA, lw=1.0,
              label=f"false alarm: forecast system matching nothing ({len(fas)})")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5 + LABW / FW / 2, 0.0), handlelength=1.8, columnspacing=1.6)

    fig.savefig(OUT / "fig7a_wb2maps.pdf", dpi=300)
    print("saved fig7a_wb2maps.pdf")


if __name__ == "__main__":
    main()
