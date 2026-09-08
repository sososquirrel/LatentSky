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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "wb2_benchmark"))
from pairs_graphcast import CACHE, IOU_MIN, THR, get_landmask, lut  # noqa: E402

from latentsky.metrics import cell_area_km2, label_systems, match_systems  # noqa: E402

matplotlib.rcParams["savefig.bbox"] = "tight"
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

    order = matches.sort_values("A_t", ascending=False)
    t20 = plt.get_cmap("tab20")
    pair_colors = [t20(i) for i in range(20) if i not in (6, 7)]
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
            rgba[m] = (*col[:3], 0.28)
            rgba[m & edge] = (*col[:3], 1.0)
        for lid in other_ids:
            m = lab == lid
            rgba[m] = (*other_col, 0.22)
            rgba[m & edge] = (*other_col, 0.95)
        return rgba

    ov_t = overlay(lt, color_of_t, miss_ids, (0.05, 0.05, 0.05))
    ov_p = overlay(lp, color_of_p, fa_ids, (0.84, 0.15, 0.16))

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

    # ---- layout: two belts (aspect ny/nx at textwidth) + zoom pair --------
    belt_h = W2 * ny / nx
    zoom_w = 0.40 * W2
    LAB, GAP, LEG = 0.55 * CM, 0.35 * CM, 0.8 * CM
    FH = 2 * (belt_h + LAB) + zoom_w + LAB + LEG + 2 * GAP
    fig = plt.figure(figsize=(W2, FH))

    def frac(y, h):
        return y / FH, h / FH

    y_cursor = FH
    axes = []
    labels = ["A. truth (ERA5, valid 2020-08-15 06 UTC)",
              "B. GraphCast +6 h"]
    for k, (f, ov) in enumerate([(truth, ov_t), (fc, ov_p)]):
        y_cursor -= LAB + belt_h
        ax = fig.add_axes([0.0, y_cursor / FH, 1.0, belt_h / FH])
        ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                  interpolation="nearest", aspect="auto")
        ax.imshow(ov, origin="lower", interpolation="nearest", aspect="auto")
        ax.contour(lsm, levels=[0.5], colors=["0.3"], linewidths=0.6)
        ax.add_patch(Rectangle((j0, i0), P, P, fill=False, ec="0.2", lw=1.2,
                               ls="--"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.0, 1.01, labels[k], transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="bottom")
        axes.append(ax)
        y_cursor -= GAP

    # zoom pair, centered, equal squares
    y_cursor -= LAB + zoom_w
    x_left = (W2 - 2 * zoom_w - 1.4 * CM) / 2 / W2
    zoom_labels = [f"C. truth — 128-px block ({lon0:.0f}–{lon1:.0f}°E)",
                   "D. GraphCast +6 h — same block"]
    for k, (f, ov) in enumerate([(truth, ov_t), (fc, ov_p)]):
        ax = fig.add_axes([x_left + k * (zoom_w + 1.4 * CM) / W2,
                           y_cursor / FH, zoom_w / W2, zoom_w / FH])
        ax.imshow(f[zoom], origin="lower", norm=norm, cmap=cmap,
                  interpolation="nearest", aspect="auto")
        ax.imshow(ov[zoom], origin="lower", interpolation="nearest",
                  aspect="auto")
        ax.contour(lsm[zoom], levels=[0.5], colors=["0.3"], linewidths=0.6)
        if k == 0:
            ybar = 10
            latbar = -40 + 0.25 * (i0 + ybar)
            px_km = 27.83 * np.cos(np.deg2rad(latbar))
            nbar = 500.0 / px_km
            ax.plot([6, 6 + nbar], [ybar, ybar], "-", color="0.1", lw=2.5,
                    solid_capstyle="butt")
            ax.text(6 + nbar / 2, ybar + 4, "500 km", fontsize=7.5,
                    ha="center", color="0.1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.0, 1.01, zoom_labels[k], transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="left", va="bottom")

    handles = [
        Line2D([], [], color=pair_colors[0], lw=4,
               label=f"matched pair — same color both panels ({len(matches)})"),
        Line2D([], [], color="0.05", lw=4, alpha=0.55,
               label=f"miss ({len(misses)})"),
        Line2D([], [], color="#d62728", lw=4, alpha=0.55,
               label=f"false alarm ({len(fas)})")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.savefig(OUT / "fig7a_wb2maps.pdf", dpi=300)
    print("saved fig7a_wb2maps.pdf")


if __name__ == "__main__":
    main()
