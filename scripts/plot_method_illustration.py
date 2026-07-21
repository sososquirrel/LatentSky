"""Paper-quality four-panel illustration of the object-based decomposition,
on a single (frame x region) example. Same logic as plot_patch_examples.py
but tuned for a method-figure: bigger fonts, panel letters (a)-(d), explicit
legend, single colorbar."""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import cmocean
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch

from latentsky import load_run
from latentsky.inference import predict_piece
from latentsky.scale_analysis import (
    cell_area_km2, label_systems, _overlap_matrix,
    _DEFAULT_LAT_EXTENT, _DEFAULT_LON_EXTENT,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "method_illustration.png"

# Defaults: West Africa, lead=1h, frame 15 of 24 (LST ~ 15.8 h).
DEFAULTS = dict(
    run="mseuv_lead1h", piece=2, frame=15,
    lat_lo=-5.0, lat_hi=20.0, lon_lo=340.0, lon_hi=380.0,
    region_name="West Africa",
)


def crop_indices(ny, nx, lat_lo, lat_hi, lon_lo, lon_hi):
    lat_extent = _DEFAULT_LAT_EXTENT
    lon_extent = _DEFAULT_LON_EXTENT
    dy = (lat_extent[1] - lat_extent[0]) / ny
    dx = (lon_extent[1] - lon_extent[0]) / nx
    i0 = int((lat_lo - lat_extent[0]) / dy)
    i1 = int((lat_hi - lat_extent[0]) / dy)
    j0 = int((lon_lo % 360.0) / dx)
    j1 = int(((lon_hi - 1e-9) % 360.0) / dx)
    return i0, i1, j0, j1


def take(arr2d, sl):
    i0, i1, j0, j1 = sl
    if j1 >= j0:
        return arr2d[i0:i1, j0:j1]
    return np.concatenate([arr2d[i0:i1, j0:], arr2d[i0:i1, :j1]], axis=1)


def match_labels(truth, pred, threshold, area_row, iou_min):
    lt, tt = label_systems(truth, threshold, area_row)
    lp, tp = label_systems(pred, threshold, area_row)
    matches_t, matches_p = {}, {}
    miss_t, fa_p = set(), set()
    if len(tt) and len(tp):
        area2d = np.broadcast_to(area_row[:, None], truth.shape)
        inter = _overlap_matrix(lt, lp, len(tt), len(tp), area2d)
        At = tt["A"].to_numpy(); Ap = tp["A"].to_numpy()
        union = At[:, None] + Ap[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        order = np.argsort(iou, axis=None)[::-1]
        used_t, used_p = set(), set()
        sid = 0
        for flat in order:
            i, j = divmod(int(flat), len(tp))
            if iou[i, j] < iou_min:
                break
            if i in used_t or j in used_p:
                continue
            used_t.add(i); used_p.add(j); sid += 1
            matches_t[i + 1] = sid
            matches_p[j + 1] = sid
        miss_t = {l for l in range(1, len(tt) + 1) if (l - 1) not in used_t}
        fa_p = {l for l in range(1, len(tp) + 1) if (l - 1) not in used_p}
    else:
        miss_t = set(range(1, len(tt) + 1))
        fa_p = set(range(1, len(tp) + 1))
    return lt, lp, matches_t, matches_p, miss_t, fa_p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULTS["run"])
    p.add_argument("--piece", type=int, default=DEFAULTS["piece"])
    p.add_argument("--frame", type=int, default=DEFAULTS["frame"])
    p.add_argument("--lat-lo", type=float, default=DEFAULTS["lat_lo"])
    p.add_argument("--lat-hi", type=float, default=DEFAULTS["lat_hi"])
    p.add_argument("--lon-lo", type=float, default=DEFAULTS["lon_lo"])
    p.add_argument("--lon-hi", type=float, default=DEFAULTS["lon_hi"])
    p.add_argument("--region-name", default=DEFAULTS["region_name"])
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--keep-frames", type=int, default=24)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    r = load_run(args.run)
    X, Y, Yp, day_idx, fpd = predict_piece(
        r, args.piece, batch_size=1, device=device, keep_frames=args.keep_frames,
    )
    Y2 = Y[args.frame, 0]
    Yp2 = Yp[args.frame, 0]
    ny, nx = Y2.shape
    area_row_full = cell_area_km2(ny, nx)
    sl = crop_indices(ny, nx, args.lat_lo, args.lat_hi, args.lon_lo, args.lon_hi)
    i0, i1, j0, j1 = sl
    truth = take(Y2, sl)
    pred = take(Yp2, sl)
    area_row = area_row_full[i0:i1]

    lt, lp, matches_t, matches_p, miss_t, fa_p = match_labels(
        truth, pred, args.threshold, area_row, args.iou_min)

    cat_t = np.zeros_like(lt, dtype=int)
    for l in matches_t: cat_t[lt == l] = 1
    for l in miss_t:    cat_t[lt == l] = 2
    cat_p = np.zeros_like(lp, dtype=int)
    for l in matches_p: cat_p[lp == l] = 1
    for l in fa_p:      cat_p[lp == l] = 3

    tod_h = (day_idx[args.frame] * 24.0 / fpd) % 24.0
    lon_c = 0.5 * (args.lon_lo + args.lon_hi) % 360.0
    lst_h = (tod_h + lon_c / 15.0) % 24.0

    # Use longitudes in -180..180 for cartopy. West Africa crosses 0E, so
    # lon_lo=340 / lon_hi=380 maps to -20 / +20.
    lon0 = j0 / nx * 360.0
    lon1 = (j1 if j1 > j0 else j1 + nx) / nx * 360.0
    if lon0 > 180: lon0 -= 360
    if lon1 > 180: lon1 -= 360
    lat0 = i0 / ny * 80.0 - 40.0
    lat1 = i1 / ny * 80.0 - 40.0
    extent_xy = (lon0, lon1, lat0, lat1)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, wspace=0.08)
    axes = [fig.add_subplot(gs[0, k], projection=proj) for k in range(4)]

    vmax = float(max(np.percentile(truth, 99.5), np.percentile(pred, 99.5), 1.0))
    im0 = axes[0].imshow(truth, origin="lower", extent=extent_xy,
                          vmin=0, vmax=vmax, cmap=cmocean.cm.rain,
                          transform=proj, aspect="auto")
    im1 = axes[1].imshow(pred,  origin="lower", extent=extent_xy,
                          vmin=0, vmax=vmax, cmap=cmocean.cm.rain,
                          transform=proj, aspect="auto")
    cb = fig.colorbar(im1, ax=axes[:2], shrink=0.85, pad=0.02,
                       label="precipitation rate [mm]")
    cmap_seg = ListedColormap(["white", "#2ca02c", "#d62728", "#ff7f0e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_seg.N)
    axes[2].imshow(cat_t, origin="lower", extent=extent_xy,
                   cmap=cmap_seg, norm=norm,
                   transform=proj, aspect="auto")
    axes[3].imshow(cat_p, origin="lower", extent=extent_xy,
                   cmap=cmap_seg, norm=norm,
                   transform=proj, aspect="auto")

    for ax in axes:
        ax.set_extent(extent_xy, crs=proj)
        # Coastlines on top of imshow, thick + dark so they read against
        # the white background and the colored segmentation patches.
        ax.coastlines(resolution="50m", linewidth=1.6, color="black",
                      zorder=10)
        gl = ax.gridlines(draw_labels=True, linewidth=0.7, color="0.3",
                           alpha=0.7, linestyle="--", zorder=9)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}

    titles = [
        "(a) truth precipitation",
        "(b) predicted precipitation",
        "(c) truth systems",
        "(d) predicted systems",
    ]
    for ax, ttl in zip(axes, titles):
        ax.set_title(ttl, fontsize=12, loc="left", weight="bold")

    # Numeric IDs on matched systems (shared ID across c and d).
    from scipy import ndimage as _nd
    H, W = truth.shape
    def cell_to_xy(rr, cc):
        x = extent_xy[0] + (cc + 0.5) * (extent_xy[1] - extent_xy[0]) / W
        y = extent_xy[2] + (rr + 0.5) * (extent_xy[3] - extent_xy[2]) / H
        return x, y
    for l, sid in matches_t.items():
        rr, cc = _nd.center_of_mass(lt == l)
        x, y = cell_to_xy(rr, cc)
        axes[2].text(x, y, str(sid), fontsize=8, weight="bold",
                     ha="center", va="center", color="black",
                     bbox=dict(facecolor="white", alpha=0.7, pad=0.6,
                                edgecolor="none"))
    for l, sid in matches_p.items():
        rr, cc = _nd.center_of_mass(lp == l)
        x, y = cell_to_xy(rr, cc)
        axes[3].text(x, y, str(sid), fontsize=8, weight="bold",
                     ha="center", va="center", color="black",
                     bbox=dict(facecolor="white", alpha=0.7, pad=0.6,
                                edgecolor="none"))

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="match (truth ↔ pred)"),
        Patch(facecolor="#d62728", edgecolor="black", label="miss (truth only)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="false alarm (pred only)"),
    ]
    axes[3].legend(handles=legend_handles, loc="lower right", fontsize=9,
                   frameon=True, facecolor="white", framealpha=0.85)

    n_match = len(matches_t)
    n_miss = len(miss_t)
    n_fa = len(fa_p)
    fig.suptitle(
        f"Object-based precipitation decomposition — {args.region_name}, "
        f"lead = {r.lead_hours:g} h, LST ≈ {lst_h:.1f} h.   "
        f"Systems: {n_match} matched, {n_miss} missed, {n_fa} false alarm.   "
        f"Threshold = {args.threshold} mm, IoU ≥ {args.iou_min}.",
        y=1.04, fontsize=11)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
