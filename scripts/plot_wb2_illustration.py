"""Four-panel illustration of the object-based decomposition for three
WeatherBench2 models — same West Africa region and date as method_illustration.

Layout (2 rows × 4 columns):
  Row 1: ERA5 truth | GraphCast t+6h | FuXi t+6h | GenCast t+12h (÷2 → mm/6h)
  Row 2: ERA5 systems | GC systems    | FuXi systems | GenCast systems

Valid time: 2020-08-15 18:00 UTC (peak convective afternoon over West Africa).
  GraphCast: init 2020-08-15 12Z, lead 6 h
  FuXi:      init 2020-08-15 12Z, lead 6 h
  GenCast:   init 2020-08-15 06Z, lead 12 h  (normalized ÷2 → mm/6h equiv.)
"""
from __future__ import annotations

import argparse
import os
import pathlib

import fsspec
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage as _nd

from latentsky.scale_analysis import (
    cell_area_km2, label_systems, _overlap_matrix,
)

from latentsky.registry import REPO_ROOT as REPO
GC6_CACHE = (pathlib.Path("/glade/derecho/scratch") / os.environ["USER"]
             / "graphcast_eval" / "graphcast_tp6hr_2020-08_tropics.nc")

GRAPHCAST_URL = ("gs://weatherbench2/datasets/graphcast/2020/"
                 "date_range_2019-11-16_2021-02-01_12_hours.zarr")
FUXI_URL      = "gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr"
GENCAST_URL   = "gs://weatherbench2/datasets/gencast/2020-1440x721_mean.zarr"
ERA5_URL      = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

# West Africa window (same as method_illustration defaults)
LAT_LO, LAT_HI = -5.0, 20.0
LON_LO, LON_HI = -15.0, 15.0   # degrees east

VALID_TIME     = np.datetime64("2020-08-15T18:00")  # GC + FuXi valid time
GC_INIT        = np.datetime64("2020-08-15T12:00")
FX_INIT        = np.datetime64("2020-08-15T12:00")
GC_ENS_INIT    = np.datetime64("2020-08-15T12:00")  # GenCast init=12Z, lead=12h
GENCAST_VALID  = np.datetime64("2020-08-16T00:00")  # → valid=00Z Aug 16


def _crop(da: xr.DataArray) -> xr.DataArray:
    for old, new in [("latitude", "lat"), ("longitude", "lon")]:
        if old in da.coords:
            da = da.rename({old: new})
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    # normalise lon to 0-360 if needed
    if da.lon.values[0] < 0:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    lon_lo = LON_LO % 360   # 345
    lon_hi = LON_HI % 360   # 15
    da = da.sel(lat=slice(LAT_LO, LAT_HI))
    if lon_lo > lon_hi:
        # wrap-around: concatenate the two sides of the dateline
        da = xr.concat([da.sel(lon=slice(lon_lo, 360)),
                        da.sel(lon=slice(0, lon_hi))], dim="lon")
    else:
        da = da.sel(lon=slice(lon_lo, lon_hi))
    return da


def _sel_td(ds: xr.Dataset, var: str, td_hours: int) -> xr.DataArray:
    """Select prediction_timedelta, handling int64-hours and timedelta64."""
    da = ds[var]
    td = ds["prediction_timedelta"]
    if np.issubdtype(td.dtype, np.timedelta64):
        return da.sel(prediction_timedelta=np.timedelta64(td_hours, "h"))
    return da.sel(prediction_timedelta=td_hours)


def load_graphcast() -> np.ndarray:
    if GC6_CACHE.exists():
        ds = xr.open_dataset(GC6_CACHE)
        da = ds["total_precipitation_6hr"].sel(valid_time=VALID_TIME)
    else:
        ds = xr.open_zarr(fsspec.get_mapper(GRAPHCAST_URL, anon=True),
                          consolidated=True)
        da = _sel_td(ds, "total_precipitation_6hr", 6).sel(time=GC_INIT)
        da = da * 1000.0  # m → mm
    return _crop(da).values.astype("float32")


def load_fuxi() -> np.ndarray:
    ds = xr.open_zarr(fsspec.get_mapper(FUXI_URL, anon=True),
                      consolidated=True)
    da = _sel_td(ds, "total_precipitation_6hr", 6).sel(time=FX_INIT)
    da = _crop(da)
    print("fuxi loading...", flush=True)
    return da.compute().values.astype("float32")  # already mm


def load_gencast() -> np.ndarray:
    """Returns 12-hr accumulation divided by 2 → mm/6h equivalent."""
    ds = xr.open_zarr(fsspec.get_mapper(GENCAST_URL, anon=True),
                      consolidated=True)
    da = _sel_td(ds, "total_precipitation_12hr", 12).sel(time=GC_ENS_INIT)
    da = _crop(da)
    print("gencast loading...", flush=True)
    return (da.compute().values.astype("float32") * 1000.0) / 2.0  # m→mm, ÷2


def _load_era5_at(valid: np.datetime64) -> np.ndarray:
    ds = xr.open_zarr(fsspec.get_mapper(ERA5_URL, anon=True),
                      consolidated=True)
    da = ds["total_precipitation_6hr"].sel(time=valid)
    da = _crop(da)
    print(f"era5 loading at {valid}...", flush=True)
    return (da.compute().values.astype("float32") * 1000.0)  # m → mm


def load_era5() -> np.ndarray:
    return _load_era5_at(VALID_TIME)


def load_era5_gencast() -> np.ndarray:
    return _load_era5_at(GENCAST_VALID)


def match_labels(truth, pred, threshold, area_row, iou_min):
    lt, tt = label_systems(truth, threshold, area_row)
    lp, tp = label_systems(pred,  threshold, area_row)
    matches_t, matches_p = {}, {}
    miss_t, fa_p = set(), set()
    if len(tt) and len(tp):
        area2d = np.broadcast_to(area_row[:, None], truth.shape)
        inter = _overlap_matrix(lt, lp, len(tt), len(tp), area2d)
        At = tt["A"].to_numpy()
        Ap = tp["A"].to_numpy()
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
        fa_p   = {l for l in range(1, len(tp) + 1) if (l - 1) not in used_p}

    H, W = truth.shape
    cat_t = np.zeros((H, W), dtype=np.uint8)
    cat_p = np.zeros((H, W), dtype=np.uint8)
    for l in range(1, (0 if lt is None else lt.max() + 1) if lt is not None else 1):
        if l in matches_t:
            cat_t[lt == l] = 1
        elif l in miss_t:
            cat_t[lt == l] = 2
    for l in range(1, (0 if lp is None else lp.max() + 1) if lp is not None else 1):
        if l in matches_p:
            cat_p[lp == l] = 1
        elif l in fa_p:
            cat_p[lp == l] = 3
    return cat_t, cat_p, matches_t, matches_p, lt, lp


def add_labels(ax, label_map, matches, extent_xy):
    H, W = label_map.shape
    def xy(r, c):
        x = extent_xy[0] + (c + 0.5) * (extent_xy[1] - extent_xy[0]) / W
        y = extent_xy[2] + (r + 0.5) * (extent_xy[3] - extent_xy[2]) / H
        return x, y
    for l, sid in matches.items():
        rr, cc = _nd.center_of_mass(label_map == l)
        x, y = xy(rr, cc)
        ax.text(x, y, str(sid), fontsize=7, weight="bold",
                ha="center", va="center", color="black",
                bbox=dict(facecolor="white", alpha=0.7, pad=0.5,
                          edgecolor="none"), transform=ax.transData, zorder=12)


def _label_fa(ax, label_map, fa_set, extent_xy):
    """Print 'FA' at centroid of each false-alarm prediction object."""
    H, W = label_map.shape
    def xy(r, c):
        x = extent_xy[0] + (c + 0.5) * (extent_xy[1] - extent_xy[0]) / W
        y = extent_xy[2] + (r + 0.5) * (extent_xy[3] - extent_xy[2]) / H
        return x, y
    for l in fa_set:
        rr, cc = _nd.center_of_mass(label_map == l)
        x, y = xy(rr, cc)
        ax.text(x, y, "FA", fontsize=6, weight="bold",
                ha="center", va="center", color="white",
                bbox=dict(facecolor="#ff7f0e", alpha=0.8, pad=0.5,
                          edgecolor="none"), transform=ax.transData, zorder=12)


def _label_miss(ax, label_map, miss_set, extent_xy):
    """Print 'M' at centroid of each missed truth object."""
    H, W = label_map.shape
    def xy(r, c):
        x = extent_xy[0] + (c + 0.5) * (extent_xy[1] - extent_xy[0]) / W
        y = extent_xy[2] + (r + 0.5) * (extent_xy[3] - extent_xy[2]) / H
        return x, y
    for l in miss_set:
        rr, cc = _nd.center_of_mass(label_map == l)
        x, y = xy(rr, cc)
        ax.text(x, y, "M", fontsize=6, weight="bold",
                ha="center", va="center", color="white",
                bbox=dict(facecolor="#d62728", alpha=0.8, pad=0.5,
                          edgecolor="none"), transform=ax.transData, zorder=12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Wet-pixel threshold in mm/6h (default 0.1).")
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print("Loading data...", flush=True)
    truth     = load_era5()
    truth_gen = load_era5_gencast()
    gc        = load_graphcast()
    fuxi      = load_fuxi()
    gen       = load_gencast()
    print("All loaded.", flush=True)

    ny, nx = truth.shape
    # Pass the actual geographic span (LON_HI - LON_LO = 30°), not modular values
    area_row = cell_area_km2(ny, nx, (LAT_LO, LAT_HI), (LON_LO, LON_HI))

    thr = args.threshold
    # Each model paired with its own ERA5 truth at the correct valid time
    models = [
        ("GraphCast", "t+6h  valid=18Z Aug 15",  gc,   truth),
        ("FuXi",      "t+6h  valid=18Z Aug 15",  fuxi, truth),
        ("GenCast",   "t+12h valid=00Z Aug 16",  gen,  truth_gen),
    ]

    results = []
    for name, lead_str, pred, truth_m in models:
        ct, cp, mt, mp, lt, lp = match_labels(truth_m, pred, thr,
                                               area_row, args.iou_min)
        n_match = len(mt)
        n_miss  = sum(1 for l in range(1, (lt.max() if lt is not None else 0) + 1)
                      if l not in mt)
        n_fa    = sum(1 for l in range(1, (lp.max() if lp is not None else 0) + 1)
                      if l not in mp)
        # combined contingency: 1=match, 2=miss(truth), 3=FA(pred)
        combined = np.zeros_like(ct)
        combined[ct == 2] = 2   # missed truth systems
        combined[cp == 3] = 3   # false alarm prediction systems
        combined[ct == 1] = 1   # matched truth (overwrite, green wins)
        combined[cp == 1] = 1   # matched prediction pixels also green
        results.append((name, lead_str, pred, truth_m, combined, mt, mp, lt, lp,
                        n_match, n_miss, n_fa))

    # ERA5 truth panel: all truth systems in neutral blue (cat=4), labelled
    lt0, _ = label_systems(truth, thr, area_row)
    cat_era5 = np.zeros(truth.shape, dtype=np.uint8)
    n_truth = 0
    if lt0 is not None:
        n_truth = int(lt0.max())
        cat_era5[lt0 > 0] = 4

    extent_xy = (LON_LO, LON_HI, LAT_LO, LAT_HI)
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4, hspace=0.08, wspace=0.06)
    axes_top = [fig.add_subplot(gs[0, k], projection=proj) for k in range(4)]
    axes_bot = [fig.add_subplot(gs[1, k], projection=proj) for k in range(4)]

    vmax = float(max(np.nanpercentile(truth, 99.5),
                     max(np.nanpercentile(m[2], 99.5) for m in results), 1.0))

    # 0=bg, 1=match(green), 2=miss/red, 3=FA/orange, 4=truth-neutral/blue
    cmap_seg = ListedColormap(["white", "#2ca02c", "#d62728", "#ff7f0e", "#4a90d9"])
    norm_seg = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_seg.N)

    # Row 1: precipitation maps
    for ax, arr, title in zip(
        axes_top,
        [truth] + [r[2] for r in results],
        ["ERA5 truth\n18Z Aug 15"] + [f"{r[0]}\n{r[1]}" for r in results],
    ):
        im = ax.imshow(arr, origin="lower", extent=extent_xy,
                       vmin=0, vmax=vmax, cmap="Blues",
                       transform=proj, aspect="auto")
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.coastlines(resolution="50m", linewidth=1.4, color="black", zorder=10)
        ax.set_extent(extent_xy, crs=proj)
        gl = ax.gridlines(draw_labels=(ax is axes_top[0]),
                          linewidth=0.3, color="0.7", alpha=0.4,
                          linestyle=":", zorder=1)
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}

    fig.colorbar(im, ax=axes_top, shrink=0.7, pad=0.02,
                 label="precipitation [mm/6h]")

    # Row 2, col 0: ERA5 truth systems — all systems in neutral blue
    ax0 = axes_bot[0]
    ax0.imshow(cat_era5, origin="lower", extent=extent_xy,
               cmap=cmap_seg, norm=norm_seg, transform=proj, aspect="auto")
    ax0.set_title(f"ERA5 truth systems\n{n_truth} objects", fontsize=10,
                  loc="left", weight="bold")
    ax0.coastlines(resolution="50m", linewidth=1.4, color="black", zorder=10)
    ax0.set_extent(extent_xy, crs=proj)
    gl = ax0.gridlines(draw_labels=True, linewidth=0.3, color="0.7",
                       alpha=0.4, linestyle=":", zorder=1)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
    # label all ERA5 systems
    if lt0 is not None:
        era5_matches = {l: l for l in range(1, n_truth + 1)}
        add_labels(ax0, lt0, era5_matches, extent_xy)

    # Row 2, cols 1-3: per-model contingency maps
    for ax, r in zip(axes_bot[1:], results):
        name, lead_str, pred, truth_m, combined, mt, mp, lt, lp, n_match, n_miss, n_fa = r
        ax.imshow(combined, origin="lower", extent=extent_xy,
                  cmap=cmap_seg, norm=norm_seg, transform=proj, aspect="auto")
        ax.set_title(
            f"{name}  {lead_str}\n"
            f"match={n_match}  miss={n_miss}  FA={n_fa}",
            fontsize=10, loc="left", weight="bold")
        ax.coastlines(resolution="50m", linewidth=1.4, color="black", zorder=10)
        ax.set_extent(extent_xy, crs=proj)
        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7",
                          alpha=0.4, linestyle=":", zorder=1)
        gl.top_labels = False; gl.right_labels = False
        # label matched prediction objects
        if lp is not None and mp:
            add_labels(ax, lp, mp, extent_xy)
        # label false alarm prediction objects
        if lp is not None and lp.max() > 0:
            fa_set = {l for l in range(1, int(lp.max()) + 1) if l not in mp}
            _label_fa(ax, lp, fa_set, extent_xy)
        if lt is not None and lt.max() > 0:
            miss_set = {l for l in range(1, int(lt.max()) + 1) if l not in mt}
            _label_miss(ax, lt, miss_set, extent_xy)

    legend_handles = [
        Patch(facecolor="#4a90d9", edgecolor="black", label="ERA5 truth system"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="match (truth ∩ pred)"),
        Patch(facecolor="#d62728", edgecolor="black", label="miss (truth, not detected)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="false alarm (pred, not in truth)"),
    ]
    axes_bot[-1].legend(handles=legend_handles, loc="lower right",
                        fontsize=8, frameon=True, facecolor="white",
                        framealpha=0.85)

    fig.suptitle(
        f"WB2 models — West Africa, 2020-08-15  "
        f"threshold = {thr:g} mm/6h, IoU ≥ {args.iou_min:g}   "
        "(GenCast 12-hr accumulation normalised ÷2)",
        fontsize=10, y=1.01,
    )

    thr_str = f"{args.threshold:g}".replace(".", "p")
    out = pathlib.Path(args.out or REPO / "docs" / "figures" /
                       f"wb2_illustration_2020-08-15_thr{thr_str}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
