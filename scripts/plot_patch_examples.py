"""Illustration gallery: per-patch truth/pred precip, segmented systems, and
truth<->pred matching. Generates a bunch of examples to pick from for the
paper. One PNG per example, four panels each:

  (1) truth precipitation field
  (2) predicted precipitation field
  (3) truth systems, coloured by match status (match=green, miss=red)
  (4) pred  systems, coloured by match status (match=green, false_alarm=orange)

Matched truth<->pred systems share an integer label drawn on both panels (3)
and (4) so the reader can read the correspondence at a glance.

Each example is one frame × one geographic crop of about 200×200 grid cells
(~50° × 50°), chosen across leads, regions, and times of day so we can
pick the most illustrative few for the paper.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch

from latentsky import load_run
from latentsky.inference import predict_piece
from latentsky.scale_analysis import (
    cell_area_km2, label_systems, match_systems,
    _DEFAULT_LAT_EXTENT, _DEFAULT_LON_EXTENT,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "figures" / "patch_examples"

# Set of geographic crops worth showcasing. (lat_lo, lat_hi, lon_lo, lon_hi)
# Pick a mix of land/ocean and known convective regions.
WINDOWS = [
    ("maritime_continent", -10.0, 10.0, 90.0, 140.0),
    ("amazon",             -15.0,  5.0, 280.0, 320.0),   # West Africa actually
    ("west_africa",         -5.0, 20.0, 340.0, 380.0),   # crosses 0E
    ("itcz_pacific",        -5.0, 15.0, 180.0, 230.0),
    ("indian_ocean",       -15.0,  5.0,  50.0,  90.0),
    ("western_pacific",    -10.0, 20.0, 130.0, 170.0),
]

# Mix of leads, frame indices (0..23 across the daily sample), pieces.
EXAMPLES = []
for piece in (2, 22):
    for lead_run in ("mseuv_lead1h", "mseuv_lead3h"):
        for frame_idx in (4, 10, 15, 20):    # roughly 04, 10, 15, 20 LST UTC
            EXAMPLES.append((piece, lead_run, frame_idx))


def crop(field, lat_lo, lat_hi, lon_lo, lon_hi, ny, nx):
    """Return (i0, i1, j0, j1) slice for the named geographic window."""
    lat_extent = _DEFAULT_LAT_EXTENT
    lon_extent = _DEFAULT_LON_EXTENT
    dy = (lat_extent[1] - lat_extent[0]) / ny
    dx = (lon_extent[1] - lon_extent[0]) / nx
    i0 = int((lat_lo - lat_extent[0]) / dy)
    i1 = int((lat_hi - lat_extent[0]) / dy)
    # lon_hi may exceed lon_extent[1] (e.g. 0E crossing handled mod 360)
    lon_lo_m = lon_lo % 360.0
    lon_hi_m = (lon_hi - 1e-9) % 360.0
    j0 = int(lon_lo_m / dx)
    j1 = int(lon_hi_m / dx)
    return i0, i1, j0, j1


def take_window(arr2d, sl):
    """Slice a (ny,nx) array, handling lon wrap-around."""
    i0, i1, j0, j1 = sl
    if j1 >= j0:
        return arr2d[i0:i1, j0:j1]
    return np.concatenate([arr2d[i0:i1, j0:], arr2d[i0:i1, :j1]], axis=1)


def plot_example(truth2d, pred2d, area_row_full, sl, lead_h, lst_h, region,
                  threshold, iou_min, out_path):
    """Make the 4-panel illustration for one (truth, pred) pair, cropped to sl."""
    i0, i1, j0, j1 = sl
    truth = take_window(truth2d, sl)
    pred  = take_window(pred2d,  sl)
    area_row = area_row_full[i0:i1]

    # Segmentation on the *cropped* fields (this is how a small system is
    # treated in the global pipeline too — connectivity is local).
    lt, tt = label_systems(truth, threshold, area_row)
    lp, tp = label_systems(pred,  threshold, area_row)
    match_df = match_systems(truth, pred, threshold, area_row, iou_min=iou_min)

    # Build per-label match-status arrays. Indexing aligns with `label`==idx
    # in tt/tp (1-based). For match rows we want to know which truth label
    # was paired with which pred label to colour them green and write a
    # shared integer ID on both.
    matches_t = {}   # truth_label -> shared_id
    matches_p = {}   # pred_label  -> shared_id
    miss_t = set()
    fa_p = set()

    # Re-run the matching to get the label correspondences. match_df has
    # centroid (row,col) but not labels, so rebuild from the IoU pass.
    if len(tt) and len(tp):
        from latentsky.scale_analysis import _overlap_matrix
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
        fa_p   = {l for l in range(1, len(tp) + 1) if (l - 1) not in used_p}
    else:
        miss_t = set(range(1, len(tt) + 1))
        fa_p   = set(range(1, len(tp) + 1))

    # Categorical maps: 0=bg, 1=match, 2=miss (truth-only), 3=fa (pred-only)
    cat_t = np.zeros_like(lt, dtype=int)
    for l in matches_t:
        cat_t[lt == l] = 1
    for l in miss_t:
        cat_t[lt == l] = 2
    cat_p = np.zeros_like(lp, dtype=int)
    for l in matches_p:
        cat_p[lp == l] = 1
    for l in fa_p:
        cat_p[lp == l] = 3

    # ------- plotting -------
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), constrained_layout=True)

    extent_full = (j0 / pred2d.shape[1] * 360.0,
                   (j1 if j1 > j0 else j1 + pred2d.shape[1]) / pred2d.shape[1] * 360.0,
                   i0 / pred2d.shape[0] * 80.0 - 40.0,
                   i1 / pred2d.shape[0] * 80.0 - 40.0)

    vmax = max(np.percentile(truth, 99.5), np.percentile(pred, 99.5), 1.0)
    for ax, fld, ttl in [
        (axes[0], truth, "truth precip [mm]"),
        (axes[1], pred,  "pred precip [mm]"),
    ]:
        im = ax.imshow(fld, origin="lower", extent=extent_full,
                       vmin=0, vmax=vmax, cmap="Blues", aspect="auto")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    cmap_seg = ListedColormap(["white", "#2ca02c", "#d62728", "#ff7f0e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_seg.N)
    axes[2].imshow(cat_t, origin="lower", extent=extent_full,
                   cmap=cmap_seg, norm=norm, aspect="auto")
    axes[2].set_title("truth systems  (green=match, red=miss)", fontsize=10)
    axes[2].set_xlabel("longitude"); axes[2].set_ylabel("latitude")

    axes[3].imshow(cat_p, origin="lower", extent=extent_full,
                   cmap=cmap_seg, norm=norm, aspect="auto")
    axes[3].set_title("pred systems  (green=match, orange=false-alarm)", fontsize=10)
    axes[3].set_xlabel("longitude"); axes[3].set_ylabel("latitude")

    # Annotate matched systems with their shared ID on both segmentation panels.
    H, W = truth.shape
    lat_extent = _DEFAULT_LAT_EXTENT
    lon_extent = _DEFAULT_LON_EXTENT
    def cell_to_xy(rr, cc):
        x = extent_full[0] + (cc + 0.5) * (extent_full[1] - extent_full[0]) / W
        y = extent_full[2] + (rr + 0.5) * (extent_full[3] - extent_full[2]) / H
        return x, y

    from scipy import ndimage as _nd
    for l, sid in matches_t.items():
        rr, cc = _nd.center_of_mass(lt == l)
        x, y = cell_to_xy(rr, cc)
        axes[2].text(x, y, str(sid), fontsize=7, weight="bold",
                     ha="center", va="center", color="black",
                     bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"))
    for l, sid in matches_p.items():
        rr, cc = _nd.center_of_mass(lp == l)
        x, y = cell_to_xy(rr, cc)
        axes[3].text(x, y, str(sid), fontsize=7, weight="bold",
                     ha="center", va="center", color="black",
                     bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"))

    n_match = len(matches_t)
    n_miss = len(miss_t)
    n_fa = len(fa_p)
    fig.suptitle(
        f"{region}  |  lead = {lead_h:g} h  |  LST≈{lst_h:.1f} h  |  "
        f"match={n_match}  miss={n_miss}  fa={n_fa}  |  "
        f"threshold={threshold} mm, iou_min={iou_min}",
        y=1.04, fontsize=10)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--max-examples", type=int, default=24)
    p.add_argument("--keep-frames", type=int, default=24)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # Cache predictions per (run, piece) so we don't reload models or pieces.
    cache: dict[tuple[str, int], tuple] = {}

    n_written = 0
    for piece, lead_run, frame_idx in EXAMPLES:
        if n_written >= args.max_examples:
            break
        key = (lead_run, piece)
        if key not in cache:
            r = load_run(lead_run)
            X, Y, Yp, day_idx, fpd = predict_piece(
                r, piece, batch_size=1, device=device,
                keep_frames=args.keep_frames,
            )
            cache[key] = (Y[..., 0, :, :] if Y.ndim == 4 else Y,
                          Yp[..., 0, :, :] if Yp.ndim == 4 else Yp,
                          day_idx, fpd, r.lead_hours)
        Yfull, Ypfull, day_idx, fpd, lead_h = cache[key]
        if frame_idx >= Yfull.shape[0]:
            continue
        truth2d = Yfull[frame_idx]
        pred2d  = Ypfull[frame_idx]
        ny, nx = truth2d.shape
        area_row_full = cell_area_km2(ny, nx)

        tod_h = (day_idx[frame_idx] * 24.0 / fpd) % 24.0

        for region, lat_lo, lat_hi, lon_lo, lon_hi in WINDOWS:
            if n_written >= args.max_examples:
                break
            sl = crop(truth2d, lat_lo, lat_hi, lon_lo, lon_hi, ny, nx)
            # representative LST at window center longitude
            lon_c = 0.5 * (lon_lo + lon_hi) % 360.0
            lst_h = (tod_h + lon_c / 15.0) % 24.0
            tag = f"{lead_run}_piece{piece:02d}_f{frame_idx:02d}_{region}"
            out_path = OUT_DIR / f"{tag}.png"
            try:
                plot_example(truth2d, pred2d, area_row_full, sl,
                             lead_h, lst_h, region,
                             args.threshold, args.iou_min, out_path)
                n_written += 1
                print(f"[{n_written}] wrote {out_path.name}", flush=True)
            except Exception as e:
                print(f"  skipped {tag}: {e}", file=sys.stderr)

    print(f"\nwrote {n_written} examples to {OUT_DIR}/")


if __name__ == "__main__":
    main()
