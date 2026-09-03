"""Scale-aware, object-based precipitation decomposition
(recovered from _archive/latentsky/scale_analysis.py, 2026-09-03; method doc:
docs/scale_decomposition_method.md).

Replaces bulk RMSE/correlation framing with a physically interpretable
P = rho * A decomposition, evaluated per precipitating *system* (connected
component) and stratified by system spatial scale.

Definitions (per system, area-weighted so the identity is exact):
    A   = total precipitating area               [km^2]   (pixels > threshold)
    P   = sum(rate * cell_area) over the system   [mm*km^2] (precip "volume")
    rho = P / A                                   [mm]     (mean intensity)
    L   = sqrt(A)                                 [km]     (characteristic scale)

  => P = rho * A holds exactly.

Model error is split log-additively (chosen convention):
    dlnP = ln(P_p/P_t) = ln(A_p/A_t) + ln(rho_p/rho_t) = dlnA + dlnrho
no cross term, scale-free across systems spanning orders of magnitude.

Systems are matched truth<->pred by spatial overlap (greedy max-IoU). An
unmatched truth system is a *miss*; an unmatched predicted system is a
*false alarm*. These detection failures carry precip that the log
decomposition cannot represent (P_t=0 or P_p=0) and are accounted for
separately in the bias budget.

All functions take numpy arrays — no torch, no model — matching the
convention in `latentsky.plotting`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

# Tropical SAM crop, same default as latentsky.plotting.
_DEFAULT_LAT_EXTENT = (-40.0, 40.0)
_DEFAULT_LON_EXTENT = (0.0, 360.0)
R_EARTH_KM = 6371.0


def cell_area_km2(
    ny: int,
    nx: int,
    lat_extent=_DEFAULT_LAT_EXTENT,
    lon_extent=_DEFAULT_LON_EXTENT,
) -> np.ndarray:
    """Per-row grid-cell area in km^2 (lon-uniform on a regular lat/lon grid).

    Returns shape (ny,): area depends only on latitude via cos(lat).
    """
    dlat = (lat_extent[1] - lat_extent[0]) / ny
    dlon = (lon_extent[1] - lon_extent[0]) / nx
    lat_c = lat_extent[0] + (np.arange(ny) + 0.5) * dlat
    dlat_km = np.deg2rad(dlat) * R_EARTH_KM
    dlon_km = np.deg2rad(dlon) * R_EARTH_KM * np.cos(np.deg2rad(lat_c))
    return dlat_km * dlon_km  # (ny,)


def _grid_centers(ny: int, nx: int, lat_extent, lon_extent):
    """Cell-center lat / lon (1-D each), matching latentsky.plotting._grid_latlon."""
    lat = lat_extent[0] + (np.arange(ny) + 0.5) * (lat_extent[1] - lat_extent[0]) / ny
    lon = lon_extent[0] + (np.arange(nx) + 0.5) * (lon_extent[1] - lon_extent[0]) / nx
    return lat, lon


def land_sea_mask(
    ny: int,
    nx: int,
    lat_extent=_DEFAULT_LAT_EXTENT,
    lon_extent=_DEFAULT_LON_EXTENT,
    resolution: str = "50m",
    cache_dir: str | None = None,
) -> np.ndarray:
    """Boolean (ny, nx) grid mask, True = land.

    Rasterizes Natural Earth land polygons (via cartopy, cached shapefile)
    onto the grid cell centers with shapely's vectorized point-in-polygon.
    The (ny, nx, extent, resolution) result is cached to .npy so the
    expensive polygon test runs once, not every eval.
    """
    repo = pathlib.Path(__file__).resolve().parents[3]
    cdir = pathlib.Path(cache_dir) if cache_dir else repo / "data" / "eval_legacy"
    tag = f"{ny}x{nx}_{lat_extent[0]:g}_{lat_extent[1]:g}_{resolution}"
    cache = cdir / f"_landmask_{tag}.npy"
    if cache.exists():
        return np.load(cache)

    import cartopy.io.shapereader as shpreader
    from shapely import contains_xy
    from shapely.geometry import shape
    from shapely.ops import unary_union

    fn = shpreader.natural_earth(resolution=resolution, category="physical",
                                 name="land")
    land = unary_union([shape(r.geometry) for r in shpreader.Reader(fn).records()])

    lat, lon = _grid_centers(ny, nx, lat_extent, lon_extent)
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)  # Natural Earth is -180..180
    LON, LAT = np.meshgrid(lon180, lat)
    mask = contains_xy(land, LON, LAT)

    cdir.mkdir(parents=True, exist_ok=True)
    np.save(cache, mask)
    return mask


def default_scale_edges(min_km: float = 25.0, max_km: float = 3000.0, n: int = 8) -> np.ndarray:
    """Log-spaced characteristic-scale (L = sqrt(area)) bin edges, km.

    Default lower edge ~25 km is roughly one 0.25-deg cell in the tropics;
    upper edge ~3000 km covers synoptic-scale precip envelopes.
    """
    return np.geomspace(min_km, max_km, n + 1)


def _object_table(field: np.ndarray, labels: np.ndarray, n: int,
                   area_row: np.ndarray) -> pd.DataFrame:
    """Per-object A, P, rho, L, centroid for a single labeled 2D field."""
    if n == 0:
        return pd.DataFrame(
            columns=["label", "npix", "A", "P", "rho", "L", "row", "col"]
        )
    idx = np.arange(1, n + 1)
    area2d = np.broadcast_to(area_row[:, None], field.shape)
    A = ndimage.sum(area2d, labels, idx)                 # km^2
    P = ndimage.sum(field * area2d, labels, idx)         # mm*km^2
    npix = ndimage.sum(np.ones_like(field), labels, idx)
    rho = P / A                                          # mm (area-weighted)
    centroids = ndimage.center_of_mass(field, labels, idx)
    centroids = np.atleast_2d(np.asarray(centroids))
    return pd.DataFrame(
        {
            "label": idx,
            "npix": npix.astype(int),
            "A": A,
            "P": P,
            "rho": rho,
            "L": np.sqrt(A),
            "row": centroids[:, 0],
            "col": centroids[:, 1],
        }
    )


def label_systems(field: np.ndarray, threshold: float, area_row: np.ndarray,
                   connectivity: int = 2):
    """Connected-component label one 2D precip field above `threshold`.

    connectivity=2 -> 8-connectivity (diagonal touches join). Returns
    (labels (H,W) int, per-object DataFrame).
    """
    mask = field > threshold
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n = ndimage.label(mask, structure=structure)
    return labels, _object_table(field, labels, n, area_row)


def _overlap_matrix(lt: np.ndarray, lp: np.ndarray, nt: int, npd: int,
                    area2d: np.ndarray) -> np.ndarray:
    """Intersection area (km^2) between truth label i+1 and pred label j+1."""
    inter = np.zeros((nt, npd), dtype=float)
    both = (lt > 0) & (lp > 0)
    if not both.any():
        return inter
    ti = lt[both] - 1
    pj = lp[both] - 1
    np.add.at(inter, (ti, pj), area2d[both])
    return inter


def match_systems(
    truth: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    area_row: np.ndarray,
    iou_min: float = 0.1,
    connectivity: int = 2,
) -> pd.DataFrame:
    """Match truth<->pred systems in one frame by greedy descending IoU.

    Returns a long-form DataFrame, one row per system, with columns:
      role        : 'match' | 'miss' | 'false_alarm'
      A_t,rho_t,P_t,L_t : truth attributes  (NaN for false_alarm)
      A_p,rho_p,P_p,L_p : pred  attributes  (NaN for miss)
      iou         : IoU of the matched pair (NaN otherwise)
      scale       : L_t if truth present else L_p   [km]  (binning key)
      dlnA,dlnrho,dlnP  : log decomposition (matches only; dlnA+dlnrho==dlnP)
    """
    lt, tt = label_systems(truth, threshold, area_row, connectivity)
    lp, tp = label_systems(pred, threshold, area_row, connectivity)
    nt, npd = len(tt), len(tp)
    area2d = np.broadcast_to(area_row[:, None], truth.shape)

    matched_t: set[int] = set()
    matched_p: set[int] = set()
    rows: list[dict] = []

    if nt and npd:
        inter = _overlap_matrix(lt, lp, nt, npd, area2d)
        At = tt["A"].to_numpy()
        Ap = tp["A"].to_numpy()
        union = At[:, None] + Ap[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        order = np.argsort(iou, axis=None)[::-1]
        for flat in order:
            i, j = divmod(int(flat), npd)
            v = iou[i, j]
            if v < iou_min:
                break
            if i in matched_t or j in matched_p:
                continue
            matched_t.add(i)
            matched_p.add(j)
            rt, rp = tt.iloc[i], tp.iloc[j]
            dlnA = np.log(rp["A"] / rt["A"])
            dlnrho = np.log(rp["rho"] / rt["rho"])
            rows.append(
                dict(
                    role="match",
                    A_t=rt["A"], rho_t=rt["rho"], P_t=rt["P"], L_t=rt["L"],
                    A_p=rp["A"], rho_p=rp["rho"], P_p=rp["P"], L_p=rp["L"],
                    iou=float(v), scale=rt["L"],
                    dlnA=dlnA, dlnrho=dlnrho, dlnP=dlnA + dlnrho,
                    row=rt["row"], col=rt["col"],
                    row_p=rp["row"], col_p=rp["col"],
                )
            )

    for i in range(nt):
        if i in matched_t:
            continue
        rt = tt.iloc[i]
        rows.append(
            dict(
                role="miss",
                A_t=rt["A"], rho_t=rt["rho"], P_t=rt["P"], L_t=rt["L"],
                A_p=np.nan, rho_p=np.nan, P_p=np.nan, L_p=np.nan,
                iou=np.nan, scale=rt["L"],
                dlnA=np.nan, dlnrho=np.nan, dlnP=np.nan,
                row=rt["row"], col=rt["col"],
            )
        )
    for j in range(npd):
        if j in matched_p:
            continue
        rp = tp.iloc[j]
        rows.append(
            dict(
                role="false_alarm",
                A_t=np.nan, rho_t=np.nan, P_t=np.nan, L_t=np.nan,
                A_p=rp["A"], rho_p=rp["rho"], P_p=rp["P"], L_p=rp["L"],
                iou=np.nan, scale=rp["L"],
                dlnA=np.nan, dlnrho=np.nan, dlnP=np.nan,
                row=rp["row"], col=rp["col"],
            )
        )

    cols = ["role", "A_t", "rho_t", "P_t", "L_t", "A_p", "rho_p", "P_p",
            "L_p", "iou", "scale", "dlnA", "dlnrho", "dlnP", "row", "col",
            "row_p", "col_p"]
    return pd.DataFrame(rows, columns=cols)


def match_systems_many_to_one(
    truth: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    area_row: np.ndarray,
    iou_min: float = 0.1,
    connectivity: int = 2,
) -> pd.DataFrame:
    """Match truth<->pred systems in one frame, but allow MULTIPLE predicted
    objects to jointly count as the prediction for one truth object, instead
    of match_systems' strict 1-1 greedy-IoU pairing.

    Motivation: a model that adds real small-scale texture (e.g. diffusion)
    can split what should be one connected storm into several separate
    predicted blobs; strict 1-1 matching then only credits whichever single
    fragment wins the IoU contest, making the matched prediction look
    systematically undersized -- the mirror image of an oversmoothing model
    merging several truth systems into one oversized blob (which 1-1
    matching handles fine, since it's still one predicted object to match).

    Each PREDICTED object is assigned to whichever truth object it overlaps
    most (so predicted area is never double-counted across two truth
    systems); every predicted object assigned to a given truth object is
    then summed into one combined A_p/P_p/rho_p before computing
    dlnA/dlnrho/dlnP. A truth object whose merged assignment still doesn't
    clear iou_min counts as a miss (same semantics as match_systems' break
    on iou < iou_min), and any predicted object not assigned to a
    sufficiently-matched truth object is a false_alarm.

    Same return schema as match_systems, so it's a drop-in alternative
    wherever match_systems is called (see system_table's `matching` arg).
    """
    lt, tt = label_systems(truth, threshold, area_row, connectivity)
    lp, tp = label_systems(pred, threshold, area_row, connectivity)
    nt, npd = len(tt), len(tp)
    area2d = np.broadcast_to(area_row[:, None], truth.shape)

    rows: list[dict] = []
    pred_used = np.zeros(npd, dtype=bool)

    if nt and npd:
        inter = _overlap_matrix(lt, lp, nt, npd, area2d)
        best_truth_for_pred = np.argmax(inter, axis=0)
        has_overlap = inter.max(axis=0) > 0
        At = tt["A"].to_numpy()

        for i in range(nt):
            rt = tt.iloc[i]
            assigned = np.where(has_overlap & (best_truth_for_pred == i))[0]
            if len(assigned) == 0:
                rows.append(dict(
                    role="miss", A_t=rt["A"], rho_t=rt["rho"], P_t=rt["P"], L_t=rt["L"],
                    A_p=np.nan, rho_p=np.nan, P_p=np.nan, L_p=np.nan,
                    iou=np.nan, scale=rt["L"], dlnA=np.nan, dlnrho=np.nan, dlnP=np.nan,
                    row=rt["row"], col=rt["col"]))
                continue

            sub = tp.iloc[assigned]
            A_p_sum = float(sub["A"].sum())
            P_p_sum = float(sub["P"].sum())
            inter_sum = float(inter[i, assigned].sum())
            union = At[i] + A_p_sum - inter_sum
            iou_val = inter_sum / union if union > 0 else 0.0

            if iou_val < iou_min:
                rows.append(dict(
                    role="miss", A_t=rt["A"], rho_t=rt["rho"], P_t=rt["P"], L_t=rt["L"],
                    A_p=np.nan, rho_p=np.nan, P_p=np.nan, L_p=np.nan,
                    iou=np.nan, scale=rt["L"], dlnA=np.nan, dlnrho=np.nan, dlnP=np.nan,
                    row=rt["row"], col=rt["col"]))
                continue

            pred_used[assigned] = True
            rho_p_merged = P_p_sum / A_p_sum if A_p_sum > 0 else np.nan
            row_p = float(np.average(sub["row"], weights=sub["A"]))
            col_p = float(np.average(sub["col"], weights=sub["A"]))
            dlnA = np.log(A_p_sum / rt["A"])
            dlnrho = np.log(rho_p_merged / rt["rho"])
            rows.append(dict(
                role="match", A_t=rt["A"], rho_t=rt["rho"], P_t=rt["P"], L_t=rt["L"],
                A_p=A_p_sum, rho_p=rho_p_merged, P_p=P_p_sum, L_p=float(np.sqrt(A_p_sum)),
                iou=float(iou_val), scale=rt["L"],
                dlnA=dlnA, dlnrho=dlnrho, dlnP=dlnA + dlnrho,
                row=rt["row"], col=rt["col"], row_p=row_p, col_p=col_p))
    else:
        for i in range(nt):
            rt = tt.iloc[i]
            rows.append(dict(
                role="miss", A_t=rt["A"], rho_t=rt["rho"], P_t=rt["P"], L_t=rt["L"],
                A_p=np.nan, rho_p=np.nan, P_p=np.nan, L_p=np.nan,
                iou=np.nan, scale=rt["L"], dlnA=np.nan, dlnrho=np.nan, dlnP=np.nan,
                row=rt["row"], col=rt["col"]))

    for j in range(npd):
        if pred_used[j]:
            continue
        rp = tp.iloc[j]
        rows.append(dict(
            role="false_alarm", A_t=np.nan, rho_t=np.nan, P_t=np.nan, L_t=np.nan,
            A_p=rp["A"], rho_p=rp["rho"], P_p=rp["P"], L_p=rp["L"],
            iou=np.nan, scale=rp["L"], dlnA=np.nan, dlnrho=np.nan, dlnP=np.nan,
            row=rp["row"], col=rp["col"]))

    cols = ["role", "A_t", "rho_t", "P_t", "L_t", "A_p", "rho_p", "P_p",
            "L_p", "iou", "scale", "dlnA", "dlnrho", "dlnP", "row", "col",
            "row_p", "col_p"]
    return pd.DataFrame(rows, columns=cols)


def system_table(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    threshold: float = 0.1,
    lat_extent=_DEFAULT_LAT_EXTENT,
    lon_extent=_DEFAULT_LON_EXTENT,
    iou_min: float = 0.1,
    connectivity: int = 2,
    frame_day_idx: np.ndarray | None = None,
    frames_per_day: int = 96,
    add_geo: bool = True,
    matching: str = "greedy",
) -> pd.DataFrame:
    """Build the per-system table for a stack of frames.

    Y_true / Y_pred: (T, 1, H, W) or (T, H, W). Returns the concatenated
    per-frame `match_systems` output with `frame` plus, when add_geo:

      lat, lon   system-centroid location (deg; lon in the lon_extent range)
      is_land    True if the centroid falls on Natural Earth land
      tod_hour   target-frame time of day, 0–24 h (NaN if frame_day_idx None)
      lst_hour   local solar time = (tod_hour + lon/15) mod 24

    `frame_day_idx[t]` is the within-day frame number (0..frames_per_day-1)
    of output frame t, used for the diurnal cycle. Pass None to leave the
    time columns NaN (location columns are still filled).

    `matching`: "greedy" (default, match_systems -- strict 1-1 max-IoU
    pairing) or "many_to_one" (match_systems_many_to_one -- lets several
    predicted objects jointly count as the prediction for one truth object,
    see that function's docstring for why this matters for models that add
    real small-scale texture, e.g. diffusion).
    """
    matcher = {"greedy": match_systems, "many_to_one": match_systems_many_to_one}[matching]
    Yt = np.squeeze(np.asarray(Y_true), axis=1) if Y_true.ndim == 4 else np.asarray(Y_true)
    Yp = np.squeeze(np.asarray(Y_pred), axis=1) if Y_pred.ndim == 4 else np.asarray(Y_pred)
    ny = Yt.shape[-2]
    nx = Yt.shape[-1]
    area_row = cell_area_km2(ny, nx, lat_extent, lon_extent)

    frames = []
    for t in range(Yt.shape[0]):
        df = matcher(Yt[t], Yp[t], threshold, area_row, iou_min, connectivity)
        df.insert(0, "frame", t)
        if frame_day_idx is not None:
            df.insert(1, "day_frame", int(frame_day_idx[t]))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    if add_geo and len(out):
        r = out["row"].to_numpy()
        c = out["col"].to_numpy()
        out["lat"] = lat_extent[0] + (r + 0.5) * (lat_extent[1] - lat_extent[0]) / ny
        out["lon"] = lon_extent[0] + (c + 0.5) * (lon_extent[1] - lon_extent[0]) / nx
        if "row_p" in out.columns:
            rp = out["row_p"].to_numpy(dtype=float)
            cp = out["col_p"].to_numpy(dtype=float)
            out["lat_p"] = lat_extent[0] + (rp + 0.5) * (lat_extent[1] - lat_extent[0]) / ny
            out["lon_p"] = lon_extent[0] + (cp + 0.5) * (lon_extent[1] - lon_extent[0]) / nx
        mask = land_sea_mask(ny, nx, lat_extent, lon_extent)
        ri = np.clip(np.round(r).astype(int), 0, ny - 1)
        ci = np.clip(np.round(c).astype(int), 0, nx - 1)
        out["is_land"] = mask[ri, ci]
        if frame_day_idx is not None:
            tod = (out["day_frame"].to_numpy() * (24.0 / frames_per_day)) % 24.0
            out["tod_hour"] = tod
            out["lst_hour"] = (tod + out["lon"].to_numpy() / 15.0) % 24.0
        else:
            out["tod_hour"] = np.nan
            out["lst_hour"] = np.nan
    return out


@dataclass
class ScaleSummary:
    by_scale: pd.DataFrame   # one row per scale bin: skill + bias budget
    distributions: dict      # {'A'|'rho'|'P': {'truth': arr, 'pred': arr}}


def summarize_by_scale(df: pd.DataFrame, scale_edges: np.ndarray) -> ScaleSummary:
    """Per-scale-bin object skill and log-additive precip-bias budget.

    Per bin (binned on `scale` = truth L, or pred L for false alarms):
      n_truth, n_pred, n_match, n_miss, n_fa
      POD  = hits / (hits + misses)            object detection by size
      FAR  = false_alarms / (hits + fa)
      dlnA_med, dlnrho_med, dlnP_med  + IQR    per-system error by size
      area_frac, int_frac : precip-weighted split of mean dlnP into the
                            area vs intensity contribution (sum to 1)
      miss_P_frac : truth precip in missed systems / total truth precip
      fa_P_frac   : pred  precip in false systems  / total pred  precip
    The last two quantify the detection-failure contribution the log
    decomposition cannot represent.
    """
    if df.empty:
        return ScaleSummary(pd.DataFrame(), {})

    b = np.clip(np.digitize(df["scale"].to_numpy(), scale_edges) - 1,
                0, len(scale_edges) - 2)
    df = df.assign(_bin=b)

    rows = []
    for k in range(len(scale_edges) - 1):
        g = df[df["_bin"] == k]
        if g.empty:
            continue
        m = g[g["role"] == "match"]
        n_match = len(m)
        n_miss = int((g["role"] == "miss").sum())
        n_fa = int((g["role"] == "false_alarm").sum())
        n_truth = n_match + n_miss
        n_pred = n_match + n_fa
        pod = n_match / n_truth if n_truth else np.nan
        far = n_fa / (n_match + n_fa) if (n_match + n_fa) else np.nan

        if n_match:
            w = m["P_t"].to_numpy()
            wsum = w.sum()
            dlnP_w = float(np.sum(w * m["dlnP"]) / wsum) if wsum else np.nan
            dlnA_w = float(np.sum(w * m["dlnA"]) / wsum) if wsum else np.nan
            dlnrho_w = float(np.sum(w * m["dlnrho"]) / wsum) if wsum else np.nan
            denom = abs(dlnA_w) + abs(dlnrho_w)
            area_frac = abs(dlnA_w) / denom if denom else np.nan
            int_frac = abs(dlnrho_w) / denom if denom else np.nan
            dlnA_med = float(m["dlnA"].median())
            dlnrho_med = float(m["dlnrho"].median())
            dlnP_med = float(m["dlnP"].median())
            dlnA_iqr = float(m["dlnA"].quantile(0.75) - m["dlnA"].quantile(0.25))
            dlnrho_iqr = float(m["dlnrho"].quantile(0.75) - m["dlnrho"].quantile(0.25))
        else:
            dlnP_w = dlnA_w = dlnrho_w = area_frac = int_frac = np.nan
            dlnA_med = dlnrho_med = dlnP_med = dlnA_iqr = dlnrho_iqr = np.nan

        P_t_tot = g["P_t"].sum(skipna=True)
        P_p_tot = g["P_p"].sum(skipna=True)
        miss_P = g.loc[g["role"] == "miss", "P_t"].sum()
        fa_P = g.loc[g["role"] == "false_alarm", "P_p"].sum()

        rows.append(
            dict(
                scale_lo=scale_edges[k], scale_hi=scale_edges[k + 1],
                scale_mid=float(np.sqrt(scale_edges[k] * scale_edges[k + 1])),
                n_truth=n_truth, n_pred=n_pred, n_match=n_match,
                n_miss=n_miss, n_fa=n_fa,
                POD=pod, FAR=far,
                dlnA_w=dlnA_w, dlnrho_w=dlnrho_w, dlnP_w=dlnP_w,
                area_frac=area_frac, int_frac=int_frac,
                dlnA_med=dlnA_med, dlnrho_med=dlnrho_med, dlnP_med=dlnP_med,
                dlnA_iqr=dlnA_iqr, dlnrho_iqr=dlnrho_iqr,
                miss_P_frac=float(miss_P / P_t_tot) if P_t_tot else np.nan,
                fa_P_frac=float(fa_P / P_p_tot) if P_p_tot else np.nan,
            )
        )

    by_scale = pd.DataFrame(rows)
    distributions = {
        "A": {
            "truth": df.loc[df["A_t"].notna(), "A_t"].to_numpy(),
            "pred": df.loc[df["A_p"].notna(), "A_p"].to_numpy(),
        },
        "rho": {
            "truth": df.loc[df["rho_t"].notna(), "rho_t"].to_numpy(),
            "pred": df.loc[df["rho_p"].notna(), "rho_p"].to_numpy(),
        },
        "P": {
            "truth": df.loc[df["P_t"].notna(), "P_t"].to_numpy(),
            "pred": df.loc[df["P_p"].notna(), "P_p"].to_numpy(),
        },
    }
    return ScaleSummary(by_scale, distributions)
