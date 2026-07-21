"""Population diurnal cycle of small precipitating systems.

For each dataset, counts small systems (L < SMALL_KM) by local solar time
and normalises by the daily mean — so 1.0 = average, 1.5 = 50% more than
average.  This tests whether a model reproduces *when* convection occurs,
independent of whether it locates the exact same cells (which is not possible
at 6 h lead for km-scale convection).

Sources (6-hr window, threshold = 0.4 mm/hr):
  LatentSky    : docs/scale_systems_v3.csv.gz        (0.1 mm/15min, 6h lead)
  GraphCast 6h : docs/scale_systems_graphcast_2020-08_thr2p4.csv.gz
  FuXi 6h      : docs/scale_systems_fuxi_2020-08_6h.csv.gz

Truth references:
  DYAMOND truth: extracted from LatentSky CSV (role = match | miss)
  ERA5 truth   : extracted from GraphCast CSV (role = match | miss)
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO     = pathlib.Path(__file__).resolve().parents[1]
SMALL_KM = 65
LST_EDGES = np.arange(0, 25, 2)
LST_MID   = 0.5 * (LST_EDGES[:-1] + LST_EDGES[1:])


def _smooth(arr: np.ndarray, w: int = 2) -> np.ndarray:
    """Circular rolling mean over ±w bins (wraps 24h)."""
    out = np.empty_like(arr)
    n = len(arr)
    for i in range(n):
        idxs = [(i + k) % n for k in range(-w, w + 1)]
        out[i] = np.nanmean(arr[idxs])
    return out


def _count_by_lst(df: pd.DataFrame, kind: str, is_land: bool) -> np.ndarray:
    """Count small systems by LST bin, normalised by daily mean.

    kind = 'truth'  → role in {match, miss},  size = L_t  (truth systems)
    kind = 'pred'   → role in {match, false_alarm}, size = L_p (predicted)
    """
    if kind == "truth":
        mask = df["role"].isin(["match", "miss"])
        size = df["L_t"]
    else:
        mask = df["role"].isin(["match", "false_alarm"])
        size = df["L_p"].fillna(df["L_t"])   # L_p NaN only for misses (excluded)
    sub = df[mask & (size < SMALL_KM) & (df["is_land"] == is_land)].copy()
    sub["b"] = np.digitize(sub["lst_hour"] % 24, LST_EDGES) - 1
    sub = sub[(sub["b"] >= 0) & (sub["b"] < len(LST_MID))]
    counts = np.array([(sub["b"] == b).sum() for b in range(len(LST_MID))],
                      dtype=float)
    mean = counts.mean()
    return counts / mean if mean > 0 else np.ones_like(counts)


def load(path: pathlib.Path, lead_h=None, pred_set=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if lead_h is not None:
        df = df[df["lead_h"] == lead_h]
    if pred_set is not None:
        df = df[df["predictor_set"] == pred_set]
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--small-km", type=float, default=65)
    args = p.parse_args()
    global SMALL_KM
    SMALL_KM = args.small_km

    ls_df = load(REPO / "docs/scale_systems_v3.csv.gz",
                 lead_h=6.0, pred_set="9var (mse+uv)")
    gc_df = load(REPO / "docs/scale_systems_graphcast_2020-08_thr2p4.csv.gz")
    fx_df = load(REPO / "docs/scale_systems_fuxi_2020-08_6h.csv.gz")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True,
                             sharey=True)

    for ax, is_land, title in [
        (axes[0], True,  "(a) Land"),
        (axes[1], False, "(b) Ocean"),
    ]:
        # --- truth reference lines ---
        dyamond = _count_by_lst(ls_df, "truth", is_land)
        era5    = _count_by_lst(gc_df, "truth", is_land)

        # LatentSky has dense sampling → light smoothing; WB2 has only 2 UTC
        # times so LST bins have high sampling noise → stronger smoothing
        ax.plot(LST_MID, _smooth(dyamond, 1), color="black", lw=2.5, ls="-",
                label="DYAMOND truth (km-scale)")
        ax.plot(LST_MID, _smooth(era5,    2), color="0.55",  lw=2.5, ls="-",
                label="ERA5 truth (31 km)")

        # --- predicted diurnal cycles ---
        ls_pred = _count_by_lst(ls_df, "pred", is_land)
        gc_pred = _count_by_lst(gc_df, "pred", is_land)
        fx_pred = _count_by_lst(fx_df, "pred", is_land)

        ax.plot(LST_MID, _smooth(ls_pred, 1), color="#1f77b4", lw=2,   ls="--",
                label="LatentSky 6h (DYAMOND-trained)")
        ax.plot(LST_MID, _smooth(gc_pred, 2), color="#e6194b", lw=1.8, ls="--",
                label="GraphCast 6h (ERA5-trained)")
        ax.plot(LST_MID, _smooth(fx_pred, 2), color="#3cb44b", lw=1.8, ls="-.",
                label="FuXi 6h (ERA5-trained)")

        ax.axhline(1.0, color="k", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlabel("local solar time [h]", fontsize=11)
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("system count / daily mean", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Population diurnal cycle — small precipitating systems (L < {SMALL_KM} km)\n"
        "August 2020, tropics ±40°.  Threshold: 0.1 mm/15min (LatentSky) "
        "= 2.4 mm/6h (WB2) = 0.4 mm/hr.  Counts normalised by daily mean.",
        fontsize=9,
    )

    km_str = f"{int(args.small_km)}km"
    out = pathlib.Path(
        args.out or REPO / "docs" / "figures" / f"diurnal_population_wb2_vs_latentsky_{km_str}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
