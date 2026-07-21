"""Distribution of precipitating system sizes: truth vs AI forecast models.

X axis: characteristic scale L = sqrt(area)  [km, log scale]
Y axis: probability density (normalised so area under curve = 1)

Truth systems : role in {match, miss},       size = L_t
Predicted systems: role in {match, false_alarm}, size = L_p
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from latentsky.registry import REPO_ROOT as REPO

L_MIN, L_MAX = 30, 2000   # km
L_GRID = np.geomspace(L_MIN, L_MAX, 300)


def kde(values: np.ndarray) -> np.ndarray:
    v = values[(values >= L_MIN) & (values <= L_MAX)]
    if len(v) < 10:
        return np.zeros_like(L_GRID)
    lv = np.log(v)
    bw = v.std() / v.mean() * 0.4   # heuristic bandwidth in log space
    k = gaussian_kde(lv, bw_method=bw)
    return k(np.log(L_GRID))


def extract(df: pd.DataFrame, kind: str, is_land: bool | None = None) -> np.ndarray:
    if is_land is not None:
        df = df[df["is_land"] == is_land]
    if kind == "truth":
        sub = df[df["role"].isin(["match", "miss"])]
        return sub["L_t"].dropna().values
    else:
        sub = df[df["role"].isin(["match", "false_alarm"])]
        return sub["L_p"].fillna(sub["L_t"]).dropna().values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # --- load data ---
    ls_df = pd.read_csv(REPO / "docs/scale_systems_v3.csv.gz")
    ls_df = ls_df[(ls_df["lead_h"] == 6.0) &
                  (ls_df["predictor_set"] == "9var (mse+uv)")]
    # Use 6hr WB2 data for fair comparison (same ~0.4 mm/hr rain-rate threshold)
    gc_df = pd.read_csv(REPO / "docs/scale_systems_graphcast_2020-08_thr2p4.csv.gz")
    fx_df = pd.read_csv(REPO / "docs/scale_systems_fuxi_2020-08_6h.csv.gz")
    # GenCast has no 6hr data — use 24hr with note
    gn_df = pd.read_csv(REPO / "docs/scale_systems_gencast_2020-08_24h_thr9p6.csv.gz")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True,
                             sharey=False)

    for ax, is_land, title in [
        (axes[0], True,  "(a) Land"),
        (axes[1], False, "(b) Ocean"),
    ]:
        # truth references
        dyamond_t = kde(extract(ls_df, "truth", is_land))
        era5_t    = kde(extract(gc_df, "truth", is_land))

        ax.plot(L_GRID, dyamond_t, color="black", lw=2.5, ls="-",
                label="DYAMOND truth (km-scale)")
        ax.plot(L_GRID, era5_t,   color="0.55",  lw=2.5, ls="-",
                label="ERA5 truth (31 km)")

        # predictions
        ls_p  = kde(extract(ls_df, "pred", is_land))
        gc_p  = kde(extract(gc_df, "pred", is_land))
        fx_p  = kde(extract(fx_df, "pred", is_land))
        gn_p  = kde(extract(gn_df, "pred", is_land))

        ax.plot(L_GRID, ls_p, color="#1f77b4", lw=2,   ls="--",
                label="LatentSky 6h")
        ax.plot(L_GRID, gc_p, color="#e6194b", lw=1.8, ls="--",
                label="GraphCast 6h")
        ax.plot(L_GRID, fx_p, color="#3cb44b", lw=1.8, ls="-.",
                label="FuXi 6h")
        ax.plot(L_GRID, gn_p, color="#4363d8", lw=1.8, ls=":",
                label="GenCast 24h*")

        ax.set_xscale("log")
        ax.set_xlim(L_MIN, L_MAX)
        ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
        ax.set_ylabel("probability density", fontsize=11)
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.grid(alpha=0.3, which="both")
        ax.axvline(56, color="0.4", lw=0.8, ls=":", alpha=0.7)   # ERA5 Nyquist
        ax.text(58, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.01,
                "ERA5\nNyquist\n(56 km)", fontsize=7, color="0.4", va="top")

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Distribution of precipitating system sizes — AI models vs truth\n"
        "August 2020, tropics ±40°.  Threshold: 0.1 mm/15min (LatentSky) "
        "= 2.4 mm/6h (GC/FuXi) = 0.4 mm/hr.  *GenCast: 9.6 mm/24h.",
        fontsize=9,
    )

    out = pathlib.Path(
        args.out or REPO / "docs" / "figures" / "system_size_dist_wb2_vs_latentsky.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
