"""Distribution comparison: GraphCast vs LatentSky, truth vs pred.

The honest problem: LatentSky is verified against SAM 15-min instantaneous
precip; GraphCast against ERA5 6h-accumulated precip. A 6h accumulation has
larger, more-merged wet regions, so GraphCast systems are systematically
bigger than LatentSky systems for reasons that have nothing to do with model
skill. Comparing the raw size distributions is apples-to-oranges.

This figure shows three views:
  (a) raw system-size distribution   -> exposes the accumulation offset
  (b) size rescaled by each TRUTH's median L -> the apples-to-apples attempt:
      divide out the offset, compare the *shape* and the model's distortion
  (c) intensity (rho) distribution, density-normalized

In every panel the question is the same: does the model's predicted
distribution (dashed) match its own truth (solid), and do the two models
distort their truth the same way?
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "dist_compare_graphcast_latentsky_6h.png"

LS_SYS = REPO / "docs" / "scale_systems_v3.csv.gz"
GC_SYS = REPO / "docs" / "scale_systems_graphcast_2020-08.csv.gz"

COL = {"LatentSky": "#1f77b4", "GraphCast": "#d62728"}


def load_sides(path, lead=None, predictor=None):
    df = pd.read_csv(path)
    if lead is not None:
        df = df[df["lead_h"] == lead]
    if predictor is not None:
        df = df[df["predictor_set"] == predictor]
    Lt = df.loc[df["L_t"].notna(), "L_t"].to_numpy()
    Lp = df.loc[df["L_p"].notna(), "L_p"].to_numpy()
    rt = df.loc[df["rho_t"].notna(), "rho_t"].to_numpy()
    rp = df.loc[df["rho_p"].notna(), "rho_p"].to_numpy()
    return dict(Lt=Lt, Lp=Lp, rt=rt, rp=rp)


def step_hist(ax, vals, bins, color, ls, label, density=True):
    ax.hist(vals[vals > 0], bins=bins, histtype="step", color=color,
            lw=1.9, ls=ls, density=density, label=label)


def main():
    ls = load_sides(LS_SYS, lead=6.0, predictor="9var (mse+uv)")
    gc = load_sides(GC_SYS)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    # ---- (a) raw size distribution (the oranges) ----
    be = np.geomspace(20, 1500, 35)
    step_hist(ax[0], ls["Lt"], be, COL["LatentSky"], "-",  "LatentSky truth (SAM)")
    step_hist(ax[0], ls["Lp"], be, COL["LatentSky"], "--", "LatentSky pred")
    step_hist(ax[0], gc["Lt"], be, COL["GraphCast"], "-",  "GraphCast truth (ERA5)")
    step_hist(ax[0], gc["Lp"], be, COL["GraphCast"], "--", "GraphCast pred")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("system scale L = √A  (km)")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) raw size distribution\n(apples vs oranges: "
                    "accumulation offset)", fontsize=10, loc="left")
    ax[0].legend(fontsize=7); ax[0].grid(True, which="both", alpha=0.25)

    # ---- (b) size rescaled by each truth's median L (the apples attempt) ----
    ls_med = np.median(ls["Lt"])
    gc_med = np.median(gc["Lt"])
    ber = np.geomspace(0.1, 10, 35)
    step_hist(ax[1], ls["Lt"] / ls_med, ber, COL["LatentSky"], "-",
              f"LatentSky truth (med={ls_med:.0f} km)")
    step_hist(ax[1], ls["Lp"] / ls_med, ber, COL["LatentSky"], "--",
              "LatentSky pred")
    step_hist(ax[1], gc["Lt"] / gc_med, ber, COL["GraphCast"], "-",
              f"GraphCast truth (med={gc_med:.0f} km)")
    step_hist(ax[1], gc["Lp"] / gc_med, ber, COL["GraphCast"], "--",
              "GraphCast pred")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].axvline(1.0, color="0.6", lw=0.8, ls=":")
    ax[1].set_xlabel("L / median(L_truth)   (scale-free)")
    ax[1].set_ylabel("density")
    ax[1].set_title("(b) size rescaled by truth median\n"
                    "(apples-to-apples: shape & distortion)", fontsize=10,
                    loc="left")
    ax[1].legend(fontsize=7); ax[1].grid(True, which="both", alpha=0.25)

    # ---- (c) intensity distribution ----
    bi = np.geomspace(0.05, 50, 40)
    step_hist(ax[2], ls["rt"], bi, COL["LatentSky"], "-",  "LatentSky truth")
    step_hist(ax[2], ls["rp"], bi, COL["LatentSky"], "--", "LatentSky pred")
    step_hist(ax[2], gc["rt"], bi, COL["GraphCast"], "-",  "GraphCast truth")
    step_hist(ax[2], gc["rp"], bi, COL["GraphCast"], "--", "GraphCast pred")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("system mean intensity ρ  (mm)")
    ax[2].set_ylabel("density")
    ax[2].set_title("(c) intensity distribution\n(ρ; note different units: "
                    "mm/15min vs mm/6h)", fontsize=10, loc="left")
    ax[2].legend(fontsize=7); ax[2].grid(True, which="both", alpha=0.25)

    fig.suptitle("Distribution comparison — GraphCast (ERA5) vs LatentSky (SAM), "
                 "lead = 6 h.  solid = truth, dashed = prediction",
                 y=1.06, fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"LatentSky median L_truth = {ls_med:.1f} km, GraphCast = {gc_med:.1f} km")


if __name__ == "__main__":
    main()
