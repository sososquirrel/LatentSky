"""Does GraphCast show the small-system diurnal POD cycle that LatentSky does?

Small (L<65 km) system POD vs local solar time, land vs ocean, GraphCast
(ERA5 truth) overlaid on LatentSky at the same 6 h lead (SAM truth).

CAVEAT: GraphCast valid times are only 06 and 18 UTC (12-hourly ICs + 6 h
lead), so LST is collinear with longitude here — each LST bin draws from a
narrow longitude band, and the land curve mixes whichever continents sit at
that local time in the two daily snapshots. LatentSky sampled 24 frames
across the full day, so its LST axis is clean. Read GraphCast's curve as
suggestive, not as a clean diurnal measurement.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO
OUT = REPO / "docs" / "figures" / "diurnal_graphcast_vs_latentsky_6h.png"

SMALL_KM = 65
LST_EDGES = np.arange(0, 25, 3)          # 3-h bins (coarser: less data per bin)
LST_MID = 0.5 * (LST_EDGES[:-1] + LST_EDGES[1:])


def small_pod_by_lst(df):
    """POD(LST) for small systems, split by is_land. Returns dict[is_land] ->
    array over LST bins."""
    df = df.copy()
    scale = np.where(df["role"] == "false_alarm", df["L_p"], df["L_t"])
    df = df[scale < SMALL_KM]
    df["b"] = np.digitize(df["lst_hour"] % 24, LST_EDGES) - 1
    df = df[(df["b"] >= 0) & (df["b"] < len(LST_MID))]
    out = {}
    for is_land in (True, False):
        sub = df[df["is_land"] == is_land]
        pod = []
        for b in range(len(LST_MID)):
            g = sub[sub["b"] == b]
            nm = (g["role"] == "match").sum()
            nmiss = (g["role"] == "miss").sum()
            n = nm + nmiss
            pod.append(nm / n if n else np.nan)
        out[is_land] = np.array(pod)
    return out


def main():
    gc = pd.read_csv(REPO / "docs/scale_systems_graphcast_2020-08.csv.gz")
    ls = pd.read_csv(REPO / "docs/scale_systems_v3.csv.gz")
    ls = ls[(ls["lead_h"] == 6.0) & (ls["predictor_set"] == "9var (mse+uv)")]

    gc_pod = small_pod_by_lst(gc)
    ls_pod = small_pod_by_lst(ls)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True,
                           sharex=True)

    for a, pod, title, color in [
        (ax[0], ls_pod, "(a) LatentSky 6 h (SAM truth)\nclean LST axis (24 frames/day)",
         "#1f77b4"),
        (ax[1], gc_pod, "(b) GraphCast 6 h (ERA5 truth)\nLST=longitude-confounded (06/18 UTC only)",
         "#d62728"),
    ]:
        a.plot(LST_MID, pod[True], "-o", color=color, lw=2, label="land")
        a.plot(LST_MID, pod[False], "--o", color=color, lw=2, label="ocean",
               markerfacecolor="white")
        a.set_title(title, fontsize=10, loc="left")
        a.set_xlabel("local solar time [h]")
        a.set_xticks(range(0, 25, 6))
        a.set_xlim(0, 24)
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    ax[0].set_ylabel(f"POD  (small systems, L < {SMALL_KM} km)")

    # Shared y-limits per panel are fine; POD ranges differ between models.
    fig.suptitle("Small-system POD vs local solar time — does the diurnal "
                 "signature appear in GraphCast?", y=1.05, fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    # also print the numbers
    for name, pod in [("LatentSky", ls_pod), ("GraphCast", gc_pod)]:
        print(f"\n{name} small-system POD by LST (3h bins):")
        print("  LST :", " ".join(f"{m:4.0f}" for m in LST_MID))
        print("  land:", " ".join(f"{v:4.2f}" for v in pod[True]))
        print("  ocn :", " ".join(f"{v:4.2f}" for v in pod[False]))


if __name__ == "__main__":
    main()
