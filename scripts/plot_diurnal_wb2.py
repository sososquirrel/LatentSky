"""Small-system diurnal POD cycle: LatentSky vs WB2 models.

Land (left) and ocean (right) panels; all models overlaid.

Data sources
------------
LatentSky : docs/scale_systems_v3.csv.gz  — 15-min steps, clean LST axis
GraphCast : docs/scale_systems_graphcast_2020-08.csv.gz  — valid times 06Z/18Z only
FuXi      : docs/scale_systems_fuxi_2020-08_6h.csv.gz   — valid times 06Z/18Z only

For WB2 models, only two UTC times are available (12-hourly inits + 6h lead),
so the LST axis is confounded with longitude: each LST bin draws from a narrow
longitude band rather than the same location across days. LatentSky samples
all 96 15-min frames across the full diurnal cycle, giving a clean LST axis.
GenCast is excluded: minimum 12-hr lead gives only two UTC times per day
(00Z and 12Z), insufficient for a meaningful diurnal cycle plot.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO

SMALL_KM   = 65          # same threshold as existing diurnal_graphcast figure
LST_EDGES  = np.arange(0, 25, 2)          # 2-h bins for smoother curves
LST_MID    = 0.5 * (LST_EDGES[:-1] + LST_EDGES[1:])

MODELS = [
    # (label, csv_path, color, linestyle, lead_h filter or None, predictor_set or None)
    # LatentSky: threshold=0.1 mm/15min = 0.4 mm/hr  (native resolution)
    # WB2 models: threshold=2.4 mm/6h  = 0.4 mm/hr  (physically equivalent)
    ("LatentSky 6h",  "docs/scale_systems_v3.csv.gz",
     "#1f77b4", "-",   6.0,  "9var (mse+uv)"),
    ("GraphCast 6h",  "docs/scale_systems_graphcast_2020-08_thr2p4.csv.gz",
     "#e6194b", "--",  None, None),
    ("FuXi 6h",       "docs/scale_systems_fuxi_2020-08_6h.csv.gz",
     "#3cb44b", "-.",  None, None),
]


def pod_by_lst(df: pd.DataFrame, small_km: float = SMALL_KM) -> dict:
    """Return dict[is_land] -> POD array over LST bins for small systems."""
    df = df.copy()
    scale = np.where(df["role"] == "false_alarm", df["L_p"], df["L_t"])
    df = df[scale < small_km]
    df["b"] = np.digitize(df["lst_hour"] % 24, LST_EDGES) - 1
    df = df[(df["b"] >= 0) & (df["b"] < len(LST_MID))]
    out = {}
    for is_land in (True, False):
        sub = df[df["is_land"] == is_land]
        pod = []
        for b in range(len(LST_MID)):
            g = sub[sub["b"] == b]
            n_match = (g["role"] == "match").sum()
            n_miss  = (g["role"] == "miss").sum()
            n = n_match + n_miss
            pod.append(n_match / n if n >= 5 else np.nan)
        out[is_land] = np.array(pod)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True,
                             sharey=True)
    ax_land, ax_ocn = axes

    for label, rel_path, color, ls, lead_h, pred_set in MODELS:
        path = REPO / rel_path
        if not path.exists():
            print(f"missing (skip): {path}", flush=True)
            continue
        df = pd.read_csv(path)
        if lead_h is not None:
            df = df[df["lead_h"] == lead_h]
        if pred_set is not None:
            df = df[df["predictor_set"] == pred_set]
        pod = pod_by_lst(df)

        n_lst = (~np.isnan(pod[True])).sum()
        lw = 2.2 if "LatentSky" in label else 1.8
        kw = dict(color=color, ls=ls, lw=lw, marker="o", ms=4)
        ax_land.plot(LST_MID, pod[True],  label=label, **kw)
        ax_ocn.plot( LST_MID, pod[False], label=label, **kw)
        print(f"{label}: {n_lst} valid LST bins (land)", flush=True)

    for ax, title in [(ax_land, "(a) Land"), (ax_ocn, "(b) Ocean")]:
        ax.set_xlabel("local solar time [h]", fontsize=11)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlim(0, 24)
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.grid(alpha=0.3)
        ax.axvspan(18, 24, alpha=0.05, color="navy")
        ax.axvspan(0,   6, alpha=0.05, color="navy")

    ax_land.set_ylabel(f"POD  (small systems, L < {SMALL_KM} km)", fontsize=11)
    ax_land.legend(frameon=False, fontsize=10)

    fig.suptitle(
        "Diurnal cycle of small-system detection — WB2 models vs LatentSky\n"
        "August 2020, tropics ±40°.  "
        "WB2 LST axis is longitude-confounded (12-hourly inits, 6h lead).",
        fontsize=9,
    )

    out = pathlib.Path(args.out or REPO / "docs" / "figures" / "diurnal_wb2_vs_latentsky.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
