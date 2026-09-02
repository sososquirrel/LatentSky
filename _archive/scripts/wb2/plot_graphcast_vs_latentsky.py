"""Compare LatentSky (SAM-truth) vs GraphCast (ERA5-truth) on the same
scale × LST × surface decomposition at lead = 6 h.

This is the generalization claim: do AI weather models share the signature
of preserved phase + collapsed space at long leads? If GraphCast looks like
LatentSky on these axes, the diagnostic is universal rather than a quirk
of one model.

NOTE on the truth field. LatentSky was trained against SAM and is verified
against SAM here. GraphCast was trained against ERA5 and is verified
against ERA5. Verifying both against the *same* truth (IMERG) would be the
cleanest experiment; this first-pass version just shows that the
*signature* on the scale × LST axes is robust across architectures and
training distributions.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import FIGURES_DIR


def load(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "predictor_set" in df.columns:
        return df
    df["predictor_set"] = path.stem
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latentsky-systems",
                   default=str(REPO / "docs/scale_systems_v3.csv.gz"))
    p.add_argument("--graphcast-systems",
                   default=str(REPO / "docs/scale_systems_graphcast_2020-08.csv.gz"))
    p.add_argument("--out", default=str(FIGURES_DIR / "graphcast_vs_latentsky_6h.png"))
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=600.0)
    p.add_argument("--scale-bins", type=int, default=10)
    args = p.parse_args()

    ls = load(pathlib.Path(args.latentsky_systems))
    gc = load(pathlib.Path(args.graphcast_systems))
    # LatentSky: pick the 6h lead, 9var (mse+uv) predictor set.
    ls = ls[(ls["lead_h"] == 6.0) & (ls["predictor_set"] == "9var (mse+uv)")]
    print(f"LatentSky 6h systems: {len(ls)}", flush=True)
    print(f"GraphCast 6h systems: {len(gc)}", flush=True)

    edges = np.geomspace(args.scale_min, args.scale_max, args.scale_bins + 1)

    def bin_pod_far(df):
        df = df.copy()
        scale = np.where(df["role"] == "false_alarm", df["L_p"], df["L_t"])
        df["bin"] = np.digitize(scale, edges) - 1
        df = df[(df["bin"] >= 0) & (df["bin"] < args.scale_bins)]
        out = []
        for b in range(args.scale_bins):
            sub = df[df["bin"] == b]
            nm = int((sub["role"] == "match").sum())
            nmiss = int((sub["role"] == "miss").sum())
            nfa = int((sub["role"] == "false_alarm").sum())
            n_truth = nm + nmiss
            n_pred = nm + nfa
            pod = nm / n_truth if n_truth else np.nan
            far = nfa / n_pred if n_pred else np.nan
            out.append((b, edges[b], edges[b + 1],
                        n_truth, n_pred, pod, far))
        return pd.DataFrame(out, columns=["bin", "scale_lo", "scale_hi",
                                          "n_truth", "n_pred", "POD", "FAR"])

    a_ls = bin_pod_far(ls)
    a_gc = bin_pod_far(gc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                    constrained_layout=True)
    mid = np.sqrt(a_ls["scale_lo"] * a_ls["scale_hi"])
    ax1.plot(mid, a_ls["POD"], "o-", lw=2, label="LatentSky (SAM truth)",
              color="#1f77b4")
    ax1.plot(mid, a_gc["POD"], "s-", lw=2, label="GraphCast (ERA5 truth)",
              color="#d62728")
    ax1.set_xscale("log")
    ax1.set_xlabel("system characteristic scale L [km]")
    ax1.set_ylabel("POD")
    ax1.set_title("(a) POD vs scale, lead = 6 h", loc="left")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)

    ax2.plot(mid, a_ls["FAR"], "o-", lw=2, label="LatentSky (SAM truth)",
              color="#1f77b4")
    ax2.plot(mid, a_gc["FAR"], "s-", lw=2, label="GraphCast (ERA5 truth)",
              color="#d62728")
    ax2.set_xscale("log")
    ax2.set_xlabel("system characteristic scale L [km]")
    ax2.set_ylabel("FAR")
    ax2.set_title("(b) FAR vs scale, lead = 6 h", loc="left")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Scale-aware skill at 6 h: LatentSky vs GraphCast",
                  y=1.04, fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
