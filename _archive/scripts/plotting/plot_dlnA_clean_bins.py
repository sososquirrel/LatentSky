"""Simple, single plot: |dlnA_med(t)| vs lead time, one curve per clean
round-number scale bin (56-100, 100-200, 200-400, 400-800, 800-1600km).

Deliberately minimal -- no joint fit, no bootstrap, no regime-splitting.
Just the raw per-bin decay curves, so the ~200km break between "real decay"
and "no decay" is visible directly, without any fitting machinery in the way.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR

DEFAULT_EDGES = [56, 100, 200, 400, 800, 1600]
PALETTE = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#f0f0f0", "#fdae6b", "#e6550d"]


def _labels(edges):
    e = np.round(edges).astype(int)
    return [f"<{e[1]}km" if i == 0 else f"{e[i]}-{e[i+1]}km"
            for i in range(len(e) - 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems_diffusion.csv.gz")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--tag", default="diffusion")
    p.add_argument("--quantity", choices=["dlnA", "dlnrho"], default="dlnA")
    p.add_argument("--signed", action="store_true",
                    help="show the signed median (over/under-estimate) instead of |median|")
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES,
                    help="bin edges in km, e.g. --edges 25 60 100 200 (ignored if --equal-count is set)")
    p.add_argument("--equal-count", type=int, default=None,
                    help="split into this many bins with equal NUMBER of matched systems per "
                         "bin (quantiles of scale), instead of fixed km edges.")
    p.add_argument("--floor-km", type=float, default=60.0,
                    help="drop systems below this scale before binning -- avoids the "
                         "quantization-artifact zone (<~56km, 1-3 pixel systems) dominating "
                         "one of the equal-count bins.")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    sysdf = pd.read_csv(repo / args.systems)
    sysdf = sysdf[sysdf["name"].isin(args.runs) & (sysdf["role"] == "match")].copy()
    sysdf = sysdf[sysdf["scale"] >= args.floor_km]

    if args.equal_count is not None:
        _, qedges = pd.qcut(sysdf["scale"], q=args.equal_count, retbins=True, duplicates="drop")
        edges = qedges
        edges[0], edges[-1] = args.floor_km, sysdf["scale"].max() + 1
    else:
        edges = np.array(args.edges)
        sysdf = sysdf[(sysdf["scale"] >= edges[0]) & (sysdf["scale"] < edges[-1])]
    labels = _labels(edges)
    colors = PALETTE[:len(labels)]
    b = np.clip(np.digitize(sysdf["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    sysdf["bin"] = np.array(labels)[b]

    fig, ax = plt.subplots(figsize=(7, 5.3))
    for lbl, color in zip(labels, colors):
        sub = sysdf[sysdf["bin"] == lbl]
        g = sub.groupby("lead_h")[args.quantity].agg(med="median", n="count").reset_index()
        g = g.sort_values("lead_h")
        n_total = int(g["n"].sum())
        y = g["med"] if args.signed else g["med"].abs()
        ax.plot(g["lead_h"], y, "o-", color=color, lw=2, ms=6,
               markeredgecolor="k", markeredgewidth=0.5,
               label=f"{lbl}  (n={n_total:,})", zorder=3)

    ax.axhline(0.0, color="0.4" if args.signed else "0.7", lw=1.0 if args.signed else 0.8,
              ls=":", zorder=1)
    ax.set_xlabel("lead (h)")
    var = "A" if args.quantity == "dlnA" else "rho"
    base = f"ln({var}_pred / {var}_truth)"
    qlabel = base if args.signed else f"|{base}|"
    ax.set_ylabel(f"{qlabel}  (median)")
    sign_note = "\n0 = perfect; negative = underestimate, positive = overestimate" if args.signed else ""
    ax.set_title(f"{qlabel} vs lead, by clean scale bin -- {args.tag}{sign_note}")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)

    out = FIGURES_DIR / f"{args.quantity}_clean_bins_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
