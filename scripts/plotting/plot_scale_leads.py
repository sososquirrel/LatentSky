"""Cross-lead overlay — how the scale-resolved skill degrades with horizon.

For one predictor set, overlay every lead time on shared scale axes:

  (a) POD vs scale          detection of real systems, by size
  (b) FAR vs scale          spurious systems, by size
  (c) dlnA_med vs scale     area bias (sign: + = predicted too big)
  (d) dlnrho_med vs scale   intensity bias (sign: - = predicted too weak)

This is the payoff figure of the framework: it shows *at which scale* and
*at which lead* the model breaks, which bulk RMSE/correlation cannot. Lead
time is colour-coded (dark = nowcast, bright = long horizon).

Reads docs/scale_summary.csv (or _v2 after the retuned re-eval).
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--out", default=None,
                   help="Default docs/figures/scale_leads_<predictor>.png")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    df = pd.read_csv(repo / args.summary)

    g = df[df["predictor_set"] == args.predictor_set]
    if g.empty:
        opts = sorted(df["predictor_set"].unique())
        raise SystemExit(f"no rows for predictor set {args.predictor_set!r}; "
                         f"available: {opts}")

    leads = sorted(g["lead_h"].unique())
    norm = Normalize(vmin=min(leads), vmax=max(leads))
    cmap = cm.viridis

    panels = [
        ("POD", "(a) detection: POD vs scale", "POD", (0, 1)),
        ("FAR", "(b) false alarms: FAR vs scale", "FAR", (0, 1)),
        ("dlnA_med", "(c) area bias: median dln A  (+ = too big)",
         "median dln A", None),
        ("dlnrho_med", "(d) intensity bias: median dln ρ  (− = too weak)",
         "median dln ρ", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (col, title, ylab, ylim) in zip(axes.ravel(), panels):
        for lead in leads:
            sub = g[g["lead_h"] == lead].sort_values("scale_mid")
            m = sub[col].notna()
            ax.plot(sub.loc[m, "scale_mid"], sub.loc[m, col],
                    "o-", ms=4, color=cmap(norm(lead)), label=f"{lead:g} h")
        if col.startswith("dln"):
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("system scale L = √A  (km)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("lead time (h)")

    fig.suptitle(
        f"Scale-resolved skill vs forecast horizon — {args.predictor_set}\n"
        f"(P = ρ·A,  dlnP = dlnA + dlnρ)", fontsize=14)

    safe = args.predictor_set.split()[0]
    out = (repo / (args.out or f"docs/figures/scale_leads_{safe}.png")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
