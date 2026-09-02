"""POD/FAR vs scale L, at a single lead, one overlaid series per --runs entry.

Genericized (2026-07) from plot_pod_far_at_6h_compare.py (hardcoded to 3
specific mseuv_*_lead6h runs, each with its own literal CSV path) into a
--runs/--summary/--tag CLI tool: all runs are read from ONE shared summary
CSV (the unified convention scripts/analysis/eval_scale_decomposition.py
already writes, keyed by the `name` column), and the lead shown in the
title is read from the data itself rather than hardcoded -- so this same
script works for any lead and any run family without editing constants.
"""
from __future__ import annotations
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import figure_path

MARKERS = ["o", "s", "^", "D", "v", "P"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run `name` values to overlay")
    p.add_argument("--labels", nargs="+", default=None,
                    help="legend labels, same order as --runs (default: run names)")
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--tag", required=True, help="model-family label, used in output filename")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    labels = args.labels if args.labels is not None else args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must have the same length as --runs")

    df_all = pd.read_csv(REPO / args.summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lead_h = None
    for name, label, color, marker in zip(args.runs, labels, COLORS, MARKERS * 2):
        df = df_all[df_all["name"] == name].sort_values("scale_mid")
        if df.empty:
            opts = sorted(df_all["name"].unique())
            raise SystemExit(f"no rows for run name {name!r} in {args.summary}; available: {opts}")
        df = df[df["n_truth"] > 0]  # drop empty bins (NaN POD)
        this_lead = df["lead_h"].iloc[0]
        if lead_h is None:
            lead_h = this_lead
        elif this_lead != lead_h:
            print(f"WARNING: {name!r} is lead={this_lead}h, differs from first run's lead={lead_h}h "
                  "-- this figure is meant for a single lead", flush=True)
        axes[0].plot(df["scale_mid"], df["POD"], marker + "-", color=color, ms=7, lw=1.6, label=label)
        axes[1].plot(df["scale_mid"], df["FAR"], marker + "-", color=color, ms=7, lw=1.6, label=label)

    for ax, kind in zip(axes, ["POD", "FAR"]):
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("system scale  L = sqrt(area)  [km]", fontsize=10.5)
        ax.set_title(f"{kind} vs scale, lead={lead_h:g}h", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("POD", fontsize=10.5)
    axes[1].set_ylabel("FAR", fontsize=10.5)
    axes[0].legend(fontsize=8.5, frameon=False)

    out = figure_path("pod_far_vs_scale", args.tag, lead_h) if args.out is None else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
