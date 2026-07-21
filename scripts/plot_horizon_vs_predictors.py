"""Headless plot for notebooks/01_horizon_vs_predictors.ipynb.

Reads docs/horizon_vs_predictors.csv (produced by eval_horizon_vs_predictors.py)
and saves a 2x3 panel of skill metrics vs lead time, one curve per predictor set.
No GPU needed — fine to run on a login node.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="docs/horizon_vs_predictors.csv")
    p.add_argument("--out", default="docs/figures/horizon_vs_predictors.png")
    p.add_argument(
        "--r2-all-out",
        default="docs/figures/horizon_r2_all.png",
        help="Path for the standalone R2_all-only figure.",
    )
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo / args.csv)
    df = df.sort_values(["predictor_set", "lead_h"]).reset_index(drop=True)
    print(f"{len(df)} rows, {df['predictor_set'].nunique()} predictor sets", flush=True)

    metric_cols = ["R2_all", "R2_rain", "MSE", "MAE", "Correlation", "JS_div"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, m in zip(axes.flat, metric_cols):
        for label, sub in df.groupby("predictor_set"):
            ax.plot(sub["lead_h"], sub[m], marker="o", label=label)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(m)
        ax.set_title(m)
        if m == "R2_all":
            ax.set_ylim(0, 1)
        elif m == "R2_rain":
            ymin = min(df[m].min(), 0.0)
            ax.set_ylim(ymin - 0.05, 1)
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Skill vs forecast horizon, by predictor set", fontsize=13)
    fig.tight_layout()

    out = (repo / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)

    fig2, ax = plt.subplots(figsize=(7, 5))
    for label, sub in df.groupby("predictor_set"):
        ax.plot(sub["lead_h"], sub["R2_all"], marker="o", label=label)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("R2_all")
    ax.set_title("R² (all pixels) vs forecast horizon, by predictor set")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig2.tight_layout()

    out2 = (repo / args.r2_all_out).resolve()
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}", flush=True)


if __name__ == "__main__":
    main()
