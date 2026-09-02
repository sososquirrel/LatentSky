"""Figure: R^2 (bulk, all-pixel, no object detection) vs forecast lead,
for the 9var (mse+uv) model only.

Uses R2_all from latentsky.metrics.r2_all_pixels_safe (plain pixel-wise
R^2 over the whole field), precomputed in docs/horizon_vs_predictors.csv
by scripts/eval_horizon_vs_predictors.py -> latentsky.inference.evaluate_run.
This is the model's raw precipitation-prediction performance -- no
system detection or IoU matching involved.
"""
from __future__ import annotations
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"


def main():
    df = pd.read_csv(REPO / "docs/horizon_vs_predictors.csv")
    df = df[df["predictor_set"] == "9var (mse+uv)"].sort_values("lead_h")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(df["lead_h"], df["R2_all"], "o-", color="#2ca02c", ms=9, lw=2.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("lead time (h)", fontsize=13)
    ax.set_ylabel("R2_all", fontsize=13)
    ax.set_title("R² (all pixels) vs forecast horizon — 9var (mse+uv)", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.4)

    out = OUT_DIR / "fig04_r2_vs_lead.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
