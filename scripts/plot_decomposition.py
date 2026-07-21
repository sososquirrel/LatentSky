"""Closed mean-bias decomposition: eps_freq + eps_int = mean_pred - mean_truth.

Uses rho_truth, rho_pred, mu_truth, mu_pred from the CSV. Plots:
  Left  : eps_freq, eps_int, total bias vs lead time, one set per predictor set
  Right : skill view — CSI (P_I) and R2_rain (tau) for context
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
    p.add_argument("--out", default="docs/figures/horizon_decomposition.png")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo / args.csv)
    df = df.sort_values(["predictor_set", "lead_h"]).reset_index(drop=True)

    needed = ["rho_truth", "rho_pred", "mu_truth", "mu_pred", "CSI", "R2_rain"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns {missing} — rerun the eval script.")

    df["eps_freq"] = (df["rho_pred"] - df["rho_truth"]) * df["mu_truth"]
    df["eps_int"] = df["rho_pred"] * (df["mu_pred"] - df["mu_truth"])
    df["bias_total"] = df["rho_pred"] * df["mu_pred"] - df["rho_truth"] * df["mu_truth"]
    df["bias_check"] = df["eps_freq"] + df["eps_int"]

    print("identity residual max |bias_total - (eps_freq + eps_int)|:",
          (df["bias_total"] - df["bias_check"]).abs().max())

    predictor_sets = sorted(df["predictor_set"].unique())
    colors = {label: c for label, c in zip(predictor_sets, ["C0", "C1", "C2", "C3", "C4"])}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-left: eps_freq vs lead
    ax = axes[0, 0]
    for label in predictor_sets:
        sub = df[df["predictor_set"] == label]
        ax.plot(sub["lead_h"], sub["eps_freq"], marker="o", color=colors[label], label=label)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("ε_freq  (mm)")
    ax.set_title("ε_freq = (ρ_p − ρ_t)·μ_t  — area-error contribution")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # Top-right: eps_int vs lead
    ax = axes[0, 1]
    for label in predictor_sets:
        sub = df[df["predictor_set"] == label]
        ax.plot(sub["lead_h"], sub["eps_int"], marker="o", color=colors[label], label=label)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("ε_int  (mm)")
    ax.set_title("ε_int = ρ_p·(μ_p − μ_t)  — intensity-error contribution")
    ax.grid(True, alpha=0.3)

    # Bottom-left: total bias and the sum of the two parts (should overlay exactly)
    ax = axes[1, 0]
    for label in predictor_sets:
        sub = df[df["predictor_set"] == label]
        ax.plot(sub["lead_h"], sub["bias_total"], marker="o", color=colors[label],
                label=f"{label}")
        ax.plot(sub["lead_h"], sub["bias_check"], marker="x", linestyle=":",
                color=colors[label], alpha=0.7)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("mean-precip bias  (mm)")
    ax.set_title("total = ρ_p·μ_p − ρ_t·μ_t   (× = ε_freq + ε_int; verifies closure)")
    ax.grid(True, alpha=0.3)

    # Bottom-right: skill view (context)
    ax = axes[1, 1]
    for label in predictor_sets:
        sub = df[df["predictor_set"] == label]
        ax.plot(sub["lead_h"], sub["CSI"], marker="o", color=colors[label],
                label=f"{label} CSI (P_I)")
        ax.plot(sub["lead_h"], sub["R2_rain"], marker="s", color=colors[label],
                linestyle="--", label=f"{label} R²_rain (τ)")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("skill")
    ax.set_title("Component skill: CSI (area) vs R²_rain (intensity)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="best")

    fig.suptitle("Precip skill decomposition  —  P̄ = ρ · μ,   ε_total = ε_freq + ε_int", fontsize=13)
    fig.tight_layout()
    out = (repo / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
