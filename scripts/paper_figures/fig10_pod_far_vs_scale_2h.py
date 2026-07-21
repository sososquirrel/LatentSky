"""Figure: POD and FAR vs system scale, lead=0h, 9var (mse+uv) model."""
from __future__ import annotations
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
LEAD_H = 2.0


def main():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[(df["predictor_set"] == "9var (mse+uv)") & (df["lead_h"] == LEAD_H)]
    df = df.sort_values("scale_mid")

    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    ax.plot(df["scale_mid"], df["POD"], "o-", color="C0", ms=8, lw=2, label="POD")
    ax.plot(df["scale_mid"], df["FAR"], "s--", color="C1", ms=8, lw=2, label="FAR")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("system scale  L = √A  (km)", fontsize=12)
    ax.set_ylabel("object skill", fontsize=12)
    ax.set_title(f"POD / FAR vs scale — 9var (mse+uv), lead={LEAD_H:g}h", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    out = OUT_DIR / "fig10_pod_far_vs_scale_2h.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
