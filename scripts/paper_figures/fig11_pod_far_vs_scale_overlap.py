"""Figure: POD and FAR vs system scale, lead=0h and lead=2h overlaid,
9var (mse+uv) model."""
from __future__ import annotations
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"


def main():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[df["predictor_set"] == "9var (mse+uv)"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    # Color encodes metric (POD=blue, FAR=orange); linestyle encodes lead
    # (0h=solid, 2h=dashed).
    POD_COLOR = "#1f77b4"
    FAR_COLOR = "#e76f51"
    lead_styles = {0.0: "-", 2.0: "--"}
    for lead, ls in lead_styles.items():
        sub = df[df["lead_h"] == lead].sort_values("scale_mid")
        ax.plot(sub["scale_mid"], sub["POD"], "o", lw=2, ms=8, color=POD_COLOR, ls=ls,
                label=f"POD, lead={lead:g}h")
        ax.plot(sub["scale_mid"], sub["FAR"], "s", lw=2, ms=8, color=FAR_COLOR, ls=ls,
                label=f"FAR, lead={lead:g}h")

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("system scale  L = √A  (km)", fontsize=12)
    ax.set_ylabel("object skill", fontsize=12)
    ax.set_title("POD / FAR vs scale — 9var (mse+uv), lead=0h vs 2h", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = OUT_DIR / "fig11_pod_far_vs_scale_overlap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
