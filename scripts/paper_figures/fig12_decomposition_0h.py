"""Figure: dlnP = dlnA + dlnrho decomposition vs scale, lead=0h,
9var (mse+uv) model. Median +/- IQR/2 band for each term."""
from __future__ import annotations
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
LEAD_H = 0.0


def main():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[(df["predictor_set"] == "9var (mse+uv)") & (df["lead_h"] == LEAD_H)]
    df = df.sort_values("scale_mid")
    x = df["scale_mid"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

    ax.plot(x, df["dlnA_med"], "o-", color="C0", label="dln A (area)")
    ax.fill_between(x, df["dlnA_med"] - df["dlnA_iqr"] / 2,
                    df["dlnA_med"] + df["dlnA_iqr"] / 2, color="C0", alpha=0.15)
    ax.plot(x, df["dlnrho_med"], "s-", color="C3", label="dln ρ (intensity)")
    ax.fill_between(x, df["dlnrho_med"] - df["dlnrho_iqr"] / 2,
                    df["dlnrho_med"] + df["dlnrho_iqr"] / 2, color="C3", alpha=0.15)
    ax.plot(x, df["dlnP_med"], "^:", color="0.3", label="dln P (total, = dlnA+dlnρ)")

    ax.set_xscale("log")
    ax.set_xlabel("system scale  L = √A  (km)", fontsize=12)
    ax.set_ylabel("median log error (pred/truth)", fontsize=12)
    ax.set_title(f"Area-vs-intensity error decomposition — 9var (mse+uv), lead={LEAD_H:g}h",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)

    out = OUT_DIR / "fig12_decomposition_0h.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
