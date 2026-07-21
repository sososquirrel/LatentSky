"""Figure: dlnP = dlnA + dlnrho decomposition vs scale, lead=0h and lead=2h
overlaid, 9var (mse+uv) model."""
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

    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

    lead_styles = {0.0: dict(ls="-", alpha=1.0), 2.0: dict(ls="--", alpha=0.7)}
    term_colors = {"dlnA": "C0", "dlnrho": "C3", "dlnP": "0.3"}
    term_markers = {"dlnA": "o", "dlnrho": "s", "dlnP": "^"}
    term_labels = {"dlnA": "dln A (area)", "dlnrho": "dln ρ (intensity)", "dlnP": "dln P (total)"}

    for lead, sty in lead_styles.items():
        sub = df[df["lead_h"] == lead].sort_values("scale_mid")
        x = sub["scale_mid"].to_numpy()
        for term in ["dlnA", "dlnrho", "dlnP"]:
            ax.plot(x, sub[f"{term}_med"], term_markers[term] + sty["ls"],
                    color=term_colors[term], alpha=sty["alpha"], lw=2, ms=7,
                    label=f"{term_labels[term]}, lead={lead:g}h")

    ax.set_xscale("log")
    ax.set_xlabel("system scale  L = √A  (km)", fontsize=12)
    ax.set_ylabel("median log error (pred/truth)", fontsize=12)
    ax.set_title("Area-vs-intensity error decomposition — 9var (mse+uv), lead=0h vs 2h",
                 fontsize=12)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, which="both", alpha=0.25)

    out = OUT_DIR / "fig14_decomposition_overlap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
