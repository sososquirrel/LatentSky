"""Figure: system-size (L = sqrt(area)) distribution, truth vs prediction,
at lead=2h, for the 9var (mse+uv) model.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
LEAD_H = 2.0
OTHER_LEAD_H = 0.0  # shares axes with fig07_size_distribution_2h.py -- keep in sync


def _logbins(*arrays, n=30):
    vals = np.concatenate([a[a > 0] for a in arrays if a.size])
    return np.geomspace(vals.min(), vals.max(), n)


def main():
    df_all = pd.read_csv(REPO / "docs/scale_systems_v3.csv.gz", compression="gzip",
                         usecols=["predictor_set", "lead_h", "L_t", "L_p"])
    df_all = df_all[df_all["predictor_set"] == "9var (mse+uv)"]

    # Shared bins/ylim computed from BOTH leads (0h and 2h), so this figure
    # and fig06_size_distribution_0h.py are directly, visually comparable.
    df0 = df_all[df_all["lead_h"] == LEAD_H]
    df2 = df_all[df_all["lead_h"] == OTHER_LEAD_H]
    Lt0, Lp0 = df0["L_t"].dropna().to_numpy(), df0["L_p"].dropna().to_numpy()
    Lt2, Lp2 = df2["L_t"].dropna().to_numpy(), df2["L_p"].dropna().to_numpy()
    bins = _logbins(Lt0, Lp0, Lt2, Lp2, n=30)
    ymax = max(np.histogram(a, bins=bins)[0].max() for a in [Lt0, Lp0, Lt2, Lp2] if a.size)

    df = df0
    Lt = df.loc[df["L_t"].notna(), "L_t"].to_numpy()
    Lp = df.loc[df["L_p"].notna(), "L_p"].to_numpy()

    TRUTH_COLOR = "#264653"
    PRED_COLOR = "#e76f51"

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    # Truth: solid fill. Pred: outline only (no fill) so the two never
    # color-blend into a muddy overlap region.
    ax.hist(Lt, bins=bins, histtype="stepfilled", facecolor=TRUTH_COLOR, alpha=0.55,
            edgecolor=TRUTH_COLOR, lw=1.4, label=f"truth (n={len(Lt)})", zorder=3)
    ax.hist(Lp, bins=bins, histtype="step", color=PRED_COLOR, lw=2.2,
            label=f"pred (n={len(Lp)})", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.7, ymax * 1.5)
    ax.set_xlabel("system scale  L = √A  (km)", fontsize=12.5)
    ax.set_ylabel("# systems", fontsize=12.5)
    ax.set_title(f"System-size distribution — 9var (mse+uv), lead={LEAD_H:g}h",
                fontsize=12.5, pad=10)
    ax.legend(fontsize=10.5, frameon=False)
    ax.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.grid(True, which="minor", alpha=0.12, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10.5)

    out = OUT_DIR / "fig07_size_distribution_2h.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
