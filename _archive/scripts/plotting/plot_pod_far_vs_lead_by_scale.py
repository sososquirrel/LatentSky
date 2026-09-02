"""POD(t) and FAR(t) vs lead, one row per model, one column per metric
(POD, FAR), one curve per scale bin within each panel -- the raw
detection-skill and false-alarm-rate curves underlying the tau(L)
scaling-law fits elsewhere in this suite, shown without any exponential-
decay fit forced on them.

Uses the SAME custom scale-bin edges and color scheme as
plot_dln_3panel.py (the precip decomposition figure), computed directly
from the raw systems CSVs (latentsky.scale_analysis.system_table output)
rather than the pre-aggregated rebinned16 summaries, so bins line up
exactly with that figure for direct visual comparison.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR

_LEADS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]
TRACK_PRESETS = {
    "onehead": dict(systems="docs/scale_systems_onehead_patchsplit.csv.gz",
                     runs=[f"patchsplit_onehead_lead{L}" for L in _LEADS],
                     label="regression (new split)"),
    "two-head": dict(systems="docs/scale_systems_modelcompare.csv.gz",
                      runs=[f"patchsplit_twohead_olr_fss_lead{L}" for L in _LEADS],
                      label="two-head"),
    "diffusion": dict(systems="docs/scale_systems_diffusion.csv.gz",
                       runs=[f"diffusion_lead{L}" for L in _LEADS],
                       label="two-head+diffusion"),
}
DEFAULT_EDGES = [60, 62.0, 71.4, 80, 100, 140, 200, 300, 450, 700]
PALETTE7 = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#f0f0f0", "#fdae6b", "#e6550d"]


def _labels(edges):
    e = np.round(edges).astype(int)
    return [f"<{e[1]}km" if i == 0 else f"{e[i]}-{e[i+1]}km" for i in range(len(e) - 1)]


def _pod_far_by_bin(systems_path, runs, edges, floor_km, repo):
    df = pd.read_csv(repo / systems_path)
    df = df[df["name"].isin(runs)].copy()
    df = df[(df["scale"] >= max(floor_km, edges[0])) & (df["scale"] < edges[-1])]
    b = np.clip(np.digitize(df["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    df["bin"] = np.array(_labels(edges))[b]

    rows = []
    for (lead_h, lbl), g in df.groupby(["lead_h", "bin"]):
        n_match = int((g["role"] == "match").sum())
        n_miss = int((g["role"] == "miss").sum())
        n_fa = int((g["role"] == "false_alarm").sum())
        n_truth, n_pred = n_match + n_miss, n_match + n_fa
        rows.append(dict(lead_h=lead_h, bin=lbl,
                          POD=n_match / n_truth if n_truth else np.nan,
                          FAR=n_fa / n_pred if n_pred else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", nargs="+", default=["onehead", "two-head", "diffusion"],
                    choices=list(TRACK_PRESETS))
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--floor-km", type=float, default=60.0)
    p.add_argument("--out", default=str(FIGURES_DIR / "pod_far_vs_lead_by_scale.png"))
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    edges = np.array(args.edges)
    labels = _labels(edges)
    colors = (PALETTE7[:len(labels)] if len(labels) <= len(PALETTE7) else
              [matplotlib.colors.rgb2hex(c) for c in
               plt.get_cmap("coolwarm")(np.linspace(0, 1, len(labels)))])
    tracks = [TRACK_PRESETS[n] for n in args.tracks]

    fig, axes = plt.subplots(len(tracks), 2, figsize=(12, 4.0 * len(tracks)),
                              sharex=True, sharey=True, squeeze=False)
    for row_axes, t in zip(axes, tracks):
        pf = _pod_far_by_bin(t["systems"], t["runs"], edges, args.floor_km, repo)
        for ax, metric in zip(row_axes, ["POD", "FAR"]):
            for lbl, color in zip(labels, colors):
                sub = pf[pf["bin"] == lbl].sort_values("lead_h")
                if sub.empty:
                    continue
                ax.plot(sub["lead_h"], sub[metric], "o-", ms=5, lw=1.8, color=color, label=lbl)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
        row_axes[0].set_ylabel(f"{t['label']}\nPOD / FAR", fontsize=10)

    axes[0, 0].set_title("POD(t)", fontsize=12)
    axes[0, 1].set_title("FAR(t)", fontsize=12)
    for ax in axes[-1]:
        ax.set_xlabel("lead (h)")

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(labels), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    row_desc = "   ".join(f"row {i+1}: {t['label']}" for i, t in enumerate(tracks))
    fig.suptitle(f"POD and FAR vs lead, one curve per scale bin (same bins/colors as the precip decomposition)\n{row_desc}", y=1.02)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
