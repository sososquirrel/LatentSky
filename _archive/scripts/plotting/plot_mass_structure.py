"""45-degree reparametrization of the area/intensity decomposition:

  mass      = dlnA + dlnrho  (== dlnP, what any aggregate/MSE-style metric
              sees -- area and intensity errors of opposite sign cancel here)
  structure = dlnA - dlnrho  (how the model distributes a given total between
              area and intensity -- completely invisible to mass/dlnP, and
              therefore to any metric computed only on the aggregate field)

Two figures:
  1. mass vs structure, vs lead, by scale bin -- same row-per-model layout
     as plot_dln_3panel.py, so a bin/model with mass~0 but structure large
     is visually obvious (aggregate-perfect, physically wrong).
  2. per-model scatter (hexbin density) of dlnA vs dlnrho, all matched
     systems pooled, with the anti-diagonal dlnrho = -dlnA (mass=0) drawn --
     points ON that line are exactly the "looks-perfect-in-aggregate"
     failure mode; the further a point sits from the origin along that same
     line, the more physically wrong it is despite mass=0.
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


def _load(spec, edges, floor_km, repo):
    df = pd.read_csv(repo / spec["systems"])
    df = df[df["name"].isin(spec["runs"]) & (df["role"] == "match")].copy()
    df = df[(df["scale"] >= max(floor_km, edges[0])) & (df["scale"] < edges[-1])]
    df["mass"] = df["dlnA"] + df["dlnrho"]
    df["structure"] = df["dlnA"] - df["dlnrho"]
    b = np.clip(np.digitize(df["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    df["bin"] = np.array(_labels(edges))[b]
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", nargs="+", default=["onehead", "two-head", "diffusion"],
                    choices=list(TRACK_PRESETS))
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--floor-km", type=float, default=60.0)
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    edges = np.array(args.edges)
    labels = _labels(edges)
    colors = (PALETTE7[:len(labels)] if len(labels) <= len(PALETTE7) else
              [matplotlib.colors.rgb2hex(c) for c in
               plt.get_cmap("coolwarm")(np.linspace(0, 1, len(labels)))])
    tracks = [TRACK_PRESETS[n] for n in args.tracks]
    dfs = [_load(t, edges, args.floor_km, repo) for t in tracks]

    # ---- Figure 1: mass, structure vs lead, one row per model ----
    nrows = len(tracks)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 4.4 * nrows + 1), sharex=True, sharey=True,
                              squeeze=False)
    for row_axes, df, t in zip(axes, dfs, tracks):
        for ax, col in zip(row_axes, ["mass", "structure"]):
            for lbl, color in zip(labels, colors):
                sub = df[df["bin"] == lbl]
                g = sub.groupby("lead_h")[col].agg(med="median", n="count").reset_index().sort_values("lead_h")
                n_total = int(g["n"].sum())
                ax.plot(g["lead_h"], g["med"], "o-", color=color, lw=2, ms=6,
                       markeredgecolor="k", markeredgewidth=0.5,
                       label=f"{lbl}  (n={n_total:,})", zorder=3)
            ax.axhline(0.0, color="0.4", lw=1.0, ls=":", zorder=1)
            ax.grid(alpha=0.25)
        row_axes[0].set_ylabel(f"{t['label']}\nln(X_pred/X_truth) (median)", fontsize=10)

    axes[0, 0].set_title("mass = dlnA + dlnrho  (= dlnP -- what MSE-style metrics see)", fontsize=11)
    axes[0, 1].set_title("structure = dlnA - dlnrho  (invisible to mass; how error is\ndistributed between area and intensity)", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("lead (h)")

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(labels), fontsize=8,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    row_desc = "   ".join(f"row {i+1}: {t['label']}" for i, t in enumerate(tracks))
    fig.suptitle("Mass/structure reparametrization -- a bin with mass~0 but structure>>0 is\n"
                 f"aggregate-perfect, physically wrong.   {row_desc}", y=1.01)

    out1 = FIGURES_DIR / f"mass_structure_{'_vs_'.join(args.tracks)}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"wrote {out1}")

    # ---- Figure 2: dlnA vs dlnrho scatter (hexbin density), anti-diagonal drawn ----
    fig2, axes2 = plt.subplots(1, len(tracks), figsize=(6.0 * len(tracks), 5.6), squeeze=False)
    axes2 = axes2[0]
    lim = 3.0
    for ax, df, t in zip(axes2, dfs, tracks):
        hb = ax.hexbin(df["dlnA"], df["dlnrho"], gridsize=60, extent=[-lim, lim, -lim, lim],
                       bins="log", cmap="viridis", mincnt=1)
        xs = np.linspace(-lim, lim, 10)
        ax.plot(xs, -xs, "r--", lw=1.8, label="mass=0 (aggregate-perfect,\nphysically wrong)")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="0.6", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("dlnA (area)")
        ax.set_title(t["label"], fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        fig2.colorbar(hb, ax=ax, label="log10(count)", shrink=0.8)
    axes2[0].set_ylabel("dlnrho (intensity)")
    fig2.suptitle("dlnA vs dlnrho, all matched systems/leads/scale-bins pooled -- density along the\n"
                 "anti-diagonal is mass-cancellation: total precip looks fine, area/intensity don't", y=1.03)

    out2 = FIGURES_DIR / f"mass_structure_scatter_{'_vs_'.join(args.tracks)}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
