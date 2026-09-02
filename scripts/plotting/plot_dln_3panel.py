"""3-panel (precip, area, intensity) x N-row (one row per model track) figure:
dlnP (total precip = A*rho), dlnA (area), dlnrho (intensity), same scale
bins and same y-axis in every panel/row, so tracks' decomposition behavior
is directly comparable.

dlnP = dlnA + dlnrho exactly (P = rho*A), so that panel is not independent
information -- it's shown because "does the total precip bias look like
the area panel or the intensity panel" is itself informative.

Bin edges (km) are derived once from the FIRST track's scale distribution,
then reused unchanged for every other track -- so all rows compare the
same physical scale ranges, even though matched-system counts per bin
differ across tracks.

Pick tracks with --tracks (names resolved via TRACK_PRESETS below), e.g.
  --tracks diffusion two-head onehead
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
    "diffusion": dict(
        systems="docs/scale_systems_diffusion.csv.gz",
        runs=[f"diffusion_lead{L}" for L in _LEADS],
        label="diffusion"),
    "diffusion-manytoone": dict(
        systems="docs/scale_systems_diffusion_manytoone.csv.gz",
        runs=[f"diffusion_lead{L}" for L in _LEADS],
        label="diffusion (many-to-one matching)"),
    "diffusion-oneheadbackbone": dict(
        systems="docs/scale_systems_diffusion_oneheadbackbone.csv.gz",
        runs=[f"diffusion_lead{L}" for L in _LEADS],
        label="diffusion (plain-regression warm-start)"),
    "two-head": dict(
        systems="docs/scale_systems_modelcompare.csv.gz",
        runs=[f"patchsplit_twohead_olr_fss_lead{L}" for L in _LEADS],
        label="two-head"),
    "onehead": dict(
        systems="docs/scale_systems_onehead_patchsplit.csv.gz",
        runs=[f"patchsplit_onehead_lead{L}" for L in _LEADS],
        label="regression (new split)"),
}

DEFAULT_EDGES = [56, 100, 200, 400, 800, 1600]
PALETTE = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#f0f0f0", "#fdae6b", "#e6550d"]
QUANTITIES = [("dlnP", "P"), ("dlnA", "A"), ("dlnrho", "rho")]
TITLES = {"dlnA": "area", "dlnP": "total precip (P = rho·A)", "dlnrho": "intensity"}


def _labels(edges):
    e = np.round(edges).astype(int)
    return [f"<{e[1]}km" if i == 0 else f"{e[i]}-{e[i+1]}km"
            for i in range(len(e) - 1)]


def _load(systems_path, runs, edges, floor_km, repo):
    sysdf = pd.read_csv(repo / systems_path)
    sysdf = sysdf[sysdf["name"].isin(runs) & (sysdf["role"] == "match")].copy()
    sysdf = sysdf[(sysdf["scale"] >= max(floor_km, edges[0])) & (sysdf["scale"] < edges[-1])]
    b = np.clip(np.digitize(sysdf["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    sysdf["bin"] = np.array(_labels(edges))[b]
    return sysdf


def _plot_row(axes, sysdf, labels, colors, signed):
    for ax, (col, var) in zip(axes, QUANTITIES):
        for lbl, color in zip(labels, colors):
            sub = sysdf[sysdf["bin"] == lbl]
            g = sub.groupby("lead_h")[col].agg(med="median", n="count").reset_index()
            g = g.sort_values("lead_h")
            n_total = int(g["n"].sum())
            y = g["med"] if signed else g["med"].abs()
            ax.plot(g["lead_h"], y, "o-", color=color, lw=2, ms=6,
                   markeredgecolor="k", markeredgewidth=0.5,
                   label=f"{lbl}  (n={n_total:,})", zorder=3)
        ax.axhline(0.0, color="0.4" if signed else "0.7",
                  lw=1.0 if signed else 0.8, ls=":", zorder=1)
        ax.grid(alpha=0.25)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", nargs="+", default=["diffusion", "two-head"],
                    choices=list(TRACK_PRESETS), help="rows to include, top to bottom")
    p.add_argument("--signed", action="store_true",
                    help="show the signed median (over/under-estimate) instead of |median|")
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES,
                    help="bin edges in km (ignored if --equal-count is set)")
    p.add_argument("--equal-count", type=int, default=None,
                    help="split into this many bins with equal NUMBER of matched systems "
                         "(quantiles of the FIRST track's scale distribution)")
    p.add_argument("--floor-km", type=float, default=60.0,
                    help="drop systems below this scale before binning (quantization zone)")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    tracks = [TRACK_PRESETS[name] for name in args.tracks]

    first = tracks[0]
    raw0 = pd.read_csv(repo / first["systems"])
    raw0 = raw0[raw0["name"].isin(first["runs"]) & (raw0["role"] == "match")]
    raw0 = raw0[raw0["scale"] >= args.floor_km]

    if args.equal_count is not None:
        _, qedges = pd.qcut(raw0["scale"], q=args.equal_count, retbins=True, duplicates="drop")
        edges = qedges
        edges[0], edges[-1] = args.floor_km, raw0["scale"].max() + 1
    else:
        edges = np.array(args.edges)
    labels = _labels(edges)
    if len(labels) <= len(PALETTE):
        colors = PALETTE[:len(labels)]
    else:
        colors = [matplotlib.colors.rgb2hex(c) for c in
                  plt.get_cmap("coolwarm")(np.linspace(0, 1, len(labels)))]

    sysdfs = [_load(t["systems"], t["runs"], edges, args.floor_km, repo) for t in tracks]

    nrows = len(tracks)
    fig, axes = plt.subplots(nrows, 3, figsize=(17, 4.4 * nrows + 1), sharex=True, sharey=True,
                             squeeze=False)
    for row_axes, sysdf in zip(axes, sysdfs):
        _plot_row(row_axes, sysdf, labels, colors, args.signed)

    for ax, (col, var) in zip(axes[0], QUANTITIES):
        ax.set_title(TITLES[col], fontsize=12)
    for ax in axes[-1]:
        ax.set_xlabel("lead (h)")
    base = "ln(X_pred / X_truth)"
    qlabel = base if args.signed else f"|{base}|"
    for row_axes, t in zip(axes, tracks):
        row_axes[0].set_ylabel(f"{t['label']}\n{qlabel}  (median)", fontsize=10)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(labels), fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    sign_note = "0 = perfect; negative = underestimate, positive = overestimate" if args.signed else ""
    row_desc = "   ".join(f"row {i+1}: {t['label']}" for i, t in enumerate(tracks))
    fig.suptitle(f"Total precip / area / intensity error vs lead, by scale bin\n"
                 f"{row_desc}   |   {sign_note}", y=1.005)

    tag = "_vs_".join(args.tracks)
    out = FIGURES_DIR / f"dln_3panel_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
