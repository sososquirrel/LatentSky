"""Figure: POD(t) vs lead for every (resolved) scale bin -- GraphCast and
FuXi companions to fig15_pod_vs_lead_all_scales.py (which covers LatentSky
9var only). Single-panel raw POD vs lead, viridis-by-scale with a shared
colorbar below. All 4 archived leads (6/12/18/24h) shown as measured,
including GraphCast's 6h spin-up dip -- it's real data, not hidden.

No normalization panel and no fitted decay curve here: every anchor we
tried (POD(6h), POD(12h)) turned out not rigorous -- GraphCast's 6h point is
a spin-up artifact (confirmed threshold-independent, not an accumulation-
window bug), so any ratio or exponential fit built on top of these 3-4
sparse, unevenly-motivated points would look precise without being
trustworthy. This figure is deliberately just the descriptive raw curve;
the quantitative scaling-law fit lives in fig16/fig21 for LatentSky only.

One PNG per model, scale bins restricted to >= ERA5 grid Nyquist (~56km, same
cutoff used in fig21_scaling_law_3series.py) since GraphCast/FuXi are not
meaningfully resolved below that.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
LEADS = [6, 12, 18, 24]
ERA5_NYQUIST_KM = 56.0

MODELS = [
    ("graphcast", "GraphCast"),
    ("fuxi", "FuXi"),
]
# threshold tag -> (mm/6h value for the title, output filename suffix)
THRESHOLDS = [
    ("thr2p4", 2.4, ""),        # original threshold, keeps the original filename
    ("thr5", 5.0, "_thr5"),
    ("thr10", 10.0, "_thr10"),
]


def load_multilead(model, thr_tag, leads=LEADS):
    frames = []
    for lead in leads:
        f = REPO / f"docs/scale_summary_{model}_2020-08_lead{lead}h_{thr_tag}.csv"
        df = pd.read_csv(f)
        df["lead_h"] = lead
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_figure(model_key, model_label, thr_tag, thr_mm, suffix):
    df = load_multilead(model_key, thr_tag)
    df = df[df["scale_mid"] >= ERA5_NYQUIST_KM]
    scales = sorted(df["scale_mid"].unique())

    cmap = plt.cm.viridis
    norm = mcolors.LogNorm(vmin=min(scales), vmax=max(scales))

    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)

    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].to_numpy(dtype=float)
        pod = sub["POD"].to_numpy()
        color = cmap(norm(s))

        ax.plot(t, pod, "o-", color=color, ms=6, lw=1.4,
                markeredgecolor="k", markeredgewidth=0.3, zorder=5)

    ax.set_xlabel("forecast lead  t  [h]", fontsize=12)
    ax.set_ylabel("POD", fontsize=12)
    ax.set_title("Raw POD vs lead", fontsize=12)
    ax.grid(alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.85,
                        pad=0.14, aspect=30)
    cbar.set_label("system scale  L = √A  (km)", fontsize=11)

    fig.suptitle(f"{model_label}: POD vs lead, by scale (≥ ERA5 Nyquist {ERA5_NYQUIST_KM:.0f}km, "
                 f"threshold={thr_mm:g}mm/6h)",
                fontsize=12)

    out = OUT_DIR / f"fig15_pod_vs_lead_{model_key}{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    for model_key, model_label in MODELS:
        for thr_tag, thr_mm, suffix in THRESHOLDS:
            make_figure(model_key, model_label, thr_tag, thr_mm, suffix)


if __name__ == "__main__":
    main()
