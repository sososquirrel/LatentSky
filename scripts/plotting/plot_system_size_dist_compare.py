"""System-size distribution, compared across every model track and
baseline at a fixed lead -- shown as a CCDF (P(L > x), log-log), not a
KDE-smoothed PDF.

Why CCDF and not a KDE PDF (the previous version of this script): the
predicted-system tail is exactly what's interesting here (regression-
family tracks merge adjacent small storms into large blobs via
MSE-blur, inflating the large-L tail by 10-18x relative to truth -- see
FINDINGS.md), and a KDE's bandwidth smoothing both distorts and
visually understates tail behavior. A CCDF has no bandwidth parameter,
is monotonic by construction, and reads directly as "fraction of
systems exceeding size x" -- the exact quantity the tail-inflation claim
is about.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR

L_MIN, L_MAX = 25, 3000

TRACKS = [
    dict(label="regression (old split)", systems="docs/scale_systems_v3.csv.gz",
         name_prefix="mseuv_lead", color="tab:blue"),
    dict(label="regression (new split)", systems="docs/scale_systems_onehead_patchsplit.csv.gz",
         name_prefix="patchsplit_onehead_lead", color="tab:purple"),
    dict(label="two-head", systems="docs/scale_systems_modelcompare.csv.gz",
         name_prefix="patchsplit_twohead_olr_fss_lead", color="tab:orange"),
    dict(label="two-head+diffusion", systems="docs/scale_systems_diffusion.csv.gz",
         name_prefix="diffusion_lead", color="tab:green"),
    dict(label="persistence", systems="docs/scale_systems_baselines_patchsplit.csv.gz",
         name_prefix="persistence_lead", color="gray"),
    dict(label="climatology", systems="docs/scale_systems_baselines_patchsplit.csv.gz",
         name_prefix="climatology_lead", color="black"),
    dict(label="random", systems="docs/scale_systems_baselines_patchsplit.csv.gz",
         name_prefix="random_lead", color="brown"),
]


def ccdf(values):
    """Return (sorted_x, P(L > x)) -- empirical survival function, no
    smoothing/bandwidth of any kind."""
    v = np.asarray(values, dtype=float)
    v = v[(v >= L_MIN) & (v <= L_MAX) & np.isfinite(v)]
    v = np.sort(v)
    n = len(v)
    if n == 0:
        return v, v
    p_exceed = 1.0 - np.arange(1, n + 1) / n
    return v, p_exceed


def frac_above(values, thresh=500.0):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    return float(np.mean(v > thresh))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=float, required=True, help="lead_h to plot, e.g. 3.0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5), constrained_layout=True)
    ax_t, ax_p, ax_hist = axes
    hist_bins = np.geomspace(L_MIN, L_MAX, 18)
    truth_plotted = False
    _cache: dict[str, pd.DataFrame] = {}

    for spec in TRACKS:
        path = spec["systems"]
        if path not in _cache:
            try:
                _cache[path] = pd.read_csv(path)
            except FileNotFoundError:
                print(f"skip {spec['label']!r}: {path} not found")
                _cache[path] = None
        df = _cache[path]
        if df is None:
            continue
        name = f"{spec['name_prefix']}{args.lead:g}h"
        sub = df[df["name"] == name]
        if sub.empty:
            print(f"skip {spec['label']!r}: no rows for name={name!r} in {path}")
            continue

        truth_L = sub[sub["role"].isin(["match", "miss"])]["L_t"].dropna().to_numpy()
        pred_L = sub[sub["role"].isin(["match", "false_alarm"])]["L_p"].dropna().to_numpy()

        if not truth_plotted and len(truth_L) >= 10:
            x, y = ccdf(truth_L)
            ax_t.plot(x, y, color="black", lw=2, label=f"truth (n={len(truth_L)})")
            ax_p.plot(x, y, color="black", lw=2.4, ls="--", alpha=0.7,
                      label=f"truth (n={len(truth_L)})")
            ax_hist.hist(truth_L[(truth_L >= L_MIN) & (truth_L <= L_MAX)], bins=hist_bins,
                         histtype="step", color="black", lw=2.2, label=f"truth (n={len(truth_L)})")
            truth_plotted = True

        x, y = ccdf(pred_L)
        frac500 = frac_above(pred_L, 500.0)
        ax_p.plot(x, y, color=spec["color"], lw=1.8,
                  label=f"{spec['label']} (n={len(pred_L)}, >500km: {frac500:.1%})")
        ax_hist.hist(pred_L[(pred_L >= L_MIN) & (pred_L <= L_MAX)], bins=hist_bins,
                     histtype="step", color=spec["color"], lw=1.8,
                     label=f"{spec['label']} (n={len(pred_L)})")

    if not truth_plotted:
        raise SystemExit(f"no truth data found for lead={args.lead:g}h across any track")

    for ax, title in zip(axes[:2], [f"(a) truth @ lead={args.lead:g}h",
                                     f"(b) predicted, all tracks @ lead={args.lead:g}h "
                                     f"(dashed black = truth, for reference)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("system scale L = sqrt(area)  (km)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    ax_t.set_ylabel("P(system size > L)")
    ax_t.legend(fontsize=8, frameon=False)
    ax_p.legend(fontsize=7, frameon=False, loc="lower left")

    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("system scale L = sqrt(area)  (km)")
    ax_hist.set_ylabel("# systems (raw count, not normalized)")
    ax_hist.set_title(f"(c) raw counts @ lead={args.lead:g}h -- NOT normalized per track\n"
                       f"(unlike (a)/(b): shows absolute count differences the CCDF hides)",
                       loc="left", fontweight="bold", fontsize=9)
    ax_hist.grid(alpha=0.3, which="major")
    ax_hist.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("System-size distribution -- all tracks (CCDF, tail-normalized; raw counts, absolute)")

    out = args.out or str(FIGURES_DIR / f"2026-07-22_full_comparison/system_size_dist_lead{args.lead:g}h.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
