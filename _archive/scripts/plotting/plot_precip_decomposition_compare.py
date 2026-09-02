"""Precipitation log-decomposition (dlnA area error, dlnrho intensity
error, matched systems only) vs scale, overlaid across ALL FIVE model
tracks + THREE baselines at a fixed lead -- the cross-model companion to
scripts/plotting/plot_scale_decomposition.py (which gives a full 6-panel
single-model view instead). dlnP = dlnA + dlnrho exactly (log-additive,
no cross term); this figure isolates whether a track's precip bias comes
from getting system SIZE wrong (dlnA) or system INTENSITY wrong (dlnrho).
Tracks with no rows for a given lead are skipped with a printed warning.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR
from latentsky.predict_dispatch import summary_tracks_for_lead

_cache: dict[str, pd.DataFrame] = {}


def _load(path: str) -> pd.DataFrame | None:
    if path not in _cache:
        try:
            _cache[path] = pd.read_csv(path)
        except FileNotFoundError:
            _cache[path] = None
    return _cache[path]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=float, required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tracks = summary_tracks_for_lead(args.lead)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax_a, ax_rho = axes
    any_plotted = False

    for t in tracks:
        df = _load(t["summary"])
        if df is None:
            print(f"skip {t['label']!r}: {t['summary']} not found")
            continue
        g = df[df["name"] == t["name"]].sort_values("scale_mid")
        if g.empty:
            print(f"skip {t['label']!r}: no rows for name={t['name']!r}")
            continue
        x = g["scale_mid"].to_numpy()
        for ax, col, iqr_col in [(ax_a, "dlnA_med", "dlnA_iqr"), (ax_rho, "dlnrho_med", "dlnrho_iqr")]:
            y = g[col].to_numpy()
            iqr = g[iqr_col].to_numpy()
            ax.plot(x, y, "o-", color=t["color"], label=t["label"])
            ax.fill_between(x, y - iqr / 2, y + iqr / 2, color=t["color"], alpha=0.12)
        any_plotted = True

    if not any_plotted:
        raise SystemExit(f"no track data found for lead={args.lead:g}h")

    for ax, title in zip([ax_a, ax_rho],
                         ["(a) dln A -- system-size (area) error",
                          "(b) dln rho -- system-intensity error"]):
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("system scale L (km)")
        ax.set_ylabel("median log error (pred/truth), matched systems")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.3, which="both")
    ax_rho.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"Precipitation decomposition (dlnP = dlnA + dlnrho) -- "
                 f"all tracks, lead={args.lead:g}h")

    out = pathlib.Path(args.out) if args.out else (
        FIGURES_DIR / f"2026-07-22_full_comparison/precip_decomposition_compare_lead{args.lead:g}h.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
