"""POD and FAR vs scale at a fixed lead, overlaid across ALL FIVE model
tracks + THREE baselines -- the per-lead "detailed comparison" companion
to scripts/plotting/plot_pod_tau_track_compare.py (which pools ALL leads
together into a tau(L) scaling law instead of showing one lead's
snapshot). Tracks with no rows for a given lead are skipped with a
printed warning.
"""
from __future__ import annotations

import argparse
import pathlib

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
    fig, (ax_pod, ax_far) = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
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
        ax_pod.plot(x, g["POD"], "o-", color=t["color"], label=t["label"])
        ax_far.plot(x, g["FAR"], "o-", color=t["color"], label=t["label"])
        any_plotted = True

    if not any_plotted:
        raise SystemExit(f"no track data found for lead={args.lead:g}h")

    for ax, ylab, title in [(ax_pod, "POD", "(a) probability of detection"),
                            (ax_far, "FAR", "(b) false alarm ratio")]:
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("system scale L (km)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.3, which="both")
    ax_far.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"POD / FAR vs scale -- all tracks, lead={args.lead:g}h")

    out = pathlib.Path(args.out) if args.out else (
        FIGURES_DIR / f"2026-07-22_full_comparison/pod_far_compare_lead{args.lead:g}h.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
