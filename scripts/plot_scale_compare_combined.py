"""Two-panel POD vs scale: LatentSky (left) vs WB2 models (right).

Left  — LatentSky evaluated against DYAMOND km-scale truth, multiple leads.
Right — GraphCast + FuXi evaluated against ERA5 truth, 6-h lead, 2.4 mm/6h.

Both panels share the same x and y axes. ERA5 Nyquist (~56 km) is marked.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO = pathlib.Path(__file__).resolve().parents[1]

LS_SUMMARY = REPO / "docs/scale_summary_v3.csv"
GC_SUMMARY = REPO / "docs/scale_summary_graphcast_2020-08_6h_thr2p4.csv"
FX_SUMMARY = REPO / "docs/scale_summary_fuxi_2020-08_6h_thr2p4.csv"

LEADS    = [0.0, 1.0, 2.0, 3.0, 6.0]
PRED_SET = "9var (mse+uv)"
NYQUIST  = 56   # km — ERA5 Nyquist limit

# Sequential blue ramp for LatentSky leads (light=short, dark=long)
_blues = cm.get_cmap("Blues")
LS_COLORS = [_blues(0.35 + 0.55 * i / max(len(LEADS) - 1, 1))
             for i in range(len(LEADS))]

WB2_MODELS = [
    ("GraphCast", GC_SUMMARY, "#C0392B", "-",  "o"),
    ("FuXi",      FX_SUMMARY, "#1E8449", "--", "s"),
]

X_LIM = (30, 700)


def _nyquist_line(ax: plt.Axes) -> None:
    ax.axvline(NYQUIST, color="0.45", lw=1.0, ls=":", zorder=1)
    ax.text(NYQUIST * 1.07, 0.97, "ERA5\nNyquist\n56 km",
            fontsize=7, color="0.45", va="top", transform=ax.get_xaxis_transform())


def _style_ax(ax: plt.Axes, title: str, truth_label: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_xlabel("system scale  L = √area  [km]", fontsize=10)
    ax.set_ylabel("POD", fontsize=10)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.text(0.99, 0.03, f"truth: {truth_label}",
            transform=ax.transAxes, fontsize=8, color="0.45",
            ha="right", va="bottom")
    ax.grid(True, which="major", alpha=0.25, lw=0.7)
    ax.grid(True, which="minor", alpha=0.10, lw=0.4)
    ax.tick_params(labelsize=9)
    _nyquist_line(ax)


def main() -> None:
    ls_df = pd.read_csv(LS_SUMMARY)
    ls_df = ls_df[ls_df["predictor_set"] == PRED_SET]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8),
        constrained_layout=True,
        sharex=True,
    )
    (ax_ls_pod, ax_wb2_pod), (ax_ls_far, ax_wb2_far) = axes

    # --- LatentSky panels ---
    for lead, color in zip(LEADS, LS_COLORS):
        sub = ls_df[ls_df["lead_h"] == lead].sort_values("scale_mid")
        if sub.empty:
            continue
        label = f"{int(lead) if lead == int(lead) else lead} h"
        kw = dict(color=color, lw=1.8, marker="o", ms=5, zorder=3)
        ax_ls_pod.plot(sub["scale_mid"], sub["POD"], label=label, **kw)
        ax_ls_far.plot(sub["scale_mid"], sub["FAR"], **kw)

    _style_ax(ax_ls_pod, "(a) LatentSky — DYAMOND truth", "DYAMOND (km-scale)")
    _style_ax(ax_ls_far, "(c) LatentSky — DYAMOND truth", "DYAMOND (km-scale)")
    ax_ls_pod.set_ylabel("POD", fontsize=10)
    ax_ls_far.set_ylabel("FAR", fontsize=10)
    ax_ls_pod.legend(title="lead time", fontsize=9, title_fontsize=9,
                     frameon=False, loc="upper left")

    # --- WB2 panels ---
    for label, path, color, ls, marker in WB2_MODELS:
        if not path.exists():
            print(f"missing: {path}")
            continue
        sub = pd.read_csv(path).sort_values("scale_mid")
        kw = dict(color=color, lw=1.8, ls=ls, marker=marker, ms=5, zorder=3)
        ax_wb2_pod.plot(sub["scale_mid"], sub["POD"], label=f"{label} (6 h)", **kw)
        ax_wb2_far.plot(sub["scale_mid"], sub["FAR"], **kw)

    _style_ax(ax_wb2_pod, "(b) WB2 models — ERA5 truth", "ERA5 (31 km)")
    _style_ax(ax_wb2_far, "(d) WB2 models — ERA5 truth", "ERA5 (31 km)")
    ax_wb2_pod.set_ylabel("")
    ax_wb2_far.set_ylabel("")
    ax_wb2_pod.legend(fontsize=9, frameon=False, loc="upper left")

    for ax in axes.flat:
        ax.set_ylim(0, 1)

    fig.suptitle(
        "Scale-dependent detection: LatentSky vs WB2 SOTA models\n"
        "August 2020, tropics ±40°.  "
        "Threshold: 0.1 mm/15 min (LatentSky) = 2.4 mm/6 h (WB2) = 0.4 mm hr⁻¹.",
        fontsize=9,
    )

    out = REPO / "docs" / "figures" / "scale_compare_combined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
