"""Multi-model scale-decomposition comparison across WeatherBench2 baselines.

Reads docs/scale_summary_{model}_{month}_24h.csv for each model and overlays
them on four shared axes:

  (a) POD vs scale          — detection rate by system size
  (b) FAR vs scale          — false-alarm rate by system size
  (c) dlnA (area bias)      — is the model producing too large / small systems?
  (d) dlnρ (intensity bias) — is the model too wet / dry inside systems?

Usage:
  python scripts/plot_wb2_model_compare.py
  python scripts/plot_wb2_model_compare.py --out docs/figures/wb2_compare_24h.png
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]

MODELS_24H = [
    ("graphcast", "2020-08", "GraphCast"),
    ("fuxi",      "2020-08", "FuXi"),
    ("gencast",   "2020-08", "GenCast"),
]
MODELS_6H = [
    ("graphcast", "2020-08", "GraphCast"),
    ("fuxi",      "2020-08", "FuXi"),
    # GenCast has no 6-hr lead data
]
COLORS = ["#e6194b", "#3cb44b", "#4363d8"]


def load_summaries(docs: pathlib.Path, tag: str = "", lead: str = "24h") -> pd.DataFrame:
    models = MODELS_6H if lead == "6h" else MODELS_24H
    frames = []
    for model, month, label in models:
        path = docs / f"scale_summary_{model}_{month}_{lead}{tag}.csv"
        if not path.exists():
            print(f"missing: {path}")
            continue
        df = pd.read_csv(path)
        df["label"] = label
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No scale_summary_*_{lead}{tag}.csv files found in docs/")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default=str(REPO / "docs"))
    p.add_argument("--tag", default="",
                   help="suffix appended to CSV filenames, e.g. _thr10 (default: none)")
    p.add_argument("--lead", default="24h", choices=["6h", "24h"],
                   help="accumulation window: 6h or 24h (default: 24h)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    docs = pathlib.Path(args.docs)
    df = load_summaries(docs, tag=args.tag, lead=args.lead)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_pod, ax_far, ax_dlna, ax_dlnrho = axes.flat

    models = MODELS_6H if args.lead == "6h" else MODELS_24H
    labels_seen = []
    for (_, _, label), color in zip(models, COLORS):
        sub = df[df["label"] == label].sort_values("scale_mid")
        if sub.empty:
            continue
        x = sub["scale_mid"].values
        kw = dict(color=color, marker="o", markersize=4, linewidth=1.5, label=label)
        ax_pod.plot(x, sub["POD"].values, **kw)
        ax_far.plot(x, sub["FAR"].values, **kw)

        dlna = sub["dlnA_med"].values
        dlna_iqr = sub["dlnA_iqr"].values
        ax_dlna.plot(x, dlna, **kw)
        ax_dlna.fill_between(x, dlna - 0.5 * dlna_iqr, dlna + 0.5 * dlna_iqr,
                             color=color, alpha=0.15)

        dlnr = sub["dlnrho_med"].values
        dlnr_iqr = sub["dlnrho_iqr"].values
        ax_dlnrho.plot(x, dlnr, **kw)
        ax_dlnrho.fill_between(x, dlnr - 0.5 * dlnr_iqr, dlnr + 0.5 * dlnr_iqr,
                               color=color, alpha=0.15)

        labels_seen.append(label)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("System scale (km)")
        ax.grid(True, which="both", alpha=0.3)

    ax_pod.set_ylabel("POD")
    ax_pod.set_ylim(0, 1)
    ax_pod.set_title("(a) Detection rate")

    ax_far.set_ylabel("FAR")
    ax_far.set_ylim(0, 1)
    ax_far.set_title("(b) False-alarm rate")

    ax_dlna.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_dlna.set_ylabel("median dlnA")
    ax_dlna.set_title("(c) Area bias  (+: too large)")

    ax_dlnrho.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_dlnrho.set_ylabel("median dlnρ")
    ax_dlnrho.set_title("(d) Intensity bias  (−: too weak)")

    handles, lbls = ax_pod.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=len(labels_seen),
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    thr_note = f"  threshold={args.tag.lstrip('_')}" if args.tag else ""
    fig.suptitle(
        f"WB2 models — scale decomposition, August, tropics ±40°, {args.lead} precip{thr_note}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    stem = f"wb2_model_compare_{args.lead}{args.tag}"
    out = pathlib.Path(args.out or docs / "figures" / f"{stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
