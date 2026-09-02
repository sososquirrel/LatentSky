"""Zoomed-in patch prediction crop, one panel per track, across ALL FIVE
model tracks + THREE baselines at a fixed lead -- run once per lead
(0.5h, 1h, 3h) to get the "same event, increasing lead time" comparison.
Tracks whose run/checkpoint isn't ready yet are skipped with a printed
warning, not a crash. Same crop-region convention as
scripts/plotting/plot_qualitative_maps.py's regional view (pixel-index
ranges, not degrees).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from latentsky.paths import FIGURES_DIR
from latentsky.predict_dispatch import predict_piece_track, tracks_for_lead

VMAX = 5.0  # mm/15min, shared color scale across every panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=float, required=True)
    p.add_argument("--piece", type=int, default=2)
    p.add_argument("--frame", type=int, default=0, help="which of the loaded frames to show")
    p.add_argument("--max-frames", type=int, default=4)
    p.add_argument("--row-range", type=int, nargs=2, default=[100, 250])
    p.add_argument("--col-range", type=int, nargs=2, default=[800, 1050])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    r0, r1 = args.row_range
    c0, c1 = args.col_range

    tracks = tracks_for_lead(args.lead)
    truth_crop = None
    panels = []  # (label, crop)
    for t in tracks:
        try:
            _, Y, Yp = predict_piece_track(t, args.piece, args.max_frames, device=device)
        except Exception as e:
            print(f"skip {t['label']!r}: {e}")
            continue
        if Y.shape[0] <= args.frame:
            print(f"skip {t['label']!r}: only {Y.shape[0]} frames, need index {args.frame}")
            continue
        if truth_crop is None:
            truth_crop = Y[args.frame, 0, r0:r1, c0:c1]
        panels.append((t["label"], Yp[args.frame, 0, r0:r1, c0:c1]))

    if truth_crop is None:
        raise SystemExit(f"no tracks available at lead={args.lead:g}h")

    n = 1 + len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), constrained_layout=True)
    im = axes[0].imshow(truth_crop, cmap="turbo", vmin=0, vmax=VMAX, origin="upper")
    axes[0].set_title("truth", loc="left", fontsize=9, fontweight="bold")
    for ax, (label, crop) in zip(axes[1:], panels):
        ax.imshow(crop, cmap="turbo", vmin=0, vmax=VMAX, origin="upper")
        ax.set_title(label, loc="left", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="rain rate (mm/15min)")
    fig.suptitle(f"Patch prediction, all tracks -- lead={args.lead:g}h "
                 f"(piece {args.piece}, frame {args.frame})", y=1.06, fontsize=11)

    out = pathlib.Path(args.out) if args.out else (
        FIGURES_DIR / f"2026-07-22_full_comparison/patch_prediction_compare_lead{args.lead:g}h.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
