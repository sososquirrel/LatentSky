"""Precipitation-intensity confusion matrix, one panel per --runs entry.
Bins the rain rate (mm/15min) into physical categories and row-normalises,
so cell (i,j) = P(predicted category j | truth category i). The diagonal
is correct classification; mass below the diagonal is under-prediction
(the smoothing/under-detection bias).

Genericized (2026-07) from a script hardcoded to
[("mseuv_lead0h",...), ("mseuv_lead3h",...)] into a --runs/--tag CLI tool
-- lead labels are now derived from each run's own config, not hardcoded
strings, and the output path is tagged so multiple model families don't
collide (see latentsky.paths.figure_path).
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from latentsky.registry import load_run
from latentsky.inference import predict_piece
from latentsky.paths import figure_path

# Rain-rate category edges (mm per 15-min accumulation).
EDGES = [0.0, 0.1, 1.0, 5.0, np.inf]
LABELS = ["dry\n($<0.1$)", "light\n($0.1$-$1$)", "mod.\n($1$-$5$)", "heavy\n($>5$)"]


def confusion(yt, yp):
    bt = np.digitize(yt.ravel(), EDGES) - 1
    bp = np.digitize(yp.ravel(), EDGES) - 1
    n = len(EDGES) - 1
    C = np.zeros((n, n))
    np.add.at(C, (bt, bp), 1)
    row = C.sum(axis=1, keepdims=True)
    return C / np.maximum(row, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                    help="run names, one panel per run (e.g. patchsplit_twohead_olr_fss_lead0.5h patchsplit_twohead_olr_fss_lead1h)")
    p.add_argument("--tag", required=True, help="model-family label, used in output filename")
    p.add_argument("--piece", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    fig, axes = plt.subplots(1, len(args.runs), figsize=(5.4 * len(args.runs), 4.6),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, run_name in zip(axes, args.runs):
        run = load_run(run_name)
        title = f"lead = {run.lead_hours:g} h"
        _, Y, Yp = predict_piece(run, args.piece, max_frames=args.max_frames, device=device)
        C = confusion(Y[:, 0], Yp[:, 0])
        im = ax.imshow(C, origin="upper", cmap="Blues", vmin=0, vmax=1)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        color="white" if C[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("predicted category"); ax.set_ylabel("truth category")
        ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="P(pred $|$ truth)  (row-normalised)")
    fig.suptitle(f"Precipitation-intensity confusion matrix ({args.tag}, "
                 f"piece {args.piece})", y=1.04, fontsize=11)

    out = figure_path("confusion_matrix", args.tag) if args.out is None else pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
