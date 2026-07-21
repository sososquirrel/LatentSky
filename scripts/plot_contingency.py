"""Headless plot for notebooks/03_contingency.ipynb.

Two outputs:
  - single-run contingency heatmap for the reference run
  - side-by-side comparison across multiple runs on a shared bin grid
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latentsky import load_run
from latentsky.inference import predict_piece
from latentsky.plotting import build_contingency_table, plot_contingency_heatmap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="mseuv_lead3h")
    p.add_argument("--piece", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--compare",
        nargs="+",
        default=["mseuv_lead3h", "humid_lead3h", "mse_lead3h"],
        help="Run names to compare side-by-side on a shared bin grid.",
    )
    p.add_argument("--nbins", type=int, default=30,
                   help="Number of linear bins over [0, vmax] mm.")
    p.add_argument("--outdir", default="docs/figures")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    outdir = (repo / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run = load_run(args.run)
    _, Y, Y_pred = predict_piece(run, args.piece, batch_size=args.batch_size)
    vmax_single = float(max(Y.max(), Y_pred.max()))
    single_bins = np.linspace(0.0, vmax_single, args.nbins + 1)
    df, confusion, bins = build_contingency_table(Y, Y_pred, bins=single_bins)
    print(df, flush=True)

    plot_contingency_heatmap(confusion, bins, title=f"{args.run} — piece {args.piece}")
    out = outdir / f"contingency_single_{args.run}_p{args.piece}.png"
    plt.gcf().savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"wrote {out}", flush=True)

    preds = {}
    for n in args.compare:
        r = load_run(n)
        _, Y, Yp = predict_piece(r, args.piece, batch_size=args.batch_size)
        preds[n] = (Y, Yp)

    vmax = max(max(Y.max(), Yp.max()) for Y, Yp in preds.values())
    shared_bins = np.linspace(0.0, float(vmax), args.nbins + 1)
    n_bins = args.nbins

    fig, axes = plt.subplots(1, len(args.compare), figsize=(6 * len(args.compare), 5))
    if len(args.compare) == 1:
        axes = [axes]
    tick_idx = np.unique(np.linspace(0, n_bins, 7).astype(int))
    for ax, name in zip(axes, args.compare):
        Y, Yp = preds[name]
        _, conf, _ = build_contingency_table(Y, Yp, bins=shared_bins)
        with np.errstate(divide="ignore"):
            log_conf = np.log(np.where(conf > 0, conf, np.nan))
        im = ax.imshow(log_conf, origin="upper", aspect="auto")
        ax.plot(np.arange(n_bins), np.arange(n_bins), "o", mfc="none", mec="red", ms=4)
        ax.set_xticks(tick_idx - 0.5)
        ax.set_yticks(tick_idx - 0.5)
        ax.set_xticklabels([f"{shared_bins[i]:.2g}" for i in tick_idx], rotation=45)
        ax.set_yticklabels([f"{shared_bins[i]:.2g}" for i in tick_idx])
        ax.set_title(name)
        ax.set_xlabel("predicted (mm)")
        ax.set_ylabel("truth (mm)")
        plt.colorbar(im, ax=ax, label="log(count)")
    fig.tight_layout()

    out = outdir / f"contingency_compare_p{args.piece}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
