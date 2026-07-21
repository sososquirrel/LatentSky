"""Side-by-side contingency-heatmap comparison across predictor sets,
repeated for several lead times.

For each lead in --leads, builds one figure with three panels (mseuv, humid, mse),
on a shared bin grid, just like notebooks/03_contingency.ipynb does for one lead.
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
from latentsky.plotting import build_contingency_table


def lead_to_name(prefix: str, lead: float) -> str:
    if float(lead).is_integer():
        return f"{prefix}_lead{int(lead)}h"
    return f"{prefix}_lead{lead}h"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--piece", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--leads", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    p.add_argument("--prefixes", nargs="+", default=["mseuv", "humid", "mse"])
    p.add_argument("--nbins", type=int, default=30,
                   help="Number of linear bins over [0, vmax] mm.")
    p.add_argument("--outdir", default="docs/figures")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    outdir = (repo / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for lead in args.leads:
        names = [lead_to_name(pre, lead) for pre in args.prefixes]
        print(f"\n=== lead {lead}h: {names} ===", flush=True)

        preds = {}
        for n in names:
            r = load_run(n)
            _, Y, Yp = predict_piece(r, args.piece, batch_size=args.batch_size)
            preds[n] = (Y, Yp)

        vmax = max(max(Y.max(), Yp.max()) for Y, Yp in preds.values())
        shared_bins = np.linspace(0.0, float(vmax), args.nbins + 1)
        n_bins = args.nbins

        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
        if len(names) == 1:
            axes = [axes]
        tick_idx = np.unique(np.linspace(0, n_bins, 7).astype(int))
        for ax, name in zip(axes, names):
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
        fig.suptitle(f"contingency at lead {lead}h, piece {args.piece}", fontsize=13)
        fig.tight_layout()

        lead_tag = f"lead{int(lead)}h" if float(lead).is_integer() else f"lead{lead}h"
        out = outdir / f"contingency_compare_p{args.piece}_{lead_tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
