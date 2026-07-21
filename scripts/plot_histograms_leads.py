"""Log-scale truth vs prediction histograms for one experiment across several lead times.

Layout: 1 row × len(leads) cols. Bin edges shared across columns so the
distributions are directly comparable.
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


def _lead_to_name(prefix: str, lead: float) -> str:
    if float(lead).is_integer():
        return f"{prefix}_lead{int(lead)}h"
    return f"{prefix}_lead{lead}h"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="mseuv")
    p.add_argument("--leads", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    p.add_argument("--piece", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--outdir", default="docs/figures")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    outdir = (repo / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for lead in args.leads:
        name = _lead_to_name(args.prefix, lead)
        print(f"loading {name}...", flush=True)
        run = load_run(name)
        _, Y, Yp = predict_piece(run, args.piece, batch_size=args.batch_size)
        pairs.append((lead, name, Y[:, 0].ravel(), Yp[:, 0].ravel()))

    gmin = min(min(t.min(), p_.min()) for _, _, t, p_ in pairs)
    gmax = max(max(t.max(), p_.max()) for _, _, t, p_ in pairs)
    bw = (gmax - gmin) / args.bins
    bin_edges = np.arange(np.floor(gmin / bw) * bw, np.ceil(gmax / bw) * bw + bw, bw)

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (lead, name, t, p_) in zip(axes, pairs):
        ax.hist(t, bins=bin_edges, alpha=0.5, label="truth", edgecolor="black", log=True)
        ax.hist(p_, bins=bin_edges, alpha=0.5, label="prediction", edgecolor="black", log=True)
        ax.set_xlabel("rain (mm)")
        ax.set_title(f"{name}  (lead {lead}h)")
        ax.grid(True, linestyle=":", linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("count (log)")
            ax.legend()
    fig.suptitle(f"Truth vs pred histograms — {args.prefix}, piece {args.piece}", fontsize=12)
    fig.tight_layout()

    leads_tag = "_".join(
        (f"{int(lead)}h" if float(lead).is_integer() else f"{lead}h") for lead in args.leads
    )
    out = outdir / f"histograms_{args.prefix}_leads_{leads_tag}_p{args.piece}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
