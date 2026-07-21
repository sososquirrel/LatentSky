"""Headless plot for notebooks/02_qualitative_maps.ipynb.

Loads one run, runs predict_piece on one test piece, and saves four figures:
  - full-domain truth/pred/diff maps
  - regional crop of same
  - truth vs pred histograms
  - mean (pred - truth) binned in (PW, T2m) phase space
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky import load_run
from latentsky.inference import predict_piece
from latentsky.plotting import (
    plot_pw_tsfc_error,
    plot_regional_view,
    plot_regional_with_inputs,
    plot_two_histograms,
)


def save_current(out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="mseuv_lead3h")
    p.add_argument("--piece", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--outdir", default="docs/figures")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    outdir = (repo / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run = load_run(args.run)
    print(f"run={run.name}  lead={run.lead_hours}h  predictors={run.predictors}", flush=True)
    X, Y, Y_pred = predict_piece(run, args.piece, batch_size=args.batch_size)
    print(f"X{X.shape}  Y{Y.shape}  Y_pred{Y_pred.shape}", flush=True)

    plot_regional_view(Y, Y_pred, lat_range=[None, None], lon_range=[None, None],
                       figwidth=14, timesteps=1)
    save_current(outdir / f"qual_maps_{args.run}_p{args.piece}_full.png")

    plot_regional_view(Y, Y_pred, lat_range=[100, 250], lon_range=[800, 1050],
                       figwidth=8, timesteps=1)
    save_current(outdir / f"qual_maps_{args.run}_p{args.piece}_regional.png")

    plot_regional_with_inputs(X, Y, Y_pred,
                              lat_range=[100, 250], lon_range=[800, 1050],
                              sample_index=0, figwidth=14)
    save_current(outdir / f"qual_maps_{args.run}_p{args.piece}_regional_with_inputs.png")

    plot_two_histograms(Y[:, 0], Y_pred[:, 0], varname="rain", bins=80)
    save_current(outdir / f"qual_maps_{args.run}_p{args.piece}_histograms.png")

    plot_pw_tsfc_error(X, Y, Y_pred, pw_chan=0, tsfc_chan=1, nbins=(80, 80))
    save_current(outdir / f"qual_maps_{args.run}_p{args.piece}_pw_t2m_error.png")


if __name__ == "__main__":
    main()
