"""5-row regional figure showing all 9 mseuv input channels + the precip
truth/pred/diff row. One PNG per run in --runs.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky import load_run
from latentsky.inference import predict_piece
from latentsky.plotting import plot_regional_with_all_inputs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["mseuv_lead3h"])
    p.add_argument("--piece", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lat-range", nargs=2, type=int, default=[100, 250])
    p.add_argument("--lon-range", nargs=2, type=int, default=[800, 1050])
    p.add_argument("--figwidth", type=float, default=14.0)
    p.add_argument("--outdir", default="docs/figures")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    outdir = (repo / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for run_name in args.runs:
        print(f"\nloading {run_name}...", flush=True)
        run = load_run(run_name)
        X, Y, Y_pred = predict_piece(run, args.piece, batch_size=args.batch_size)
        print(f"X{X.shape}  Y{Y.shape}  Y_pred{Y_pred.shape}", flush=True)
        if X.shape[1] < 9:
            print(f"  skip: {run_name} has only {X.shape[1]} channels (expected 9 for mseuv)",
                  flush=True)
            continue
        plot_regional_with_all_inputs(
            X, Y, Y_pred,
            lat_range=args.lat_range, lon_range=args.lon_range,
            sample_index=0, figwidth=args.figwidth,
        )
        fig = plt.gcf()
        old = fig._suptitle.get_text() if fig._suptitle else ""
        fig.suptitle(f"{run_name} — {old}", fontsize=12)
        out = outdir / f"qual_maps_{run_name}_p{args.piece}_regional_all_inputs.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
