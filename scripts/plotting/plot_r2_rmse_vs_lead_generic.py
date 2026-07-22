"""R2 and RMSE vs lead time, for an arbitrary list of runs (any architecture
-- one_head, two_head; diffusion needs latentsky.diffusion_inference first).

New (2026-07): no prior script swept live evaluate_run() over an arbitrary
run list to build this -- every existing R2-vs-lead plot instead read a
precomputed, family-specific CSV (docs/scale_summary_v3.csv filtered by
`predictor_set`, etc). This one calls latentsky.inference.evaluate_run()
directly per run, so it works for any run in the registry without needing a
matching precomputed CSV to exist first.
"""
from __future__ import annotations
import argparse
import math
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.registry import load_run
from latentsky.inference import evaluate_run
from latentsky.paths import figure_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run names, one lead-sweep family")
    p.add_argument("--tag", required=True, help="model-family label, used in output filename")
    p.add_argument("--max-frames-per-piece", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for name in args.runs:
        run = load_run(name)
        m = evaluate_run(run, max_frames_per_piece=args.max_frames_per_piece)
        rows.append({"name": name, "lead_h": run.lead_hours,
                     "R2_all": m["R2_all"], "RMSE": math.sqrt(m["MSE"])})
        print(rows[-1], flush=True)

    rows.sort(key=lambda r: r["lead_h"])
    lead_h = [r["lead_h"] for r in rows]
    r2 = [r["R2_all"] for r in rows]
    rmse = [r["RMSE"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(lead_h, r2, "o-", color="#1f77b4", ms=7, lw=1.8)
    axes[0].axhline(0, color="0.3", lw=0.8, ls=":")
    axes[0].set_xlabel("lead time [h]", fontsize=10.5)
    axes[0].set_ylabel("R^2 (pixel-level)", fontsize=10.5)
    axes[0].set_title(f"{args.tag}: R^2 vs lead", fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].plot(lead_h, rmse, "o-", color="#d62728", ms=7, lw=1.8)
    axes[1].set_xlabel("lead time [h]", fontsize=10.5)
    axes[1].set_ylabel("RMSE (mm)", fontsize=10.5)
    axes[1].set_title(f"{args.tag}: RMSE vs lead", fontsize=11)
    axes[1].grid(alpha=0.3)

    out = figure_path("r2_rmse_vs_lead", args.tag) if args.out is None else pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
