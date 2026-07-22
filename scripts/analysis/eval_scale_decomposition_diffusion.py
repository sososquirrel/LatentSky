"""Scale-aware object decomposition for the diffusion track -- same output
schema as scripts/analysis/eval_scale_decomposition.py (systems/summary
CSV consumed unmodified by every plot_*.py / fit_*.py figure script), but
built from latentsky.diffusion_inference.evaluate_run_systems_diffusion
instead of latentsky.inference.evaluate_run_systems.

Bounded scope (--pieces/--max-frames-per-piece/--ensemble-size/--t-start):
diffusion sampling costs t_start+1 sequential forward passes per patch per
frame per ensemble draw -- far more expensive than the regression models'
single forward pass, so this does NOT sweep the full test-piece set the
way the regression eval does. Scope is always printed so results aren't
mistaken for full coverage.

Applies the same patch-split test-block leakage filter as
eval_scale_decomposition.py (latentsky.patch_split.
restrict_to_patch_test_blocks) -- diffusion is warm-started from a
patch-split regression run and shares its train/test patch assignment.
"""
from __future__ import annotations

import argparse

import pandas as pd

from latentsky.diffusion_inference import evaluate_run_systems_diffusion
from latentsky.patch_split import restrict_to_patch_test_blocks
from latentsky.registry import REPO_ROOT as REPO
from latentsky.registry import load_run
from latentsky.scale_analysis import default_scale_edges, summarize_by_scale


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="Comma-separated diffusion run names.")
    p.add_argument("--pieces", nargs="+", type=int, default=[8, 15, 25])
    p.add_argument("--max-frames-per-piece", type=int, default=2)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--t-start", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=3000.0)
    p.add_argument("--scale-bins", type=int, default=8)
    p.add_argument("--systems-out", default="docs/scale_systems_diffusion.csv.gz")
    p.add_argument("--summary-out", default="docs/scale_summary_diffusion.csv")
    args = p.parse_args()

    runs = [s.strip() for s in args.runs.split(",") if s.strip()]
    print(f"scope: pieces={args.pieces} max_frames_per_piece={args.max_frames_per_piece} "
          f"ensemble_size={args.ensemble_size} t_start={args.t_start} "
          f"(NOT full coverage -- see script docstring)", flush=True)

    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)

    sys_frames, summary_rows = [], []
    for i, name in enumerate(runs, 1):
        r = load_run(name)
        print(f"[{i}/{len(runs)}] lead={r.lead_hours}h  {r.name}", flush=True)
        df = evaluate_run_systems_diffusion(
            r, piece_ids=args.pieces, max_frames_per_piece=args.max_frames_per_piece,
            threshold=args.threshold, iou_min=args.iou_min,
            t_start=args.t_start, ensemble_size=args.ensemble_size,
        )
        if df.empty:
            print(f"  no systems for {r.name}", flush=True)
            continue
        n_before = len(df)
        df = restrict_to_patch_test_blocks(df, r)
        print(f"  patch-split filter: {n_before} -> {len(df)} systems (restricted to held-out test blocks)",
              flush=True)
        if df.empty:
            print(f"  no systems remain after test-block filter for {r.name}", flush=True)
            continue
        df.insert(0, "name", r.name)
        df.insert(0, "lead_h", r.lead_hours)
        df.insert(0, "predictor_set", "diffusion")
        sys_frames.append(df)

        summ = summarize_by_scale(df, edges).by_scale
        summ.insert(0, "name", r.name)
        summ.insert(0, "lead_h", r.lead_hours)
        summ.insert(0, "predictor_set", "diffusion")
        summary_rows.append(summ)

    systems = pd.concat(sys_frames, ignore_index=True)
    summary = pd.concat(summary_rows, ignore_index=True)

    out_sys = REPO / args.systems_out
    out_summ = REPO / args.summary_out
    systems.to_csv(out_sys, index=False)
    summary.to_csv(out_summ, index=False)
    print(f"wrote {out_summ}  ({len(summary)} run x scale-bin rows)")
    print(f"wrote {out_sys}  ({len(systems)} systems)")


if __name__ == "__main__":
    main()
