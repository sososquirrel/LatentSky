"""Quick 2-case peek at the scale decomposition while the full PBS run
(eval_scale_decomposition.py) is still in flight. Writes to *_peek.* paths
so it never collides with docs/scale_systems.* from the canonical job.

Usage: python scripts/peek_scale_decomp.py [run_name ...]
Defaults to two contrasting lead=0h runs (5-var relhum vs 9-var mse+uv).
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

from latentsky import load_run
from latentsky.inference import evaluate_run_systems
from latentsky.scale_analysis import default_scale_edges, summarize_by_scale
from eval_horizon_vs_predictors import predictor_label  # type: ignore

DEFAULT_RUNS = ["humid_lead0h", "mseuv_lead0h"]


def main() -> None:
    names = sys.argv[1:] or DEFAULT_RUNS
    repo = pathlib.Path(__file__).resolve().parents[1]
    edges = default_scale_edges(25.0, 3000.0, 8)

    sys_frames, summary_rows = [], []
    for i, name in enumerate(names, 1):
        r = load_run(name)
        label = predictor_label(r.predictors)
        print(f"[{i}/{len(names)}] {label}  lead={r.lead_hours}h  {r.name}",
              flush=True)
        df = evaluate_run_systems(r, max_frames_per_piece=8,
                                  threshold=0.1, iou_min=0.1)
        if df.empty:
            print(f"  (no systems for {name})", flush=True)
            continue
        df.insert(0, "name", r.name)
        df.insert(0, "lead_h", r.lead_hours)
        df.insert(0, "predictor_set", label)
        sys_frames.append(df)
        summ = summarize_by_scale(df, edges).by_scale
        summ.insert(0, "name", r.name)
        summ.insert(0, "lead_h", r.lead_hours)
        summ.insert(0, "predictor_set", label)
        summary_rows.append(summ)

    summary = pd.concat(summary_rows, ignore_index=True)
    systems = pd.concat(sys_frames, ignore_index=True)
    sum_out = repo / "docs" / "scale_summary_peek.csv"
    sys_out = repo / "docs" / "scale_systems_peek.csv.gz"
    summary.to_csv(sum_out, index=False)
    systems.to_csv(sys_out, index=False)
    print(f"wrote {sum_out}  ({len(summary)} rows)", flush=True)
    print(f"wrote {sys_out}  ({len(systems)} systems)", flush=True)
    print("\n=== summary preview ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
