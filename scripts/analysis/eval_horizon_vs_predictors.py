"""Evaluate every trained run on its test pieces, save metrics as CSV.

Headless equivalent of the evaluation cells in
notebooks/01_horizon_vs_predictors.ipynb. Run under PBS so the notebook
doesn't have to sit on a GPU node.
"""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import pandas as pd

from latentsky import list_runs, load_run
from latentsky.inference import evaluate_run


def predictor_label(input_vars):
    extras = [v for v in input_vars if v not in ("precipw", "t2m", "olrtoa")]
    if not extras:
        return "3var (pw,t2m,olr)"
    if any("mse" in v for v in extras) and any("uzonal" in v for v in extras):
        return "9var (mse+uv)"
    if any("mse" in v for v in extras):
        return "5var (mse)"
    if any("relhum" in v for v in extras):
        return "5var (relhum)"
    return ",".join(extras)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default="docs/horizon_vs_predictors.csv",
        help="Output CSV path (relative to repo root).",
    )
    p.add_argument(
        "--max-frames-per-piece",
        type=int,
        default=8,
        help="Frames to evaluate per test piece (matches notebook default).",
    )
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo

    groups = defaultdict(list)
    for n in list_runs():
        r = load_run(n)
        groups[predictor_label(r.predictors)].append(r)
    for runs in groups.values():
        runs.sort(key=lambda r: r.lead_hours)

    total = sum(len(v) for v in groups.values())
    print(f"evaluating {total} runs across {len(groups)} predictor sets", flush=True)

    rows = []
    i = 0
    for label, runs in groups.items():
        for r in runs:
            i += 1
            print(f"[{i}/{total}] {label}  lead={r.lead_hours}h  name={r.name}", flush=True)
            m = evaluate_run(r, max_frames_per_piece=args.max_frames_per_piece)
            rows.append({"predictor_set": label, "lead_h": r.lead_hours, "name": r.name, **m})

    df = (
        pd.DataFrame(rows)
        .sort_values(["predictor_set", "lead_h"])
        .reset_index(drop=True)
    )

    out = (repo / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
