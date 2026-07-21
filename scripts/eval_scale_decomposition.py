"""Scale-aware object decomposition for every trained run.

Headless equivalent of notebooks/04_scale_decomposition.ipynb — run under
PBS so the notebook doesn't sit on a GPU node. Writes two artifacts:

  docs/scale_systems.parquet  one row per precipitating system (truth/pred
                              matched, missed, or false-alarm) across all
                              runs. The raw table everything else derives
                              from.
  docs/scale_summary.csv      per (run, scale bin) object skill + the
                              log-additive precip-bias budget. Carries
                              predictor_set and lead_h so it slots into the
                              horizon study like horizon_vs_predictors.csv.
"""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import pandas as pd

from latentsky import list_runs, load_run
from latentsky.inference import evaluate_run_systems, daily_mean_ablation
from latentsky.scale_analysis import default_scale_edges, summarize_by_scale

# Reuse the predictor-set labeller from the horizon eval so the CSVs join.
from eval_horizon_vs_predictors import predictor_label  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems-out", default="docs/scale_systems.csv.gz",
                   help="Per-system table. .csv/.csv.gz need no extra deps; "
                        ".parquet is used only if pyarrow is importable.")
    p.add_argument("--summary-out", default="docs/scale_summary.csv")
    p.add_argument("--max-frames-per-piece", type=int, default=8)
    p.add_argument("--keep-frames", type=int, default=None,
                   help="Sample this many frames spread across each daily "
                        "piece (24 ≈ hourly) instead of the first "
                        "max-frames-per-piece. Enables the diurnal cycle "
                        "and adds day_frame/tod_hour/lst_hour columns.")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Rain threshold (mm) defining a precipitating pixel.")
    p.add_argument("--iou-min", type=float, default=0.1,
                   help="Min IoU for a truth<->pred system match.")
    p.add_argument("--scale-min", type=float, default=25.0)
    p.add_argument("--scale-max", type=float, default=3000.0)
    p.add_argument("--scale-bins", type=int, default=8)
    p.add_argument("--predictor-sets", default=None,
                   help="Comma-separated predictor-set labels to keep "
                        "(as produced by predictor_label, e.g. "
                        "'5var (mse),5var (relhum)'). Default: all runs.")
    p.add_argument("--runs", default=None,
                   help="Comma-separated exact run names to keep (e.g. "
                        "'mseuv_rain_lead0.25h,mseuv_rain_lead3h'). Use "
                        "when --predictor-sets would collide with other "
                        "runs sharing the same predictor_label. Default: "
                        "all runs.")
    p.add_argument("--ablation", default=None,
                   help="Input-channel names to replace by their per-cell "
                        "daily mean before inference (strips diurnal phase). "
                        "Channels not present in a run's input_vars are "
                        "silently skipped. Separator can be ',' or ':' — the "
                        "latter is friendlier to PBS `qsub -v` which treats "
                        "',' as the variable delimiter. e.g. 't2m' or "
                        "'t2m:mse_850hPa:olrtoa'.")
    args = p.parse_args()

    input_transform = None
    if args.ablation:
        raw = args.ablation.replace(":", ",")
        kill = [s.strip() for s in raw.split(",") if s.strip()]
        input_transform = daily_mean_ablation(kill)
        print(f"ablation: daily-mean replacement of channels {kill}", flush=True)

    repo = pathlib.Path(__file__).resolve().parents[1]
    edges = default_scale_edges(args.scale_min, args.scale_max, args.scale_bins)

    keep = None
    if args.predictor_sets:
        keep = {s.strip() for s in args.predictor_sets.split(",") if s.strip()}

    run_keep = None
    if args.runs:
        run_keep = {s.strip() for s in args.runs.split(",") if s.strip()}

    groups = defaultdict(list)
    for n in list_runs():
        if run_keep is not None and n not in run_keep:
            continue
        r = load_run(n)
        label = predictor_label(r.predictors)
        if keep is not None and label not in keep:
            continue
        groups[label].append(r)
    for runs in groups.values():
        runs.sort(key=lambda r: r.lead_hours)

    if keep is not None:
        missing = keep - set(groups)
        if missing:
            raise SystemExit(f"no runs matched predictor sets: {sorted(missing)}")
        print(f"filtered to predictor sets: {sorted(groups)}", flush=True)

    total = sum(len(v) for v in groups.values())
    print(f"scale decomposition for {total} runs, threshold={args.threshold} mm",
          flush=True)

    sys_frames, summary_rows = [], []
    i = 0
    for label, runs in groups.items():
        for r in runs:
            i += 1
            print(f"[{i}/{total}] {label}  lead={r.lead_hours}h  {r.name}",
                  flush=True)
            df = evaluate_run_systems(
                r,
                max_frames_per_piece=args.max_frames_per_piece,
                threshold=args.threshold,
                iou_min=args.iou_min,
                keep_frames=args.keep_frames,
                input_transform=input_transform,
            )
            if df.empty:
                print(f"  no systems for {r.name}", flush=True)
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

    systems = pd.concat(sys_frames, ignore_index=True)
    summary = pd.concat(summary_rows, ignore_index=True)

    sys_out = (repo / args.systems_out).resolve()
    sum_out = (repo / args.summary_out).resolve()
    sum_out.parent.mkdir(parents=True, exist_ok=True)

    # Write the small, dependency-free summary FIRST. A multi-hour GPU job
    # must never be lost to a serialization-library import error.
    summary.to_csv(sum_out, index=False)
    print(f"wrote {sum_out}  ({len(summary)} run x scale-bin rows)", flush=True)

    # Systems table: parquet only if pyarrow is available, else gzip CSV.
    if str(sys_out).endswith(".parquet"):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            sys_out = sys_out.with_suffix("").with_suffix(".csv.gz")
            print("pyarrow not available — writing systems as gzip CSV "
                  f"instead: {sys_out}", flush=True)
    if str(sys_out).endswith(".parquet"):
        systems.to_parquet(sys_out, index=False)
    else:
        systems.to_csv(sys_out, index=False,
                        compression="gzip" if str(sys_out).endswith(".gz") else None)
    print(f"wrote {sys_out}  ({len(systems)} systems)", flush=True)


if __name__ == "__main__":
    main()
