"""Plain pixel-level prediction-vs-truth metrics (RMSE, R^2) for the
9var(mse+uv) lead sweep -- the "basic" comparison, no object decomposition.

Uses latentsky.inference.evaluate_run -> latentsky.metrics.compute_metrics
(MSE/R2_all/R2_rain/etc. over the full rain field, mm units). Runs the full
test pieces (not the 8-frame default) so this is on equal footing with the
persistence/climatology/random baselines, which score every available frame.

Writes docs/bulk_metrics_v1.csv (one row per run).
"""
from __future__ import annotations
import argparse
import pathlib
import pandas as pd

from latentsky import list_runs, load_run
from latentsky.inference import evaluate_run

from latentsky.registry import REPO_ROOT as REPO
RUNS = ["mseuv_lead0h", "mseuv_lead0.25h", "mseuv_lead0.5h", "mseuv_lead0.75h",
        "mseuv_lead1h", "mseuv_lead1.5h", "mseuv_lead2h", "mseuv_lead3h",
        "mseuv_lead4h", "mseuv_lead6h"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-frames-per-piece", type=int, default=94,
                   help="Full test piece length so coverage matches the "
                        "baseline evals (persistence/climatology/random "
                        "score every available frame).")
    p.add_argument("--out", default="docs/bulk_metrics_v1.csv")
    args = p.parse_args()

    all_names = set(list_runs())
    rows = []
    for i, name in enumerate(RUNS, 1):
        if name not in all_names:
            print(f"[{i}/{len(RUNS)}] {name}: NOT FOUND, skipping", flush=True)
            continue
        run = load_run(name)
        print(f"[{i}/{len(RUNS)}] {name}  lead={run.lead_hours}h ...", flush=True)
        m = evaluate_run(run, max_frames_per_piece=args.max_frames_per_piece)
        m["name"] = name
        m["lead_h"] = run.lead_hours
        rows.append(m)
        print(f"  MSE={m['MSE']:.4g}  R2_all={m['R2_all']:.3f}  R2_rain={m['R2_rain']:.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
