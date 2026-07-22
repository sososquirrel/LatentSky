"""Quick R2/RMSE for the existing one-head 10var (mseuv_rain_lead*) models
across the 0.25/0.5/1/2/3/6h leads -- baseline reference for comparing
against the new two-head + diffusion lead sweep once those finish training.
"""
from __future__ import annotations
import math
import pandas as pd

from latentsky.inference import evaluate_run
from latentsky.registry import load_run
from latentsky.paths import DOCS_RESULTS_DIR

leads = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]
rows = []
for lead in leads:
    name = f"mseuv_rain_lead{lead}"
    run = load_run(name)
    m = evaluate_run(run, max_frames_per_piece=16)
    rows.append({"name": name, "lead_h": run.config.lead_hours,
                 "R2_all": m["R2_all"], "RMSE": math.sqrt(m["MSE"])})
    print(rows[-1])

out = DOCS_RESULTS_DIR / "mseuv_rain_lead_sweep_baseline.csv"
pd.DataFrame(rows).sort_values("lead_h").to_csv(out, index=False)
print(f"wrote {out}")
