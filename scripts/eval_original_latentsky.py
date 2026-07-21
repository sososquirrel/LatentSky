"""Pixel-level R2/RMSE for the original (pre-session) LatentSky model
(mseuv_rain_resid_lead3h -- residual regression, old leaky day-level split
test_piece_ids=[2,12,22,32]), for the model-comparison plot. Evaluated on
its OWN test set, not patchsplit_test_blocks -- this model was never
trained under the patch-level split, so there is no fair way to score it on
those blocks; it's plotted alongside the others with that caveat labeled
explicitly rather than silently mixed in as if directly comparable.
"""
from __future__ import annotations
import math
import pathlib
import pandas as pd

from latentsky.inference import evaluate_run
from latentsky.registry import load_run

from latentsky.registry import REPO_ROOT as REPO

run = load_run("mseuv_rain_resid_lead3h")
metrics = evaluate_run(run, max_frames_per_piece=16)
print(metrics)

row = {
    "predictor_set": "latentsky_original",
    "eval_set": f"piece_{'_'.join(str(p) for p in run.config.test_piece_ids)}",
    "R2_all": metrics["R2_all"],
    "RMSE": math.sqrt(metrics["MSE"]),
}
out = REPO / "docs/latentsky_original_lead3h.csv"
pd.DataFrame([row]).to_csv(out, index=False)
print(f"wrote {out}")
