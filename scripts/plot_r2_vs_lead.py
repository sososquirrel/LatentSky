"""R^2 (object-level total precip, pred vs truth) as a function of forecast
lead time, for the LatentSky 9var(mse+uv) sweep.

Uses docs/scale_systems_v3.csv.gz: per detected/missed/false-alarm system,
P_t (truth precip) and P_p (pred precip). Misses get P_p=0 (nothing
predicted there), false alarms get P_t=0 (nothing there in truth) — this
scores the full reconstructed precip field at the object level, not just
matched systems, so a "safe" model that never predicts anything is
penalized properly (P_p=0 everywhere -> R^2 <= 0 relative to truth's own
variance, not an inflated match-only score).

  R^2 = 1 - sum((P_t - P_p)^2) / sum((P_t - mean(P_t))^2)
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


df = pd.read_csv(REPO / "docs/scale_systems_v3.csv.gz", compression="gzip",
                  usecols=["predictor_set", "lead_h", "role", "P_t", "P_p"])
df = df[df["predictor_set"] == "9var (mse+uv)"].copy()
df["P_t"] = df["P_t"].fillna(0.0)
df["P_p"] = df["P_p"].fillna(0.0)

leads = sorted(df["lead_h"].unique())
r2_vals = []
n_sys = []
for L in leads:
    sub = df[df["lead_h"] == L]
    r2_vals.append(r2(sub["P_t"].to_numpy(), sub["P_p"].to_numpy()))
    n_sys.append(len(sub))

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(leads, r2_vals, "o-", color="#1f77b4", ms=7, lw=1.8)
ax.axhline(0, color="0.4", lw=1.0, ls=":")
ax.text(leads[-1] * 0.98, 0.02, "R²=0 (no better than predicting the mean)",
        ha="right", fontsize=8, color="0.4")

ax.set_xlabel("forecast lead  [h]", fontsize=11)
ax.set_ylabel("R²  (object-level total precip, pred vs truth)", fontsize=11)
ax.set_title("LatentSky (9var mse+uv): explained variance vs lead time\n"
             "P per system, misses→P_p=0, false alarms→P_t=0", fontsize=10)
ax.grid(alpha=0.3)

for L, r, n in zip(leads, r2_vals, n_sys):
    ax.annotate(f"{r:.2f}", (L, r), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)

out = REPO / "docs/figures/r2_vs_lead_latentsky.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== R^2 vs lead ===")
for L, r, n in zip(leads, r2_vals, n_sys):
    print(f"lead={L:>5.2f}h   R2={r:>7.4f}   n_systems={n}")
