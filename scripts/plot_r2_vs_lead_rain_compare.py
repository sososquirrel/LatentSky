"""R^2 (object-level total precip, pred vs truth) vs lead time:
mseuv (no rain input) vs mseuv+rain (rain_log1p added as input).

Same scoring as plot_r2_vs_lead.py: misses -> P_p=0, false alarms -> P_t=0,
so a model that predicts nothing everywhere scores R^2<=0, not an
inflated match-only number.
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


def r2_by_lead(path, name_prefix):
    df = pd.read_csv(REPO / path, compression="gzip",
                      usecols=["lead_h", "name", "P_t", "P_p"])
    df = df[df["name"].str.startswith(name_prefix)].copy()
    df["P_t"] = df["P_t"].fillna(0.0)
    df["P_p"] = df["P_p"].fillna(0.0)
    leads, vals, ns = [], [], []
    for L in sorted(df["lead_h"].unique()):
        sub = df[df["lead_h"] == L]
        leads.append(L)
        vals.append(r2(sub["P_t"].to_numpy(), sub["P_p"].to_numpy()))
        ns.append(len(sub))
    return np.array(leads), np.array(vals), ns


def r2_by_lead_resid(leads_available):
    """Residual-model systems are written one file per lead (not a single
    combined file like the direct-target sweep) — only leads that have
    finished training+eval so far are included."""
    leads, vals, ns = [], [], []
    for L in leads_available:
        suffix = f"{L:g}h".replace(".0h", "h")
        path = REPO / f"docs/scale_systems_rain_resid_lead{suffix}.csv.gz"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["lead_h", "P_t", "P_p"])
        df["P_t"] = df["P_t"].fillna(0.0)
        df["P_p"] = df["P_p"].fillna(0.0)
        leads.append(L)
        vals.append(r2(df["P_t"].to_numpy(), df["P_p"].to_numpy()))
        ns.append(len(df))
    return np.array(leads), np.array(vals), ns


ls_L, ls_r2, ls_n = r2_by_lead("docs/scale_systems_v3.csv.gz", "mseuv_lead")
rn_L, rn_r2, rn_n = r2_by_lead("docs/scale_systems_rain.csv.gz", "mseuv_rain_lead")
rs_L, rs_r2, rs_n = r2_by_lead_resid([0.25, 0.5, 1, 2, 3, 6])

fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
ax.plot(ls_L, ls_r2, "o-", color="#1f77b4", ms=7, lw=1.8, label="mseuv (no rain input)")
ax.plot(rn_L, rn_r2, "s-", color="#d62728", ms=7, lw=1.8, label="mseuv + rain_log1p input (direct target)")
ax.plot(rs_L, rs_r2, "^-", color="#2ca02c", ms=8, lw=1.8,
        label="mseuv + rain_log1p input (residual target)"
              + ("" if len(rs_L) == 6 else f"  [{len(rs_L)}/6 leads done]"))
ax.axhline(0, color="0.4", lw=1.0, ls=":")
ax.text(6, 0.02, "R²=0 (no better than predicting the mean)",
        ha="right", fontsize=8, color="0.4")

ax.set_xlabel("forecast lead  [h]", fontsize=11)
ax.set_ylabel("R²  (object-level total precip, pred vs truth)", fontsize=11)
ax.set_title("Does adding precip as an input help?\n"
             "R² vs lead, with vs without rain_log1p in the input channels", fontsize=10)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.3)

out = REPO / "docs/figures/r2_vs_lead_rain_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== R^2 vs lead: mseuv (no rain) ===")
for L, r, n in zip(ls_L, ls_r2, ls_n):
    print(f"lead={L:>5.2f}h   R2={r:>7.4f}   n_systems={n}")
print("\n=== R^2 vs lead: mseuv + rain input (direct target) ===")
for L, r, n in zip(rn_L, rn_r2, rn_n):
    print(f"lead={L:>5.2f}h   R2={r:>7.4f}   n_systems={n}")
print("\n=== R^2 vs lead: mseuv + rain input (residual target) ===")
for L, r, n in zip(rs_L, rs_r2, rs_n):
    print(f"lead={L:>5.2f}h   R2={r:>7.4f}   n_systems={n}")
