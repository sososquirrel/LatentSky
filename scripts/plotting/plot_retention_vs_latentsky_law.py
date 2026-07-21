"""Does GraphCast/FuXi's ACTUAL (measured, non-extrapolated) skill retention
from 6h->24h match what LatentSky's own physical decay law would predict?

The problem this sidesteps: fitting an independent tau for GraphCast/FuXi
from only 4 sparse points (6,12,18,24h) means those tau values are almost
entirely extrapolated beyond the data (see plot_tau_vs_scale_ai_models.py's
hollow-marker flagging) — not trustworthy as standalone numbers.

But we don't need an independent tau to ask a narrower, fully-supported
question: given LatentSky's own measured tau(L) law (fit from real 0-6h
decay, the only model we watched decay at fine time resolution), how much
POD retention would that law predict over an 18h interval (6h->24h) at a
given scale -- and does GraphCast/FuXi's ACTUAL retention over that same
real interval (both endpoints measured, zero fitting) beat it or fall
short?

retention_actual(s)    = POD_model(24h,s) / POD_model(6h,s)   [measured]
retention_LS_law(s)    = exp(-18 / tau_LatentSky(s))          [from the
                          trusted LatentSky power-law fit]
ratio(s) = retention_actual(s) / retention_LS_law(s)
  ratio > 1: model retains MORE skill over 6-24h than LatentSky's own
             decay law implies for that scale (encouraging).
  ratio < 1: model decays FASTER over 6-24h than LatentSky's law implies.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import FIGURES_DIR


def fit_latentsky_power_law():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[df["predictor_set"] == "9var (mse+uv)"]
    scales = sorted(df["scale_mid"].unique())
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    sc_out, tau_out = [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(s, np.nan)
        if np.isnan(p0) or p0 <= 0:
            continue
        norm_pod = (sub["POD"] / p0).values
        popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), t, norm_pod,
                            p0=[2.0], bounds=([0.05], [500.0]))
        sc_out.append(s)
        tau_out.append(popt[0])
    sc_out, tau_out = np.array(sc_out), np.array(tau_out)
    b, lna = np.polyfit(np.log(sc_out), np.log(tau_out), 1)
    return np.exp(lna), b  # tau_LatentSky(L) = a * L^b


def actual_retention(model, scale_lo_h=6, scale_hi_h=24):
    lo = pd.read_csv(REPO / f"docs/scale_summary_{model}_2020-08_lead{scale_lo_h}h_thr2p4.csv")
    hi = pd.read_csv(REPO / f"docs/scale_summary_{model}_2020-08_lead{scale_hi_h}h_thr2p4.csv")
    lo = lo.set_index("scale_mid")["POD"]
    hi = hi.set_index("scale_mid")["POD"]
    common = lo.index.intersection(hi.index)
    common = common[lo.loc[common] > 0]
    return common.values, (hi.loc[common] / lo.loc[common]).values


a_ls, b_ls = fit_latentsky_power_law()
print(f"LatentSky law: tau(L) = {a_ls:.4f} * L^{b_ls:.3f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

results = {}
for model, color, marker in [("graphcast", "#2ca02c", "s"), ("fuxi", "#ff7f0e", "D")]:
    scales, ret_actual = actual_retention(model)
    tau_ls_at_scale = a_ls * scales ** b_ls
    ret_predicted = np.exp(-18.0 / tau_ls_at_scale)
    ratio = ret_actual / ret_predicted
    results[model] = (scales, ret_actual, ret_predicted, ratio)

    axes[0].plot(scales, ret_actual, marker + "-", color=color, ms=8, lw=1.6,
                 label=f"{model} actual (measured, no fit)")
    axes[1].plot(scales, ratio, marker + "-", color=color, ms=8, lw=1.6, label=model)

axes[0].plot(*[a_ls * np.geomspace(25, 600, 100) ** b_ls] * 0, alpha=0)  # noop, keep axis clean
L_ref = np.geomspace(25, 600, 100)
tau_ref = a_ls * L_ref ** b_ls
axes[0].plot(L_ref, np.exp(-18.0 / tau_ref), "k--", lw=1.8,
             label="LatentSky law prediction\n(exp(-18h / τ_LatentSky(L)))")
axes[0].set_xscale("log")
axes[0].set_xlabel("system scale  L = √area  [km]", fontsize=10.5)
axes[0].set_ylabel("POD(24h) / POD(6h)  (fraction retained)", fontsize=10.5)
axes[0].set_title("Actual measured retention, 6h→24h\nvs LatentSky's physical decay law", fontsize=10.5)
axes[0].legend(fontsize=8, frameon=False)
axes[0].grid(alpha=0.3)

axes[1].axhline(1.0, color="0.3", lw=1.2, ls=":")
axes[1].text(30, 1.03, "= matches LatentSky's law", fontsize=8, color="0.3")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("system scale  L = √area  [km]", fontsize=10.5)
axes[1].set_ylabel("ratio = actual retention / LatentSky-law-predicted retention", fontsize=9.5)
axes[1].set_title("Above 1: model beats LatentSky's decay law\nBelow 1: decays faster than the law implies", fontsize=10.5)
axes[1].legend(fontsize=9, frameon=False)
axes[1].grid(alpha=0.3, which="both")

out = FIGURES_DIR / "retention_vs_latentsky_law.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== retention ratio (actual / LatentSky-law-predicted), 6h->24h ===")
for model, (scales, ret_a, ret_p, ratio) in results.items():
    print(f"\n{model}:")
    for s, ra, rp, r in zip(scales, ret_a, ret_p, ratio):
        print(f"  L={s:>7.1f}km  actual={ra:.3f}  LS-law-predicted={rp:.3f}  ratio={r:.2f}")
