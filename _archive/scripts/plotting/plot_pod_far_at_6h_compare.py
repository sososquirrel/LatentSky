"""POD/FAR vs scale at lead=6h: no-rain vs rain-input (direct target) vs
rain-input (residual/persistence-anchored target).
"""
from __future__ import annotations
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import FIGURES_DIR

series = [
    ("no-rain input", "docs/scale_summary.csv", "mseuv_lead6h", "#1f77b4", "o"),
    ("rain input (direct target)", "docs/scale_summary_rain.csv", "mseuv_rain_lead6h", "#d62728", "s"),
    ("rain input (residual target)", "docs/scale_summary_rain_resid_lead6h.csv", "mseuv_rain_resid_lead6h", "#2ca02c", "^"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for label, path, name, color, marker in series:
    df = pd.read_csv(REPO / path)
    df = df[df["name"] == name].sort_values("scale_mid")
    df = df[df["n_truth"] > 0]  # drop empty bins (NaN POD)
    axes[0].plot(df["scale_mid"], df["POD"], marker + "-", color=color, ms=7, lw=1.6, label=label)
    axes[1].plot(df["scale_mid"], df["FAR"], marker + "-", color=color, ms=7, lw=1.6, label=label)

for ax, title in zip(axes, ["POD vs scale, lead=6h", "FAR vs scale, lead=6h"]):
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("system scale  L = √area  [km]", fontsize=10.5)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("POD", fontsize=10.5)
axes[1].set_ylabel("FAR", fontsize=10.5)
axes[0].legend(fontsize=8.5, frameon=False)

out = FIGURES_DIR / "pod_far_at_6h_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)
