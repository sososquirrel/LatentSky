"""Bar-chart comparison of R2/RMSE on patchsplit_test_blocks: baselines
(persistence/climatology/random), the regression model alone, the
warm-started refinement diffusion sampler at each t_start, and the original
pure-noise-start diffusion sampler (annotated separately -- its R2=-486 is
two orders of magnitude off the scale of everything else).
"""
from __future__ import annotations
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]

baselines = pd.read_csv(REPO / "docs/baselines_lead3h_newsplits.csv")
baselines = baselines[baselines["eval_set"] == "patchsplit_test_blocks"]
refine = pd.read_csv(REPO / "docs/diffusion_refine_vs_baselines_lead3h.csv")
pure = pd.read_csv(REPO / "docs/diffusion_vs_baselines_lead3h.csv")
original = pd.read_csv(REPO / "docs/latentsky_original_lead3h.csv")

order = ["random", "persistence", "climatology", "latentsky_original", "regression_only",
         "diffusion_refine_tstart400", "diffusion_refine_tstart150", "diffusion_refine_tstart50"]
labels = {
    "random": "random",
    "persistence": "persistence",
    "climatology": "climatology",
    "latentsky_original": "LatentSky\n(original)*",
    "regression_only": "regression\n(patch-split\ntwo-head)",
    "diffusion_refine_tstart400": "refine\nt_start=400",
    "diffusion_refine_tstart150": "refine\nt_start=150",
    "diffusion_refine_tstart50": "refine\nt_start=50",
}
colors = {
    "random": "#b0b0b0",
    "persistence": "#b0b0b0",
    "climatology": "#b0b0b0",
    "latentsky_original": "#8172b2",
    "regression_only": "#4c72b0",
    "diffusion_refine_tstart400": "#dd8452",
    "diffusion_refine_tstart150": "#dd8452",
    "diffusion_refine_tstart50": "#dd8452",
}

all_df = pd.concat([baselines, original, refine], ignore_index=True)
all_df = all_df.set_index("predictor_set").loc[order].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
bars = ax.bar(labels_x := [labels[k] for k in all_df["predictor_set"]],
               all_df["R2_all"], color=[colors[k] for k in all_df["predictor_set"]])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R$^2$ (pixel-level)")
ax.set_title("R$^2$ on patch-split held-out test blocks")
ax.tick_params(axis="x", rotation=45)
for b, v in zip(bars, all_df["R2_all"]):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.015 if v >= 0 else -0.05),
            f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.text(0.02, 0.03,
        "diffusion sampled from pure noise\n(no warm start): R$^2$=-486, RMSE=4.35mm\n(off scale, catastrophic failure)",
        transform=ax.transAxes, fontsize=8, color="crimson", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.9))

ax = axes[1]
bars = ax.bar(labels_x, all_df["RMSE"], color=[colors[k] for k in all_df["predictor_set"]])
ax.set_ylabel("RMSE (mm)")
ax.set_title("RMSE on patch-split held-out test blocks")
ax.tick_params(axis="x", rotation=45)
for b, v in zip(bars, all_df["RMSE"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Warm-started ('residual/refinement') diffusion vs. baselines and pure-noise diffusion\n"
              "3h lead -- all bars on patch-split held-out test blocks, *except LatentSky (original),\n"
              "which predates that split and is shown on its own (older, day-level) held-out test days",
              fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.88])

out = REPO / "docs/figures/diffusion_refine_vs_baselines_lead3h.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"wrote {out}")
