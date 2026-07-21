"""Results summary for the mseuv_rain_lead* training sweep (6 lead times,
0.25-6h, rain_log1p included as an input variable alongside the 9 standard
atmospheric fields). Two panels: val-loss curves per run, and final
train/val loss vs lead time.
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from latentsky.registry import REPO_ROOT as REPO
RUNS = pathlib.Path("/glade/derecho/scratch/sabramian/latentsky_runs")

LEADS = [0.25, 0.5, 1, 2, 3, 6]
CONFIGS = [f"mseuv_rain_lead{L}h" if L != int(L) or L in (0.25, 0.5)
           else f"mseuv_rain_lead{int(L)}h" for L in LEADS]
# manifest folder names use the same string as in configs/*.yaml "name:" field
CONFIGS = ["mseuv_rain_lead0.25h", "mseuv_rain_lead0.5h", "mseuv_rain_lead1h",
           "mseuv_rain_lead2h", "mseuv_rain_lead3h", "mseuv_rain_lead6h"]

runs = []
for lead, cfg in zip(LEADS, CONFIGS):
    manifest = json.load(open(RUNS / cfg / "manifest.json"))
    runs.append({
        "lead": lead,
        "name": cfg,
        "train": np.array(manifest["loss_history"]["train"]),
        "val": np.array(manifest["loss_history"]["val"]),
    })

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

cmap = cm.viridis
for i, r in enumerate(runs):
    color = cmap(i / (len(runs) - 1))
    epochs = np.arange(1, len(r["val"]) + 1)
    axes[0].plot(epochs, r["val"], color=color, lw=1.6,
                 label=f"lead={r['lead']}h  (n={len(epochs)} epochs)")

axes[0].set_xlabel("epoch")
axes[0].set_ylabel("validation loss (MSE, rain_log1p target)")
axes[0].set_yscale("log")
axes[0].set_title("Val-loss curves by lead time\n(early-stopped, patience=10)")
axes[0].legend(fontsize=8, frameon=False)
axes[0].grid(alpha=0.25, which="both")

final_train = [r["train"][-1] for r in runs]
final_val = [r["val"][-1] for r in runs]
best_val = [r["val"].min() for r in runs]

axes[1].plot(LEADS, final_train, "o-", color="#1f77b4", label="final train loss")
axes[1].plot(LEADS, final_val, "s-", color="#d62728", label="final val loss")
axes[1].plot(LEADS, best_val, "^--", color="#d62728", alpha=0.5, label="best val loss")
axes[1].set_xlabel("forecast lead  [h]")
axes[1].set_ylabel("loss (MSE, rain_log1p target)")
axes[1].set_title("Skill vs lead time\n(mseuv + rain_log1p input, 9var+rain)")
axes[1].legend(fontsize=9, frameon=False)
axes[1].grid(alpha=0.25)

out = REPO / "docs/figures/rain_sweep_results.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== summary ===")
print(f"{'lead_h':>7}  {'epochs':>7}  {'final_train':>12}  {'final_val':>10}  {'best_val':>9}")
for r, ft, fv, bv in zip(runs, final_train, final_val, best_val):
    print(f"{r['lead']:>7}  {len(r['train']):>7}  {ft:>12.5f}  {fv:>10.5f}  {bv:>9.5f}")
