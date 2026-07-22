"""Bounded diffusion-track figures for figures/main/final_12/: confusion
matrix and tropic-wide map, using predict_piece_diffusion (patch-tiled,
warm-started refinement -- see latentsky/diffusion_inference.py). Kept to
a small frame/piece/ensemble scope since diffusion sampling costs
t_start+1 sequential forward passes per patch per frame per ensemble draw
-- far more expensive than the regression models' single forward pass.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.diffusion_inference import predict_piece_diffusion
from latentsky.registry import load_run

RUNS = [("diffusion_lead0.5h", "t+30min"), ("diffusion_lead1h", "t+1h")]
PIECE = 32
EDGES = [0.0, 0.1, 1.0, 5.0, np.inf]
LABELS = ["dry\n(<0.1)", "light\n(0.1-1)", "mod.\n(1-5)", "heavy\n(>5)"]


def confusion(yt, yp):
    bt = np.digitize(yt.ravel(), EDGES) - 1
    bp = np.digitize(yp.ravel(), EDGES) - 1
    n = len(EDGES) - 1
    C = np.zeros((n, n))
    np.add.at(C, (bt, bp), 1)
    return C / np.maximum(C.sum(axis=1, keepdims=True), 1)


results = {}
for name, label in RUNS:
    run = load_run(name)
    X, Y, Yp = predict_piece_diffusion(run, PIECE, max_frames=2, t_start=150, ensemble_size=3)
    results[name] = (label, X, Y, Yp)
    print(f"{name}: truth mean={Y.mean():.4f} pred mean={Yp.mean():.4f}", flush=True)

# --- confusion matrix ---
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
for ax, (name, (label, X, Y, Yp)) in zip(axes, results.items()):
    C = confusion(Y, Yp)
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("truth")
    ax.set_title(label, loc="left", fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.8, label="P(pred|truth)")
fig.suptitle(f"Diffusion -- confusion matrix (piece {PIECE})", y=1.04)
out = "figures/main/final_12/01_confusion_matrix_diffusion.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# --- tropic-wide map (frame 0 of each) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 6.5), constrained_layout=True)
for col, (name, (label, X, Y, Yp)) in enumerate(results.items()):
    vmax = max(np.percentile(Y[0, 0], 99.5), 1.0)
    axes[0, col].imshow(Y[0, 0], cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    axes[0, col].set_title(f"{label}: truth", fontsize=10)
    axes[1, col].imshow(Yp[0, 0], cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    axes[1, col].set_title(f"{label}: diffusion pred", fontsize=10)
    for ax in (axes[0, col], axes[1, col]):
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Diffusion -- tropic-wide truth vs pred (piece {PIECE}, frame 0)", y=1.03)
out = "figures/main/final_12/04_map_truth_vs_pred_diffusion.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
