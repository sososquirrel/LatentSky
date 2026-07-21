"""Figure: training input/output pipeline for the 9var (mse+uv) model --
what one training example actually looks like (input X, truth Y, model
prediction, loss). Complements fig17 (internal network architecture only).
"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"

IN_CH = 9
N_PARAMS = 124_377_665
INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa", "umeridional_850hPa"]

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-1, 8.2)
ax.axis("off")


def box(x, y, w, h, label, color, fontsize=10, fontcolor="white", fontweight="bold"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.03,rounding_size=0.08",
                       linewidth=1.5, edgecolor="k", facecolor=color, alpha=0.9, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=fontcolor,
           fontweight=fontweight, zorder=4)
    return b


def arrow(x0, y0, x1, y1, color="k", lw=1.8, style="-|>", ls="-", connectionstyle=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, linestyle=ls, zorder=2,
                        connectionstyle=connectionstyle)
    ax.add_patch(a)


# --- Input X ---
box(1.3, 4.5, 2.4, 2.6,
   "Input  X\n(t = 0)\n\n9 channels:\n" + "\n".join(INPUT_VARS),
   "#4C72B0", fontsize=7.8)

# --- Model (compact, points to fig17 for internal detail) ---
arrow(2.5, 4.5, 3.7, 4.5)
box(5.3, 4.5, 2.4, 1.6,
   f"UNet2D\nbase_ch=64, depth=5\n{N_PARAMS/1e6:.1f}M params\n(full architecture: fig17)",
   "#55A868", fontsize=8.5)

# --- Prediction ---
arrow(6.5, 4.5, 7.7, 4.5)
box(9.0, 4.5, 2.0, 1.4, "Prediction  ŷ\n(t = 0+lead)\n1 channel\nrain_total\n(Softplus, ≥0)",
   "#DD8452", fontsize=8.5)

# --- Truth (parallel path, NOT through the model) ---
box(9.0, 1.3, 2.0, 1.4, "Truth  Y\n(t = 0+lead)\n1 channel\nrain_total\n(raw data, same lead)",
   "0.35", fontsize=8.5)
ax.annotate("", xy=(1.3, 3.15), xytext=(1.3, 2.0),
           arrowprops=dict(arrowstyle="-", color="0.5", lw=1.4, ls=(0, (4, 3))))
ax.text(1.3, 1.6,
       "same dataset,\nshifted by\ndelta_index\nraw time-steps",
       ha="center", va="center", fontsize=7.5, color="0.4")
arrow(1.3, 2.0, 8.0, 1.3, color="0.4", ls=(0, (4, 3)), lw=1.4,
     connectionstyle="arc3,rad=-0.15")

# --- Loss ---
arrow(9.0, 3.75, 9.0, 2.05, color="k", lw=1.8)
arrow(9.9, 4.0, 10.9, 2.9, color="k", lw=1.8, connectionstyle="arc3,rad=0.15")
arrow(9.9, 1.6, 10.9, 2.6, color="k", lw=1.8, connectionstyle="arc3,rad=-0.15")
box(11.9, 3.0, 2.2, 1.6,
   "Loss\nmasked MSE\n\nmean( (ŷ−Y)² )\nover pixels\nwhere Y > 0",
   "#C44E52", fontsize=8.5)

ax.text(
    6.7, 7.9,
    "Training input/output pipeline — 9var (mse+uv)\n"
    "One training example: input snapshot X at t=0, model prediction ŷ, "
    "and truth Y at t=0+lead (same field, later time)",
    ha="center", va="top", fontsize=12,
)

out = OUT_DIR / "fig18_training_io_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}", flush=True)
