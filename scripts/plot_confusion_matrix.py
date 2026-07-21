"""Precipitation-intensity confusion matrix for the 9-variable model at a
couple of leads. Bins the rain rate (mm/15min) into physical categories and
row-normalises, so cell (i,j) = P(predicted category j | truth category i).
The diagonal is correct classification; mass below the diagonal is
under-prediction (the smoothing/under-detection bias)."""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from latentsky import load_run
from latentsky.inference import predict_piece

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "confusion_matrix.png"

# Rain-rate category edges (mm per 15-min accumulation).
EDGES = [0.0, 0.1, 1.0, 5.0, np.inf]
LABELS = ["dry\n($<0.1$)", "light\n($0.1$–$1$)", "mod.\n($1$–$5$)", "heavy\n($>5$)"]
RUNS = [("mseuv_lead0h", "lead = 0 h"), ("mseuv_lead3h", "lead = 3 h")]
PIECE = 2
MAX_FRAMES = 16


def confusion(yt, yp):
    bt = np.digitize(yt.ravel(), EDGES) - 1
    bp = np.digitize(yp.ravel(), EDGES) - 1
    n = len(EDGES) - 1
    C = np.zeros((n, n))
    np.add.at(C, (bt, bp), 1)
    row = C.sum(axis=1, keepdims=True)
    return C / np.maximum(row, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    fig, axes = plt.subplots(1, len(RUNS), figsize=(5.4 * len(RUNS), 4.6),
                             constrained_layout=True)
    for ax, (run_name, title) in zip(axes, RUNS):
        run = load_run(run_name)
        _, Y, Yp = predict_piece(run, PIECE, max_frames=MAX_FRAMES, device=device)
        C = confusion(Y[:, 0], Yp[:, 0])
        im = ax.imshow(C, origin="upper", cmap="Blues", vmin=0, vmax=1)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        color="white" if C[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("predicted category"); ax.set_ylabel("truth category")
        ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="P(pred $|$ truth)  (row-normalised)")
    fig.suptitle("Precipitation-intensity confusion matrix (9-variable model, "
                 f"piece {PIECE})", y=1.04, fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
