"""Smoke test for latentsky.diffusion_inference.predict_piece_diffusion():
cross-checks against a known-good single-patch result computed earlier
tonight via a different code path (scripts/plotting/plot_example_prediction_grid.py,
piece 8, block (1,4)=rows 128:256 cols 512:640, frame 0, t_start=150,
ensemble_size=3 -> refine mean ~0.1306mm) to confirm the new full-piece
tiling wrapper reproduces the same order of magnitude (exact match not
expected -- diffusion sampling is stochastic).
"""
from __future__ import annotations
import numpy as np

from latentsky.diffusion_inference import predict_piece_diffusion
from latentsky.registry import load_run

run = load_run("diffusion_lead3h")
X, Y, Y_pred = predict_piece_diffusion(
    run, piece_id=8, max_frames=1, t_start=150, ensemble_size=3,
)
print("X", X.shape, "Y", Y.shape, "Y_pred", Y_pred.shape)

P = run.config.patch_size
i, j = 1 * P, 4 * P  # block (row=1, col=4)
truth_patch = Y[0, 0, i:i + P, j:j + P]
pred_patch = Y_pred[0, 0, i:i + P, j:j + P]
print(f"block (1,4) frame 0: truth mean={truth_patch.mean():.4f} pred mean={pred_patch.mean():.4f}")
print("(expected ballpark from earlier tonight: truth~0.1247, refine[t_start=150]~0.1306)")

print("\nfull-piece stats: Y_pred min/mean/max =", Y_pred.min(), Y_pred.mean(), Y_pred.max())
print("nonzero fraction of Y_pred (patches actually covered):", (Y_pred != 0).mean())
