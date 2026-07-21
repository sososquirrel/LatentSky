"""Multi-scale (FSS-style) + wet/dry classification loss for the two-head
UNet2DTwoHead model (see latentsky/models.py).

The existing training loss (`nn.MSELoss()`, masked to wet pixels only,
`latentsky/train.py`) never sees dry pixels and has no spatial/neighborhood
structure -- both are addressed here:

  mse : identical masked-MSE math to today, unchanged.
  bce : wet/dry classification on the FULL spatial tensor (every pixel,
        wet and dry), giving the model its first-ever training signal about
        NOT predicting rain in dry regions.
  fss : a differentiable Fractions-Skill-Score-style term computed at
        multiple pooling neighborhood sizes, reusing the same L=sqrt(area)
        scale-bin edges (`scale_analysis.default_scale_edges`) used
        everywhere else in this project's evaluation -- so training and
        evaluation finally share one scale vocabulary.

mse operates on flattened, masked pixels (spatial structure discarded, fine
since it's a pointwise term). bce and fss both need the full (B,1,H,W)
tensor and so are computed *before* any masking/flattening.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from latentsky.scale_analysis import default_scale_edges

# Grid is a 0.25-deg tropical crop (lat -40..40); meridional spacing is
# constant across all rows (zonal spacing shrinks toward the poles by
# cos(lat), but a conv/pool kernel can't vary per row, so meridional
# spacing is the natural fixed reference -- matches default_scale_edges's
# own docstring anchor, "~25km ~= one 0.25-deg cell").
KM_PER_PIXEL = 27.80


def scale_kernel_sizes(
    pixel_km: float = KM_PER_PIXEL,
    max_kernel: int = 33,
    scale_edges_km: np.ndarray | None = None,
) -> list[int]:
    """Convert the project's km scale-bin edges to odd pooling-kernel sizes
    (pixels), capped at `max_kernel` and deduplicated. Default edges give
    [1, 3, 5, 11, 19, 33]."""
    edges = scale_edges_km if scale_edges_km is not None else default_scale_edges()

    def nearest_odd(v: float) -> int:
        k = int(round(v))
        return max(1, k + 1) if k % 2 == 0 else max(1, k)

    sizes = sorted({min(nearest_odd(L / pixel_km), max_kernel) for L in edges})
    return sizes


def multiscale_fss_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    kernel_sizes: list[int],
    threshold: float = 0.1,
    sharpness: float = 20.0,
) -> torch.Tensor:
    """FSS-style loss on full (B,1,H,W) tensors -- no flatten/mask.

    truth's wet indicator is a hard threshold (truth needs no gradient).
    pred's indicator is a sigmoid relaxation of the same threshold so the
    term is differentiable. At each kernel size, average-pool both
    indicator fields to get local wet-fraction fields, then MSE between
    them; average the per-scale MSEs.
    """
    pred_wet = torch.sigmoid((pred - threshold) * sharpness)
    truth_wet = (truth > threshold).float()

    terms = []
    for k in kernel_sizes:
        if k == 1:
            p_frac, t_frac = pred_wet, truth_wet
        else:
            pad = k // 2
            p_frac = F.avg_pool2d(pred_wet, k, stride=1, padding=pad, count_include_pad=False)
            t_frac = F.avg_pool2d(truth_wet, k, stride=1, padding=pad, count_include_pad=False)
        terms.append(F.mse_loss(p_frac, t_frac))
    return torch.stack(terms).mean()


def combined_loss(
    rain_pred: torch.Tensor,
    wetdry_logit: torch.Tensor,
    y_target: torch.Tensor,
    wet_mask: torch.Tensor,
    kernel_sizes: list[int],
    bce_weight: float,
    fss_weight: float,
    fss_threshold: float,
    fss_sharpness: float,
) -> dict[str, torch.Tensor]:
    if wet_mask.any():
        mse = F.mse_loss(rain_pred[wet_mask], y_target[wet_mask])
    else:
        mse = rain_pred.new_zeros(())
    bce = F.binary_cross_entropy_with_logits(wetdry_logit, wet_mask.float())
    fss = multiscale_fss_loss(rain_pred, y_target, kernel_sizes, fss_threshold, fss_sharpness)
    total = mse + bce_weight * bce + fss_weight * fss
    return {"total": total, "mse": mse.detach(), "bce": bce.detach(), "fss": fss.detach()}
