"""Block-masking for pretrain-then-fine-tune (see latentsky/train.py's
`pretrain_mask_channel` config path).

Pretraining task: same architecture/loss as the real forecast run
(UNet2DTwoHead + combined_loss), same input_vars, but at lead=0 (predict
current rain from current everything) and with the persisted-rain input
channel (`rain_log1p`) randomly blanked out in blocks. Without the masking,
lead=0 would let the model shortcut rain_log1p -> rain_total almost
directly and learn nothing useful about the other physical variables;
masking forces it to infer rain from precipw/olrtoa/mse/wind in the blanked
regions, which is the representation that's actually useful to transfer
into the real forecasting fine-tune.

Block masking (not scattered single-pixel masking) matters: single pixels
are trivially interpolated from immediate neighbors and teach the model
nothing about longer-range cross-variable structure.
"""
from __future__ import annotations

import torch


def random_block_mask_channel(
    x: torch.Tensor,
    channel_idx: int,
    mask_frac: float,
    block_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out random block_size x block_size blocks of x[:, channel_idx]
    per-sample, covering approximately `mask_frac` of the spatial area.
    Returns a new tensor (input `x` is not modified in place).
    """
    x = x.clone()
    B, C, H, W = x.shape
    nrows = (H + block_size - 1) // block_size
    ncols = (W + block_size - 1) // block_size
    n_blocks = nrows * ncols
    n_mask = max(1, int(round(n_blocks * mask_frac)))

    for b in range(B):
        perm = torch.randperm(n_blocks, generator=generator)[:n_mask]
        for flat in perm.tolist():
            r, c = divmod(flat, ncols)
            i0, j0 = r * block_size, c * block_size
            x[b, channel_idx, i0 : i0 + block_size, j0 : j0 + block_size] = 0.0
    return x
