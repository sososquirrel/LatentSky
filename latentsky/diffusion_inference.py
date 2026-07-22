"""Full-piece diffusion inference: warm-started, patch-tiled refinement
reassembled into a full (T,1,H,W) field.

Prior to this module, diffusion evaluation only existed on small test-block
patches (scripts/analysis/sample_and_eval_diffusion_refine.py) -- there was
no way to get a diffusion run's prediction over a whole piece/domain, so it
had no path into evaluate_run_systems / the object-detection pipeline that
every figure in scripts/plotting/ + scripts/analysis/ is built on.

predict_piece_diffusion() closes that gap by returning (X, Y, Y_pred) in
exactly the shape latentsky.inference.predict_piece() does, so it plugs
into the same downstream code unmodified -- e.g.
`latentsky.scale_analysis.system_table(Y, Y_pred, ...)`.

Cost warning: diffusion sampling costs `t_start+1` sequential forward
passes PER PATCH PER FRAME PER ENSEMBLE DRAW -- orders of magnitude more
expensive than predict_piece's single forward pass. This is meant to be
called with a deliberately bounded (piece, frame) scope, the same spirit as
sample_and_eval_diffusion_refine.py's --max-pieces/--frames-per-piece, not
swept over the full test set the way evaluate_run_systems is for
regression models.
"""
from __future__ import annotations

import numpy as np
import torch

from latentsky.diffusion import GaussianDiffusion
from latentsky.inference import _load_model, predict_piece
from latentsky.models import ConditionalDiffusionUNet
from latentsky.patch_split import compute_patch_grid
from latentsky.registry import Run, load_run


def predict_piece_diffusion(
    run: Run,
    piece_id: int,
    max_frames: int = 3,
    regression_run_name: str | None = None,
    t_start: int = 150,
    ensemble_size: int = 5,
    device: torch.device | None = None,
    diff_model: ConditionalDiffusionUNet | None = None,
    diffusion: GaussianDiffusion | None = None,
    reg_model=None,
):
    """Return (X, Y, Y_pred) for the first `max_frames` frames of a piece,
    with Y_pred from warm-started diffusion refinement (see
    latentsky.diffusion.GaussianDiffusion.ddpm_refine_sample) instead of a
    plain regression forward pass.

    `regression_run_name`: which regression run's prediction to warm-start
    from. Defaults to the matching-lead patch-split two-head run
    (`patchsplit_twohead_olr_fss_lead{run.lead_hours:g}h`) -- override for
    a different warm-start source.

    `t_start`/`ensemble_size`: same meaning as
    GaussianDiffusion.ddpm_refine_sample / the diffusion_refine sweep
    script. Larger t_start = more generative freedom, more compounding-
    drift risk (see tonight's t_start sweep: R^2 degraded from 0.39 to
    0.31 going from t_start=50 to t_start=400 on bulk pixel metrics).

    Pass pre-loaded `diff_model`/`diffusion`/`reg_model` to skip reloading
    across repeated calls (same spirit as predict_piece's `model` param).

    Note on domain coverage: patches tile the domain in non-overlapping
    `cfg.patch_size` blocks; if H/W isn't an exact multiple of patch_size
    (true today: 320/128=2.5), the remainder strip is left at 0 in Y_pred,
    same truncation behavior as the patch-split train/eval convention
    elsewhere in this project (latentsky.patch_split.compute_patch_grid).
    """
    cfg = run.config
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if regression_run_name is None:
        regression_run_name = f"patchsplit_twohead_olr_fss_lead{cfg.lead_hours:g}h"
    reg_run = load_run(regression_run_name)

    if diff_model is None:
        diff_model = ConditionalDiffusionUNet(
            cond_ch=cfg.in_channels, base_ch=cfg.base_ch, depth=cfg.depth,
            dropout_p=cfg.dropout_p, time_emb_dim=cfg.diffusion_time_emb_dim,
        )
        diff_model.load_state_dict(torch.load(run.checkpoint_path, map_location=device))
        diff_model = diff_model.to(device).eval()
    if diffusion is None:
        diffusion = GaussianDiffusion(cfg.diffusion_timesteps, cfg.diffusion_schedule, device=device)
    if reg_model is None:
        reg_model = _load_model(reg_run, device)

    X, Y, Yp_reg = predict_piece(reg_run, piece_id, max_frames=max_frames,
                                  device=device, model=reg_model)
    T, _, H, W = X.shape
    P = cfg.patch_size
    nrows, ncols = compute_patch_grid(H, W, P)
    scale = cfg.diffusion_target_scale

    Y_pred = np.zeros_like(Y)
    with torch.no_grad():
        for t in range(T):
            for r in range(nrows):
                for c in range(ncols):
                    i, j = r * P, c * P
                    cond = torch.from_numpy(X[t:t + 1, :, i:i + P, j:j + P]).to(device)
                    reg_patch = torch.from_numpy(Yp_reg[t:t + 1, :, i:i + P, j:j + P]).to(device)
                    x0_init = torch.log1p(torch.clamp(reg_patch, min=0.0)) / scale

                    draws = []
                    for _ in range(ensemble_size):
                        x0 = diffusion.ddpm_refine_sample(diff_model, cond, x0_init, t_start)
                        draws.append(torch.expm1(x0 * scale))
                    patch_pred = torch.clamp(torch.stack(draws).mean(dim=0), min=0.0)
                    Y_pred[t, :, i:i + P, j:j + P] = patch_pred.cpu().numpy()[0]

    return X, Y, Y_pred
