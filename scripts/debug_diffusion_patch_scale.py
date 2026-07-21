"""Diagnostic: does DDIM sampling work at the model's actual TRAINING
resolution (128x128 patches), as opposed to the full 320x1440 domain used
in sample_and_eval_diffusion.py? Isolates whether a resolution mismatch is
the cause of the poor full-domain sampling results.
"""
from __future__ import annotations
import numpy as np
import torch

from latentsky.data import load_piece_arrays
from latentsky.diffusion import GaussianDiffusion
from latentsky.models import ConditionalDiffusionUNet
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.registry import load_run


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float64)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2, dtype=np.float64)
    return 1.0 - ss_res / ss_tot


def main():
    run = load_run("diffusion_lead3h")
    cfg = run.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model = ConditionalDiffusionUNet(
        cond_ch=cfg.in_channels, base_ch=cfg.base_ch, depth=cfg.depth,
        dropout_p=cfg.dropout_p, time_emb_dim=cfg.diffusion_time_emb_dim,
    )
    state = torch.load(run.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    diffusion = GaussianDiffusion(cfg.diffusion_timesteps, cfg.diffusion_schedule, device=device)
    scale = cfg.diffusion_target_scale
    P = cfg.patch_size

    probe = np.load(f"{cfg.pieces_dir}/{cfg.input_vars[0]}_piece00.npy", mmap_mode="r")
    _, H, W = probe.shape
    nrows, ncols = compute_patch_grid(H, W, P)
    train_blocks, val_blocks, test_blocks = make_patch_split(
        cfg.n_total_pieces, nrows, ncols, cfg.patch_split_seed,
        cfg.patch_split_train_frac, cfg.patch_split_val_frac,
    )

    # Use a TRAIN block (in-distribution, resolution-matched) -- isolates
    # sampler correctness from any generalization-to-held-out-data question.
    pid = 3
    blocks = sorted(blocks_for_piece(train_blocks, pid))
    print(f"piece {pid} train blocks: {blocks[:3]}", flush=True)
    r, c = blocks[0]
    i, j = r * P, c * P

    X, Y = load_piece_arrays(cfg.pieces_dir, pid, cfg.input_vars, cfg.target_prefix, cfg.delta_index)
    fi = X.shape[0] // 2  # a mid-piece frame

    cond_patch = torch.from_numpy(X[fi:fi + 1, :, i:i + P, j:j + P]).to(device)
    truth_patch = Y[fi, 0, i:i + P, j:j + P]

    for noise_scale in [1.0, 0.5, 0.0]:
        x0 = diffusion.ddpm_sample(model, cond_patch, noise_scale=noise_scale)
        rain_pred = torch.clamp(torch.expm1(x0 * scale), min=0.0).cpu().numpy()[0, 0]
        print(f"\n--- full ancestral DDPM (all {cfg.diffusion_timesteps} steps), noise_scale={noise_scale} ---")
        print("truth: min/mean/max =", truth_patch.min(), truth_patch.mean(), truth_patch.max())
        print("pred:  min/mean/max =", rain_pred.min(), rain_pred.mean(), rain_pred.max())
        print("RMSE:", float(np.sqrt(np.mean((truth_patch - rain_pred) ** 2))))
        print("R2:  ", r2(truth_patch.ravel(), rain_pred.ravel()))

    # also sanity-check: what does q_sample -> immediate one-step denoise
    # look like at t=0 (should be nearly perfect, since almost no noise)?
    x0_true = torch.from_numpy(np.log1p(truth_patch)[None, None] / scale).float().to(device)
    t0 = torch.zeros(1, dtype=torch.long, device=device)
    noise = torch.randn_like(x0_true)
    xt0 = diffusion.q_sample(x0_true, t0, noise)
    with torch.no_grad():
        eps_pred = model(xt0, cond_patch, t0)
    x0_recon = (xt0 - diffusion.sqrt_one_minus_alphas_cumprod[0] * eps_pred) / diffusion.sqrt_alphas_cumprod[0]
    rain_recon = torch.clamp(torch.expm1(x0_recon * scale), min=0.0).detach().cpu().numpy()[0, 0]
    print("\n--- single-step reconstruction at t=0 (should be near-perfect) ---")
    print("RMSE:", float(np.sqrt(np.mean((truth_patch - rain_recon) ** 2))))
    print("R2:  ", r2(truth_patch.ravel(), rain_recon.ravel()))


if __name__ == "__main__":
    main()
