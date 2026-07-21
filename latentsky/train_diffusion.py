"""Conditional-DDPM trainer for a diffusion-based alternative to the
regression LatentSky models.

Usage:
    python -m latentsky.train_diffusion --config configs/diffusion_lead3h.yaml

Kept as a separate, self-contained script rather than folded into
latentsky.train -- diffusion training (random timestep per batch, no wet/dry
masking, whole-field loss) is different enough from the regression training
loop's branching (two_head/residual/pretrain-mask) that sharing the function
would add more risk than it saves code.

Uses the same patch-level split as the patch-split experiment (see
latentsky/patch_split.py) -- that was the best-performing split from
earlier training runs, so it's the natural default here too.

Target transform: x0 = log1p(rain_total) / cfg.diffusion_target_scale,
roughly unit range (rain values are small, mm/15min-ish). Inverting this
(expm1(x0 * scale)) is needed for a sampler, which is NOT implemented here
-- this script covers training only. See latentsky/diffusion.py.

Writes to runs/<config.name>/: manifest.json, checkpoint.pt, train.log,
loss.png -- same layout as latentsky.train.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from latentsky.config import ExperimentConfig
from latentsky.data import build_filtered_patch_loader, load_piece_arrays
from latentsky.diffusion import GaussianDiffusion
from latentsky.models import ConditionalDiffusionUNet
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.registry import REPO_ROOT


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_xy(cfg: ExperimentConfig, pid: int):
    X, Y = load_piece_arrays(
        cfg.pieces_dir, pid, cfg.input_vars, cfg.target_prefix, cfg.delta_index
    )
    return X, Y


def _run_epoch(
    model, optimizer, diffusion: GaussianDiffusion, device, cfg: ExperimentConfig,
    train_blocks, val_blocks, epoch: int, log,
) -> tuple[float, float]:
    piece_order = list(range(cfg.n_total_pieces))
    rng = random.Random(cfg.seed + epoch)
    rng.shuffle(piece_order)

    train_loss_sum, train_batches = 0.0, 0
    val_loss_sum, val_batches = 0.0, 0
    scale = cfg.diffusion_target_scale

    for pid in piece_order:
        t_blocks = blocks_for_piece(train_blocks, pid)
        v_blocks = blocks_for_piece(val_blocks, pid)
        if not t_blocks and not v_blocks:
            continue
        X, Y = _load_xy(cfg, pid)

        if t_blocks:
            model.train()
            loader = build_filtered_patch_loader(X, Y, cfg.patch_size, cfg.batch_size,
                                                  shuffle=True, allowed_blocks=t_blocks)
            if loader is not None:
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    x0 = torch.log1p(y) / scale
                    tstep = diffusion.sample_timesteps(x0.shape[0], device)
                    noise = torch.randn_like(x0)
                    x_t = diffusion.q_sample(x0, tstep, noise)
                    optimizer.zero_grad(set_to_none=True)
                    pred_noise = model(x_t, x, tstep)
                    loss = F.mse_loss(pred_noise, noise)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.item())
                    train_batches += 1
                del loader

        if v_blocks:
            model.eval()
            loader = build_filtered_patch_loader(X, Y, cfg.patch_size, cfg.batch_size,
                                                  shuffle=False, allowed_blocks=v_blocks)
            if loader is not None:
                with torch.no_grad():
                    for x, y in loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        x0 = torch.log1p(y) / scale
                        tstep = diffusion.sample_timesteps(x0.shape[0], device)
                        noise = torch.randn_like(x0)
                        x_t = diffusion.q_sample(x0, tstep, noise)
                        pred_noise = model(x_t, x, tstep)
                        loss = F.mse_loss(pred_noise, noise)
                        val_loss_sum += float(loss.item())
                        val_batches += 1
                del loader

        del X, Y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (train_loss_sum / max(train_batches, 1), val_loss_sum / max(val_batches, 1))


def train(cfg: ExperimentConfig) -> Path:
    import os
    runs_base = Path(os.environ.get("LATENTSKY_RUNS_DIR", str(REPO_ROOT / "runs")))
    run_dir = runs_base / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "train.log", "w")

    def log(msg: str) -> None:
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    stride = cfg.patch_stride if cfg.patch_stride is not None else cfg.patch_size
    if stride != cfg.patch_size:
        raise ValueError("diffusion training requires non-overlapping patches (patch_stride == patch_size)")

    probe_path = f"{cfg.pieces_dir}/{cfg.input_vars[0]}_piece00.npy"
    _, H, W = np.load(probe_path, mmap_mode="r").shape
    nrows, ncols = compute_patch_grid(H, W, cfg.patch_size)
    train_blocks, val_blocks, test_blocks = make_patch_split(
        cfg.n_total_pieces, nrows, ncols, cfg.patch_split_seed,
        cfg.patch_split_train_frac, cfg.patch_split_val_frac,
    )
    log(f"split: grid={nrows}x{ncols}  train={len(train_blocks)} val={len(val_blocks)} test={len(test_blocks)}")

    model = ConditionalDiffusionUNet(
        cond_ch=cfg.in_channels,
        base_ch=cfg.base_ch,
        depth=cfg.depth,
        dropout_p=cfg.dropout_p,
        time_emb_dim=cfg.diffusion_time_emb_dim,
    ).to(device)

    diffusion = GaussianDiffusion(cfg.diffusion_timesteps, cfg.diffusion_schedule, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)

    best_val = float("inf")
    patience_ctr = 0
    train_losses, val_losses = [], []
    ckpt_path = run_dir / "checkpoint.pt"

    log(f"=== Training {cfg.name} (diffusion) ===")
    for epoch in range(cfg.num_epochs):
        log(f"\nEpoch {epoch+1}/{cfg.num_epochs}  {datetime.now().isoformat(timespec='seconds')}")
        avg_train, avg_val = _run_epoch(
            model, optimizer, diffusion, device, cfg, train_blocks, val_blocks, epoch, log,
        )
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()
        log(f"  -> lr={scheduler.get_last_lr()[0]:.3e} train={avg_train:.6f} val={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            log("  ** val improved, saving checkpoint")
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            log(f"  no improvement ({patience_ctr}/{cfg.patience})")
            if patience_ctr >= cfg.patience:
                log("  early stop")
                break

    manifest = {
        "name": cfg.name,
        "git_sha": _git_sha(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "loss_history": {"train": train_losses, "val": val_losses},
        "best_val_loss": best_val,
        "epochs_completed": len(train_losses),
        "checkpoint_relpath": "checkpoint.pt",
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("epsilon-prediction MSE (DDPM loss)")
    plt.title(cfg.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", bbox_inches="tight")
    plt.close()

    log_f.close()
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
