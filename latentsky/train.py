"""Config-driven trainer.

Usage:
    python -m latentsky.train --config configs/mseuv_lead3h.yaml

Writes to runs/<config.name>/:
    manifest.json    config + loss history + best val + git SHA
    checkpoint.pt    best-val-loss weights
    train.log        plain-text training log
    loss.png         train/val curves

Refactored from tzhou/LatentSky/260206_newtrain.py.
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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from latentsky.config import ExperimentConfig
from latentsky.data import build_filtered_patch_loader, build_patch_loader, load_piece_arrays
from latentsky.losses import combined_loss, scale_kernel_sizes
from latentsky.masking import random_block_mask_channel
from latentsky.models import UNet2D, UNet2DTwoHead
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.registry import REPO_ROOT, classify_new_run


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


def _make_train_groups(piece_ids: list[int], group_size: int, seed: int) -> list[list[int]]:
    ids = list(piece_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return [ids[i : i + group_size] for i in range(0, len(ids), group_size)]


def _load_xy(cfg: ExperimentConfig, pid: int):
    """(X, Y) ready for build_patch_loader. For residual configs, Y is
    2-channel: [residual_target, or_mask] where or_mask marks pixels wet
    in truth OR wet in the persisted (t=0) field — the residual analogue
    of the plain `y > 0` masking used for direct-target training."""
    if cfg.residual:
        X, Y, P = load_piece_arrays(
            cfg.pieces_dir, pid, cfg.input_vars, cfg.target_prefix, cfg.delta_index,
            return_persisted=True,
        )
        resid = Y - P
        mask = ((Y > 0) | (P > 0)).astype(np.float32)
        Y = np.concatenate([resid, mask], axis=1)
    else:
        X, Y = load_piece_arrays(
            cfg.pieces_dir, pid, cfg.input_vars, cfg.target_prefix, cfg.delta_index
        )
    return X, Y


def _split_target(y: torch.Tensor, residual: bool):
    if residual:
        return y[:, 0:1], y[:, 1:2] > 0.5
    return y, y > 0


def _maybe_mask(x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    """Pretrain-only: block-mask `cfg.pretrain_mask_channel` if set (see
    latentsky/masking.py). No-op (returns x unchanged) otherwise."""
    if cfg.pretrain_mask_channel is None:
        return x
    ch_idx = cfg.input_vars.index(cfg.pretrain_mask_channel)
    return random_block_mask_channel(x, ch_idx, cfg.pretrain_mask_frac, cfg.pretrain_mask_block)


def _eval(model, piece_ids, criterion, device, cfg: ExperimentConfig,
          kernel_sizes: list[int] | None = None) -> dict[str, float]:
    """Returns {"total": avg_loss} for the legacy (single-head) path, or
    {"total","mse","bce","fss"} averages for the two-head path."""
    model.eval()
    if cfg.two_head:
        totals = {"total": 0.0, "mse": 0.0, "bce": 0.0, "fss": 0.0}
        total_batches = 0
        with torch.no_grad():
            for pid in piece_ids:
                X_v, Y_v = _load_xy(cfg, pid)
                loader = build_patch_loader(X_v, Y_v, cfg.patch_size, cfg.batch_size,
                                             shuffle=False, stride=cfg.patch_stride)
                for x, y in loader:
                    if device.type == "cuda":
                        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                        y = y.to(device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        x = x.to(device)
                        y = y.to(device)
                    x = _maybe_mask(x, cfg)
                    y_target, mask = _split_target(y, cfg.residual)
                    rain_pred, wetdry_logit = model(x)
                    parts = combined_loss(rain_pred, wetdry_logit, y_target, mask, kernel_sizes,
                                           cfg.bce_weight, cfg.fss_weight,
                                           cfg.fss_threshold, cfg.fss_sharpness)
                    for k in totals:
                        totals[k] += float(parts[k].item())
                    total_batches += 1
                del X_v, Y_v, loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return {k: v / max(total_batches, 1) for k, v in totals.items()}

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for pid in piece_ids:
            X_v, Y_v = _load_xy(cfg, pid)
            loader = build_patch_loader(X_v, Y_v, cfg.patch_size, cfg.batch_size,
                                         shuffle=False, stride=cfg.patch_stride)
            for x, y in loader:
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                    y = y.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    x = x.to(device)
                    y = y.to(device)
                x = _maybe_mask(x, cfg)
                y_target, mask = _split_target(y, cfg.residual)
                if not mask.any():
                    continue
                out = model(x)
                loss = criterion(out[mask], y_target[mask])
                total_loss += float(loss.item())
                total_batches += 1
            del X_v, Y_v, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"total": total_loss / max(total_batches, 1)}


def _to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device):
    if device.type == "cuda":
        return (x.to(device, non_blocking=True, memory_format=torch.channels_last),
                y.to(device, non_blocking=True, memory_format=torch.channels_last))
    return x.to(device), y.to(device)


def _run_epoch_patch_mode(model, optimizer, criterion, kernel_sizes, device,
                           cfg: ExperimentConfig, train_blocks, val_blocks, epoch: int,
                           log) -> tuple[dict[str, float], dict[str, float]]:
    """One epoch under `split_mode="patch"`: every piece is loaded from
    disk exactly once and contributes both train patches and val patches
    (whichever (row,col) blocks were assigned to it) -- unlike the
    piece-level split, a given piece is not exclusively train or val/test.
    """
    piece_order = list(range(cfg.n_total_pieces))
    rng = random.Random(cfg.seed + epoch)
    rng.shuffle(piece_order)

    train_totals = {"total": 0.0, "mse": 0.0, "bce": 0.0, "fss": 0.0}
    val_totals = {"total": 0.0, "mse": 0.0, "bce": 0.0, "fss": 0.0}
    train_batches = 0
    val_batches = 0

    for pid in piece_order:
        t_blocks = blocks_for_piece(train_blocks, pid)
        v_blocks = blocks_for_piece(val_blocks, pid)
        if not t_blocks and not v_blocks:
            continue
        log(f"  [Patch-split] piece {pid}: {len(t_blocks)} train blocks, {len(v_blocks)} val blocks")
        X, Y = _load_xy(cfg, pid)

        if t_blocks:
            model.train()
            loader = build_filtered_patch_loader(X, Y, cfg.patch_size, cfg.batch_size,
                                                  shuffle=True, allowed_blocks=t_blocks)
            if loader is not None:
                for x, y in loader:
                    x, y = _to_device(x, y, device)
                    x = _maybe_mask(x, cfg)
                    y_target, mask = _split_target(y, cfg.residual)
                    if cfg.two_head:
                        optimizer.zero_grad(set_to_none=True)
                        rain_pred, wetdry_logit = model(x)
                        parts = combined_loss(rain_pred, wetdry_logit, y_target, mask, kernel_sizes,
                                               cfg.bce_weight, cfg.fss_weight,
                                               cfg.fss_threshold, cfg.fss_sharpness)
                        loss = parts["total"]
                        loss.backward()
                        optimizer.step()
                        train_totals["total"] += float(loss.item())
                        for k in ("mse", "bce", "fss"):
                            train_totals[k] += float(parts[k].item())
                        train_batches += 1
                    else:
                        if not mask.any():
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        out = model(x)
                        loss = criterion(out[mask], y_target[mask])
                        loss.backward()
                        optimizer.step()
                        train_totals["total"] += float(loss.item())
                        train_batches += 1
                del loader

        if v_blocks:
            model.eval()
            loader = build_filtered_patch_loader(X, Y, cfg.patch_size, cfg.batch_size,
                                                  shuffle=False, allowed_blocks=v_blocks)
            if loader is not None:
                with torch.no_grad():
                    for x, y in loader:
                        x, y = _to_device(x, y, device)
                        x = _maybe_mask(x, cfg)
                        y_target, mask = _split_target(y, cfg.residual)
                        if cfg.two_head:
                            rain_pred, wetdry_logit = model(x)
                            parts = combined_loss(rain_pred, wetdry_logit, y_target, mask, kernel_sizes,
                                                   cfg.bce_weight, cfg.fss_weight,
                                                   cfg.fss_threshold, cfg.fss_sharpness)
                            for k in val_totals:
                                val_totals[k] += float(parts[k].item())
                            val_batches += 1
                        else:
                            if not mask.any():
                                continue
                            out = model(x)
                            loss = criterion(out[mask], y_target[mask])
                            val_totals["total"] += float(loss.item())
                            val_batches += 1
                del loader

        del X, Y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_avg = {k: v / max(train_batches, 1) for k, v in train_totals.items()}
    val_avg = {k: v / max(val_batches, 1) for k, v in val_totals.items()}
    return train_avg, val_avg


def train(cfg: ExperimentConfig) -> Path:
    import os
    runs_base = Path(os.environ.get("LATENTSKY_RUNS_DIR", str(REPO_ROOT / "runs")))
    run_dir = runs_base / classify_new_run(cfg, kind="unet") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_f = open(log_path, "w")

    def log(msg: str) -> None:
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    _seed_everything(cfg.seed)

    n_gpus = torch.cuda.device_count()
    log(f"GPUs: {n_gpus}")

    if cfg.two_head:
        model = UNet2DTwoHead(
            in_ch=cfg.in_channels,
            base_ch=cfg.base_ch,
            depth=cfg.depth,
            dropout_p=cfg.dropout_p,
            out_activation=cfg.out_activation,
            olr_channel_idx=cfg.input_vars.index(cfg.olr_var_name),
        )
    else:
        model = UNet2D(
            in_ch=cfg.in_channels,
            out_ch=1,
            base_ch=cfg.base_ch,
            depth=cfg.depth,
            dropout_p=cfg.dropout_p,
            out_activation=cfg.out_activation,
        )

    if cfg.pretrain_checkpoint:
        state = torch.load(cfg.pretrain_checkpoint, map_location="cpu")
        model.load_state_dict(state)
        log(f"loaded pretrained weights from {cfg.pretrain_checkpoint}")

    if n_gpus > 1:
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    if cfg.two_head:
        criterion = None
        kernel_sizes = scale_kernel_sizes(max_kernel=cfg.fss_max_kernel_px)
        log(f"two_head=True  FSS kernel sizes (px): {kernel_sizes}")
    else:
        criterion = nn.MSELoss()
        kernel_sizes = None
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)

    if cfg.split_mode == "patch":
        stride = cfg.patch_stride if cfg.patch_stride is not None else cfg.patch_size
        if stride != cfg.patch_size:
            raise ValueError(
                f"split_mode='patch' requires non-overlapping patches "
                f"(patch_stride == patch_size or None), got stride={stride} "
                f"!= patch_size={cfg.patch_size} -- overlapping patches would "
                f"leak pixels across the train/val/test split."
            )
        probe_path = f"{cfg.pieces_dir}/{cfg.input_vars[0]}_piece00.npy"
        import numpy as _np
        _, H, W = _np.load(probe_path, mmap_mode="r").shape
        nrows, ncols = compute_patch_grid(H, W, cfg.patch_size)
        train_blocks, val_blocks, test_blocks = make_patch_split(
            cfg.n_total_pieces, nrows, ncols, cfg.patch_split_seed,
            cfg.patch_split_train_frac, cfg.patch_split_val_frac,
        )
        log(f"split_mode=patch  grid={nrows}x{ncols} blocks/piece  "
            f"train={len(train_blocks)} val={len(val_blocks)} test={len(test_blocks)} "
            f"(test blocks held out, unused by this training run)")

    best_val = float("inf")
    patience_ctr = 0
    train_losses, val_losses = [], []
    train_components_hist, val_components_hist = [], []
    ckpt_path = run_dir / "checkpoint.pt"

    log(f"=== Training {cfg.name} ===")
    for epoch in range(cfg.num_epochs):
        log(f"\nEpoch {epoch+1}/{cfg.num_epochs}  {datetime.now().isoformat(timespec='seconds')}")

        if cfg.split_mode == "patch":
            train_avg, val_avg = _run_epoch_patch_mode(
                model, optimizer, criterion, kernel_sizes, device, cfg,
                train_blocks, val_blocks, epoch, log,
            )
            avg_train = train_avg["total"]
            avg_val = val_avg["total"]
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            if cfg.two_head:
                train_components_hist.append({k: v for k, v in train_avg.items() if k != "total"})
                val_components_hist.append({k: v for k, v in val_avg.items() if k != "total"})
            scheduler.step()

        else:
            model.train()
            total_train_loss = 0.0
            total_batches = 0
            train_components = {"mse": 0.0, "bce": 0.0, "fss": 0.0}

            groups = _make_train_groups(cfg.train_piece_ids, cfg.train_pieces_per_group, cfg.seed + epoch)
            for gi, group in enumerate(groups):
                log(f"  [Train] Group {gi+1}/{len(groups)} pieces={group}")
                X_list, Y_list = [], []
                for pid in group:
                    X, Y = _load_xy(cfg, pid)
                    X_list.append(X)
                    Y_list.append(Y)
                Xg = np.concatenate(X_list, axis=0)
                Yg = np.concatenate(Y_list, axis=0)
                del X_list, Y_list
                gc.collect()

                loader = build_patch_loader(Xg, Yg, cfg.patch_size, cfg.batch_size,
                                             shuffle=True, stride=cfg.patch_stride)
                for x, y in loader:
                    if device.type == "cuda":
                        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                        y = y.to(device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        x = x.to(device)
                        y = y.to(device)
                    x = _maybe_mask(x, cfg)
                    y_target, mask = _split_target(y, cfg.residual)
                    if cfg.two_head:
                        optimizer.zero_grad(set_to_none=True)
                        rain_pred, wetdry_logit = model(x)
                        parts = combined_loss(rain_pred, wetdry_logit, y_target, mask, kernel_sizes,
                                               cfg.bce_weight, cfg.fss_weight,
                                               cfg.fss_threshold, cfg.fss_sharpness)
                        loss = parts["total"]
                        loss.backward()
                        optimizer.step()
                        total_train_loss += float(loss.item())
                        for k in ("mse", "bce", "fss"):
                            train_components[k] += float(parts[k].item())
                        total_batches += 1
                    else:
                        if not mask.any():
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        out = model(x)
                        loss = criterion(out[mask], y_target[mask])
                        loss.backward()
                        optimizer.step()
                        total_train_loss += float(loss.item())
                        total_batches += 1

                del Xg, Yg, loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train = total_train_loss / max(total_batches, 1)
            val_result = _eval(model, cfg.val_piece_ids, criterion, device, cfg, kernel_sizes)
            avg_val = val_result["total"]
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            if cfg.two_head:
                train_components_hist.append({k: v / max(total_batches, 1) for k, v in train_components.items()})
                val_components_hist.append({k: v for k, v in val_result.items() if k != "total"})
            scheduler.step()

        log(f"  -> lr={scheduler.get_last_lr()[0]:.3e} "
            f"train={avg_train:.6f} val={avg_val:.6f}")
        if cfg.two_head:
            log(f"     train components: {train_components_hist[-1]}")
            log(f"     val components:   {val_components_hist[-1]}")

        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            log("  ** val improved, saving checkpoint")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, ckpt_path)
        else:
            patience_ctr += 1
            log(f"  no improvement ({patience_ctr}/{cfg.patience})")
            if patience_ctr >= cfg.patience:
                log("  early stop")
                break

    # Manifest
    manifest = {
        "name": cfg.name,
        "git_sha": _git_sha(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "loss_history": {"train": train_losses, "val": val_losses},
        "loss_components_history": {"train": train_components_hist, "val": val_components_hist},
        "best_val_loss": best_val,
        "epochs_completed": len(train_losses),
        "checkpoint_relpath": "checkpoint.pt",
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    if cfg.two_head:
        plt.ylabel("combined loss (mse + bce_weight*bce + fss_weight*fss)")
    else:
        plt.ylabel("MSE (residual, wet-or-persisted mask)" if cfg.residual else "MSE (masked rain>0)")
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
