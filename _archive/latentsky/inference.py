"""Run inference on a saved Run and compute standard metrics.

Used by analysis notebooks. Replaces the per-notebook `run_unet_reconstruction`
and `number_only_newM` helpers in tzhou's notebooks.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from latentsky.data import NumpyArrayDataset, load_piece_arrays
from latentsky.metrics import compute_metrics
from latentsky.scale_analysis import system_table
from latentsky.models import UNet2D, UNet2DTwoHead
from latentsky.registry import Run


InputTransform = Callable[[np.ndarray, Sequence[str]], np.ndarray]


def daily_mean_ablation(kill: Sequence[str]) -> InputTransform:
    """Return a transform that replaces each named input channel with its
    per-cell daily mean — strips the diurnal phase signal while preserving
    geographic structure. Pieces are daily (96 frames at 15 min), so axis 0
    of X is exactly one day."""
    kill = tuple(kill)

    def _apply(X: np.ndarray, input_vars: Sequence[str]) -> np.ndarray:
        X = X.copy()
        for name in kill:
            if name not in input_vars:
                continue
            c = list(input_vars).index(name)
            X[:, c] = X[:, c].mean(axis=0, keepdims=True)
        return X

    _apply.__name__ = f"daily_mean_ablation({','.join(kill)})"
    return _apply


def _load_model(run: Run, device: torch.device):
    """Builds UNet2DTwoHead when cfg.two_head, else the original UNet2D --
    same checkpoint-loading logic either way, just a different class/ctor."""
    cfg = run.config
    if getattr(cfg, "two_head", False):
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
    state = torch.load(run.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def _infer(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Returns the rain-intensity prediction only. For two-head models,
    model(x) returns (rain_pred, wetdry_logit) -- the wet/dry logit isn't
    needed by any of the downstream metrics/system_table code, which all
    expect a single continuous field, so it's discarded here."""
    ds = NumpyArrayDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            rain_pred = out[0] if isinstance(out, tuple) else out
            preds.append(rain_pred.cpu().numpy())
            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0)


def predict_piece(
    run: Run,
    piece_id: int,
    max_frames: int = 16,
    batch_size: int = 1,
    device: torch.device | None = None,
    model: UNet2D | None = None,
    keep_frames: int | None = None,
    input_transform: InputTransform | None = None,
):
    """Return (X, Y_true, Y_pred) for the first `max_frames` frames of a piece.

    Pass a pre-loaded `model` to skip the .pt re-load — useful when calling
    this in a tight loop over multiple pieces of the same run.

    Memory note: each piece is float32 at 720×1440 with up to 9 channels, so
    a full piece is ~tens of GB. This function evaluates only the first
    `max_frames` frames at batch_size=1 to avoid OOM on the A100.

    `keep_frames`: instead of the first `max_frames`, load the whole daily
    piece and keep that many frames spread across the 24 h day (diurnal
    sampling). In that case returns the 5-tuple
    (X, Y, Y_pred, day_idx, frames_per_day).
    """
    cfg = run.config
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P = None
    if keep_frames is not None:
        if cfg.residual:
            X, Y, P, day_idx, fpd = load_piece_arrays(
                cfg.pieces_dir, piece_id, cfg.input_vars, cfg.target_prefix,
                cfg.delta_index, keep_frames=keep_frames, return_day_idx=True,
                return_persisted=True,
            )
        else:
            X, Y, day_idx, fpd = load_piece_arrays(
                cfg.pieces_dir, piece_id, cfg.input_vars, cfg.target_prefix,
                cfg.delta_index, keep_frames=keep_frames, return_day_idx=True,
            )
    else:
        if cfg.residual:
            X, Y, P = load_piece_arrays(
                cfg.pieces_dir, piece_id, cfg.input_vars, cfg.target_prefix,
                cfg.delta_index, n_frames=max_frames, return_persisted=True,
            )
        else:
            X, Y = load_piece_arrays(
                cfg.pieces_dir, piece_id, cfg.input_vars, cfg.target_prefix,
                cfg.delta_index, n_frames=max_frames,
            )
    if model is None:
        model = _load_model(run, device)

    if input_transform is not None:
        X = input_transform(X, cfg.input_vars)

    Y_pred = _infer(model, X, Y, batch_size, device)
    if cfg.residual:
        # Model predicted a residual — add the persisted (t=0) field back
        # and clamp at 0 (rain can't be negative) to get a raw prediction
        # that every downstream metric/eval script can consume unchanged.
        Y_pred = np.clip(Y_pred + P, a_min=0.0, a_max=None)
    if keep_frames is not None:
        return X, Y, Y_pred, day_idx, fpd
    return X, Y, Y_pred


def evaluate_run(
    run: Run,
    piece_ids: Iterable[int] | None = None,
    max_frames_per_piece: int = 16,
    batch_size: int = 1,
) -> dict:
    """Compute metrics on the test pieces declared in the run's config.

    Defaults to 16 frames per test piece (~64 frames total over 4 pieces) —
    enough to get representative metrics without OOM. Bump
    `max_frames_per_piece` if you have memory headroom.
    """
    if piece_ids is None:
        piece_ids = run.config.test_piece_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run, device)

    all_truth, all_pred = [], []
    for pid in piece_ids:
        _, Y, Yp = predict_piece(
            run, pid, max_frames=max_frames_per_piece,
            batch_size=batch_size, device=device, model=model,
        )
        all_truth.append(Y)
        all_pred.append(Yp)
    truth = np.concatenate(all_truth, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    return compute_metrics(truth, pred)


def evaluate_run_systems(
    run: Run,
    piece_ids: Iterable[int] | None = None,
    max_frames_per_piece: int = 8,
    batch_size: int = 1,
    threshold: float = 0.1,
    iou_min: float = 0.1,
    keep_frames: int | None = None,
    input_transform: InputTransform | None = None,
) -> pd.DataFrame:
    """Scale-aware object table for a run's test pieces.

    Mirrors `evaluate_run`, but instead of bulk metrics returns the tidy
    per-system DataFrame from `latentsky.scale_analysis.system_table`
    (one row per precipitating system, with a `piece` column added).

    `keep_frames`: sample that many frames spread across each daily piece
    (instead of the first `max_frames_per_piece`) so the table carries a
    diurnal cycle — adds day_frame / tod_hour / lst_hour columns.
    """
    if piece_ids is None:
        piece_ids = run.config.test_piece_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run, device)

    frames = []
    for pid in piece_ids:
        if keep_frames is not None:
            _, Y, Yp, day_idx, fpd = predict_piece(
                run, pid, batch_size=batch_size, device=device, model=model,
                keep_frames=keep_frames, input_transform=input_transform,
            )
            df = system_table(Y, Yp, threshold=threshold, iou_min=iou_min,
                              frame_day_idx=day_idx, frames_per_day=fpd)
        else:
            _, Y, Yp = predict_piece(
                run, pid, max_frames=max_frames_per_piece,
                batch_size=batch_size, device=device, model=model,
                input_transform=input_transform,
            )
            df = system_table(Y, Yp, threshold=threshold, iou_min=iou_min)
        if not df.empty:
            df.insert(0, "piece", pid)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
