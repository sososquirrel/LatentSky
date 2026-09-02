"""Persistence/climatology/random baselines, returning the same (X, Y,
Y_pred) shape as latentsky.inference.predict_piece and
latentsky.diffusion_inference.predict_piece_diffusion, so
latentsky.scale_analysis.system_table() scores them identically to a
trained model's prediction.

These have no config/checkpoint, so the entry points take pieces_dir/
lead_hours directly instead of a Run object -- there is no registered
"run" for a baseline.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from latentsky.scale_analysis import system_table

RAIN_PREFIX = "rain_total"


def load_rain(pieces_dir: str, piece_id: int) -> np.ndarray:
    path = os.path.join(pieces_dir, f"{RAIN_PREFIX}_piece{piece_id:02d}.npy")
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)


def compute_climatology_mean(pieces_dir: str, piece_ids) -> np.ndarray:
    """Mean rain field (H, W) over all frames of the given pieces."""
    total, n = None, 0
    for pid in piece_ids:
        Y = load_rain(pieces_dir, pid)
        total = Y.sum(axis=0) if total is None else total + Y.sum(axis=0)
        n += Y.shape[0]
    return total / n


def predict_piece_baseline(
    kind: str,
    pieces_dir: str,
    piece_id: int,
    lead_hours: float,
    max_frames: int = 16,
    rand_piece_id: int | None = None,
    clim_mean: np.ndarray | None = None,
):
    """Return (X, Y, Y_pred) for the first `max_frames` output frames of a
    piece. X is just the persisted rain field (1 channel, no met inputs --
    none of these baselines condition on anything).

    kind:
      "persistence" -- Y_pred = rain at the same time index as X ("nothing
        changes" between now and the lead time).
      "climatology" -- Y_pred = a precomputed mean field (`clim_mean`),
        broadcast to every frame.
      "random" -- Y_pred = the same-time-index rain field from a
        different, decorrelated piece (`rand_piece_id`).

    delta = lead_steps_15 + 1 = round(lead_hours*4) + 1, matching
    latentsky.config.ExperimentConfig.delta_index exactly, so a baseline
    and a trained run "at the same lead" are aligned to the identical
    truth frame.
    """
    delta = round(lead_hours * 4) + 1
    Y_raw = load_rain(pieces_dir, piece_id)
    raw_n = max_frames + delta
    Y_raw = Y_raw[:raw_n]
    T = Y_raw.shape[0] - delta
    H, W = Y_raw.shape[-2], Y_raw.shape[-1]
    if T <= 0:
        empty = np.zeros((0, 1, H, W), dtype=np.float32)
        return empty, empty, empty

    persisted = Y_raw[:T]
    truth = Y_raw[delta:delta + T]
    X = persisted[:, None, :, :].copy()
    Y = truth[:, None, :, :].copy()

    if kind == "persistence":
        Y_pred = persisted[:, None, :, :].copy()
    elif kind == "climatology":
        if clim_mean is None:
            raise ValueError("climatology baseline needs clim_mean")
        Y_pred = np.broadcast_to(clim_mean[None, None], Y.shape).copy()
    elif kind == "random":
        if rand_piece_id is None:
            raise ValueError("random baseline needs rand_piece_id")
        Yr_raw = load_rain(pieces_dir, rand_piece_id)[:T]
        Y_pred = Yr_raw[:, None, :, :].copy()
    else:
        raise ValueError(f"unknown baseline kind {kind!r}")

    return X, Y, Y_pred


def evaluate_baseline_systems(
    kind: str,
    pieces_dir: str,
    lead_hours: float,
    piece_ids,
    max_frames_per_piece: int = 16,
    threshold: float = 0.1,
    iou_min: float = 0.1,
    n_total_pieces: int = 40,
    rand_offset: int = 20,
    clim_mean: np.ndarray | None = None,
) -> pd.DataFrame:
    """Baseline-track equivalent of
    latentsky.inference.evaluate_run_systems /
    latentsky.diffusion_inference.evaluate_run_systems_diffusion: same
    output schema (system_table + a `piece` column), so it plugs into the
    same downstream figure scripts unmodified.

    `clim_mean`: precomputed climatology mean field, reused across leads
    by the caller -- it doesn't depend on lead, so recomputing it per lead
    would mean redundantly re-reading all `n_total_pieces` rain arrays.
    Computed on demand (once) if not passed and kind=="climatology".
    """
    if kind == "climatology" and clim_mean is None:
        clim_mean = compute_climatology_mean(pieces_dir, range(n_total_pieces))

    frames = []
    for pid in piece_ids:
        rand_pid = (pid + rand_offset) % n_total_pieces if kind == "random" else None
        X, Y, Yp = predict_piece_baseline(
            kind, pieces_dir, pid, lead_hours, max_frames=max_frames_per_piece,
            rand_piece_id=rand_pid, clim_mean=clim_mean,
        )
        if Y.shape[0] == 0:
            continue
        df = system_table(Y, Yp, threshold=threshold, iou_min=iou_min)
        if not df.empty:
            df.insert(0, "piece", pid)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
