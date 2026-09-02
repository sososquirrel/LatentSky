"""Datasets and patch loaders for piece-based rainfall data
(recovered from _archive/latentsky/data.py, 2026-09-02).

Cherry-picked from tzhou/LatentSky/util_LS.py — trimmed to what the trainer
actually uses (NumpyArrayDataset, GridPatchSubsetDataset, load_piece_arrays,
build_patch_loader_from_arrays).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class NumpyArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


class GridPatchSubsetDataset(Dataset):
    """Tile each (C,H,W) frame into P×P patches, striding by `stride`.

    `stride=None` (default) reproduces the original non-overlapping tiling
    exactly: `stride = P`, so `nrows = (H-P)//P + 1 == H//P` whenever H is a
    multiple of P (true for every piece on disk today). Passing `stride < P`
    gives overlapping patches instead, so training exposes more boundary-
    region spatial context without changing inference (which never patches).
    """

    def __init__(self, subset: Subset, patch_size: int, stride: int | None = None):
        self.dataset = subset.dataset
        self.indices = list(subset.indices)
        self.P = patch_size
        self.S = stride if stride is not None else patch_size
        x0 = self.dataset[0][0]
        _, H, W = x0.shape
        self.nrows = (H - self.P) // self.S + 1
        self.ncols = (W - self.P) // self.S + 1
        self.T = len(self.indices)

    def __len__(self):
        return self.T * self.nrows * self.ncols

    def __getitem__(self, idx):
        ppf = self.nrows * self.ncols
        frame_idx = idx // ppf
        patch_number = idx % ppf
        row, col = divmod(patch_number, self.ncols)
        i = row * self.S
        j = col * self.S
        x_full, y_full = self.dataset[self.indices[frame_idx]]
        return (
            x_full[:, i : i + self.P, j : j + self.P],
            y_full[:, i : i + self.P, j : j + self.P],
        )


class FilteredGridPatchDataset(Dataset):
    """Non-overlapping P×P tiling like GridPatchSubsetDataset(stride=None),
    but restricted to a specific set of (row, col) blocks -- used for the
    patch-level train/val/test split (see latentsky/patch_split.py), where
    a single piece's patches are divided across train/val/test rather than
    the whole piece belonging to one split.
    """

    def __init__(self, subset: Subset, patch_size: int, allowed_blocks: set[tuple[int, int]]):
        self.dataset = subset.dataset
        self.indices = list(subset.indices)
        self.P = patch_size
        x0 = self.dataset[0][0]
        _, H, W = x0.shape
        self.nrows = H // self.P
        self.ncols = W // self.P
        self.T = len(self.indices)
        self.blocks = sorted(b for b in allowed_blocks if 0 <= b[0] < self.nrows and 0 <= b[1] < self.ncols)

    def __len__(self):
        return self.T * len(self.blocks)

    def __getitem__(self, idx):
        n_blocks = len(self.blocks)
        frame_idx = idx // n_blocks
        row, col = self.blocks[idx % n_blocks]
        i, j = row * self.P, col * self.P
        x_full, y_full = self.dataset[self.indices[frame_idx]]
        return (
            x_full[:, i : i + self.P, j : j + self.P],
            y_full[:, i : i + self.P, j : j + self.P],
        )


def build_filtered_patch_loader(
    X: np.ndarray,
    Y: np.ndarray,
    patch_size: int,
    batch_size: int,
    shuffle: bool,
    allowed_blocks: set[tuple[int, int]],
    num_workers: int = 4,
) -> DataLoader | None:
    """Like build_patch_loader, but restricted to `allowed_blocks`
    (row,col) pairs. Returns None if allowed_blocks is empty (this piece
    contributes nothing to the requested split)."""
    if not allowed_blocks:
        return None
    ds = NumpyArrayDataset(X, Y)
    subset = Subset(ds, list(range(len(ds))))
    patch_ds = FilteredGridPatchDataset(subset, patch_size, allowed_blocks)
    if len(patch_ds) == 0:
        return None
    return DataLoader(
        patch_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_piece_arrays(
    pieces_dir: str,
    piece_id: int,
    input_var_prefixes: Sequence[str],
    target_prefix: str,
    delta_index: int | None = None,
    n_frames: int | None = None,
    keep_frames: int | None = None,
    return_day_idx: bool = False,
    return_persisted: bool = False,
):
    """Load all input vars + target for one piece. Returns (X, Y) as float32.

    X: (T, C, H, W). Y: (T, 1, H, W).

    Uses mmap so requesting `n_frames` reads only those frames from disk —
    critical for piece files that can be tens of GB. If n_frames is None,
    loads the full piece (used by training).

    `keep_frames`: load the *whole* daily piece, apply the time-slip, then
    keep that many output frames evenly spread across the day (for diurnal
    sampling). Overrides n_frames. With `return_day_idx=True` also returns
    (day_idx, frames_per_day): day_idx[k] is the within-day frame number of
    each kept target frame, frames_per_day the raw piece length (96 = 24 h).

    `return_persisted=True` additionally returns `persisted`: the raw
    target field at the SAME time index as X (i.e. rain_total(t=0), before
    the delta-slip) — "freeze the initial condition forward" for each
    sample. Same shape/subsampling as Y, but aligned index-for-index with
    X (not with Y, which is the future value). Used for residual-target
    training: target - persisted is what the model actually learns.
    """
    if keep_frames is not None:
        n_frames = None  # force full-piece load; subsample after the slip
    # Account for the time-slip: X[:T-delta] -> Y[delta:].
    raw_n = n_frames + delta_index if (n_frames is not None and delta_index) else n_frames

    channels = []
    for prefix in input_var_prefixes:
        path = os.path.join(pieces_dir, f"{prefix}_piece{piece_id:02d}.npy")
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array for {path}, got {arr.shape}")
        if raw_n is not None:
            arr = np.asarray(arr[:raw_n], dtype=np.float32)
        else:
            arr = np.asarray(arr, dtype=np.float32)
        channels.append(arr)
    X = np.stack(channels, axis=1)

    rain_path = os.path.join(pieces_dir, f"{target_prefix}_piece{piece_id:02d}.npy")
    Y_raw = np.load(rain_path, mmap_mode="r")
    if Y_raw.ndim != 3:
        raise ValueError(f"Expected 3D target array, got {Y_raw.shape}")
    if raw_n is not None:
        Y_raw = np.asarray(Y_raw[:raw_n], dtype=np.float32)
    else:
        Y_raw = np.asarray(Y_raw, dtype=np.float32)
    Y = Y_raw[:, None, :, :]
    persisted = Y_raw[:, None, :, :].copy() if return_persisted else None

    frames_per_day = Y_raw.shape[0]
    delta = delta_index if (delta_index is not None and delta_index > 0) else 0
    if delta:
        X = X[:-delta, ...]
        Y = Y[delta:, ...]
        if return_persisted:
            persisted = persisted[:-delta, ...]  # aligned with X's index, not Y's

    # Post-slip output frame j corresponds to raw target frame (j + delta);
    # within a daily (96-frame) piece that is its time-of-day index.
    day_idx = np.arange(X.shape[0]) + delta

    if keep_frames is not None and X.shape[0] > keep_frames:
        sel = np.unique(np.linspace(0, X.shape[0] - 1, keep_frames).astype(int))
        X = X[sel, ...]
        Y = Y[sel, ...]
        day_idx = day_idx[sel]
        if return_persisted:
            persisted = persisted[sel, ...]

    if return_day_idx and return_persisted:
        return X, Y, persisted, day_idx, frames_per_day
    if return_day_idx:
        return X, Y, day_idx, frames_per_day
    if return_persisted:
        return X, Y, persisted
    return X, Y


def build_patch_loader(
    X: np.ndarray,
    Y: np.ndarray,
    patch_size: int,
    batch_size: int,
    shuffle: bool,
    stride: int | None = None,
    num_workers: int = 4,
) -> DataLoader:
    ds = NumpyArrayDataset(X, Y)
    subset = Subset(ds, list(range(len(ds))))
    patch_ds = GridPatchSubsetDataset(subset, patch_size, stride)
    return DataLoader(
        patch_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
