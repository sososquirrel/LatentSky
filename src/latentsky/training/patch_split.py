"""Patch-level train/val/test split (recovered from _archive 2026-09-02;
restrict_to_patch_test_blocks stays archived until the evaluation module
needs it)."""
from __future__ import annotations

import random


def compute_patch_grid(H: int, W: int, patch_size: int) -> tuple[int, int]:
    """Non-overlapping tiling grid (matches GridPatchSubsetDataset with
    stride=None, i.e. stride=patch_size)."""
    nrows = H // patch_size
    ncols = W // patch_size
    return nrows, ncols


def make_patch_split(
    n_pieces: int,
    nrows: int,
    ncols: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """Randomly assign every (piece, row, col) block to train/val/test.

    Returns (train_blocks, val_blocks, test_blocks), each a set of
    (piece_id, row, col) tuples. test_frac = 1 - train_frac - val_frac.
    """
    assert 0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1

    blocks = [(p, r, c) for p in range(n_pieces) for r in range(nrows) for c in range(ncols)]
    rng = random.Random(seed)
    rng.shuffle(blocks)

    n = len(blocks)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_blocks = set(blocks[:n_train])
    val_blocks = set(blocks[n_train : n_train + n_val])
    test_blocks = set(blocks[n_train + n_val :])
    return train_blocks, val_blocks, test_blocks


def blocks_for_piece(
    blocks: set[tuple[int, int, int]], piece_id: int
) -> set[tuple[int, int]]:
    """Filter a (piece,row,col) block set down to the (row,col) pairs
    belonging to one specific piece -- what FilteredGridPatchDataset needs."""
    return {(r, c) for (p, r, c) in blocks if p == piece_id}
