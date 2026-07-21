"""Generates the leakage-safe train/val/test/buffer piece-ID split used by
configs/twohead_olr_fss_lead3h.yaml.

The 40 pieces are sequential days of one continuous DYAMOND run. The
project's original split (test=[2,12,22,32], val=[6,16,26,36], fixed
stride-10, no randomization) leaves every held-out piece's immediate
temporal neighbors in the training set -- a real autocorrelation leak given
weather persistence across day boundaries.

This script partitions the 40 pieces into 8 blocks of 5 consecutive days and
holds out one INTERIOR day per block (never a block edge, so both its
temporal neighbors are guaranteed to fall inside the same block) --
alternating test (even blocks) / val (odd blocks). The held-out piece's two
immediate neighbors become buffer pieces, excluded from training entirely.
This closes the leak by construction.

This is a documentation/reproducibility script, not a package function --
the split must never silently re-randomize between runs, so the generated
lists are frozen as literal `test_piece_ids`/`val_piece_ids`/
`buffer_piece_ids` in the YAML config, not recomputed at train time.

Run: python scripts/make_split.py
"""
from __future__ import annotations

import random

N_TOTAL = 40
BLOCK_SIZE = 5
SEED = 1919810  # reuses the repo's existing default ExperimentConfig.seed


def make_split(n_total: int = N_TOTAL, block_size: int = BLOCK_SIZE, seed: int = SEED):
    rng = random.Random(seed)
    n_blocks = n_total // block_size
    test_ids, val_ids, buffer_ids = [], [], []

    for b in range(n_blocks):
        start = b * block_size
        interior = list(range(start + 1, start + block_size - 1))  # never edges
        held = rng.choice(interior)
        (test_ids if b % 2 == 0 else val_ids).append(held)
        buffer_ids.extend([held - 1, held + 1])

    held_all = set(test_ids) | set(val_ids) | set(buffer_ids)
    train_ids = [p for p in range(n_total) if p not in held_all]
    return sorted(test_ids), sorted(val_ids), sorted(buffer_ids), sorted(train_ids)


if __name__ == "__main__":
    test_ids, val_ids, buffer_ids, train_ids = make_split()

    all_ids = test_ids + val_ids + buffer_ids + train_ids
    assert sorted(all_ids) == list(range(N_TOTAL)), "split does not partition range(40) exactly"
    for t in test_ids + val_ids:
        assert (t - 1) in buffer_ids and (t + 1) in buffer_ids, f"piece {t} not fully buffered"

    print(f"test_piece_ids:   {test_ids}")
    print(f"val_piece_ids:    {val_ids}")
    print(f"buffer_piece_ids: {buffer_ids}")
    print(f"train_piece_ids:  {train_ids}")
    print(f"\ncounts: test={len(test_ids)} val={len(val_ids)} "
          f"buffer={len(buffer_ids)} train={len(train_ids)} total={N_TOTAL}")
