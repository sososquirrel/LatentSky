"""Layout of the DYAMOND MPAS dataset as the models see it (no torch import,
so plotting and WB2 scripts can use it).

One place for what every script used to restate: where the piece files and
the trained runs live, the ten input channels, the 0.25-degree tropical grid
and its 128-pixel blocks, the log-rain transform of the flow, the object
threshold, and the example case shown in the paper (piece 8, frame 64,
block row 1 / column 4, West Pacific warm pool).

Paths default to the GLADE scratch layout and can be redirected with the
LATENTSKY_PIECES / LATENTSKY_RUNS environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PIECES = Path(os.environ.get("LATENTSKY_PIECES",
                             "/glade/derecho/scratch/sabramian/latentsky_pieces"))
RUNS = Path(os.environ.get("LATENTSKY_RUNS",
                           "/glade/derecho/scratch/sabramian/LatentSky_runs"))

# ten input channels at time t; the current rain is last (persistence is nested)
INPUT_VARS = [
    "precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
    "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
    "umeridional_850hPa", "rain_log1p",
]
TARGET_VAR = "rain_total"

H, W = 320, 1440           # 40S-40N at 0.25 deg
P = 128                     # block (patch) side, pixels
FRAMES_PER_HOUR = 4         # 15-min output

LOG_RAIN_SCALE = 2.0        # flow space: z = ln(1 + R) / LOG_RAIN_SCALE
WET_THRESHOLD_MM = 0.1      # object threshold, mm per 15 min
IOU_MIN = 0.1               # matching acceptance

EXAMPLE_PIECE, EXAMPLE_FRAME, EXAMPLE_BLOCK = 8, 64, (1, 4)


def piece_path(var: str, piece: int) -> Path:
    """Path of one variable's piece file."""
    return PIECES / f"{var}_piece{piece:02d}.npy"


def load_frame(var: str, frame: int = EXAMPLE_FRAME, piece: int = EXAMPLE_PIECE) -> np.ndarray:
    """One full (H, W) frame of a variable, as float32 (memory-mapped read)."""
    return np.asarray(np.load(piece_path(var, piece), mmap_mode="r")[frame], dtype=np.float32)


def block_slice(block: tuple[int, int] = EXAMPLE_BLOCK, p: int = P):
    """Numpy index of one block (row, col) of the block grid."""
    r, c = block
    return np.s_[r * p:(r + 1) * p, c * p:(c + 1) * p]


def lead_frames(lead_h: float) -> int:
    """Number of 15-min frames between t and t + lead."""
    n = lead_h * FRAMES_PER_HOUR
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"lead {lead_h} h is not a multiple of 15 min")
    return int(round(n))


def rain_from_z(z, zmax: float = 2.5):
    """Rain (mm per 15 min) from the flow's log space, z = ln(1 + R) / LOG_RAIN_SCALE;
    z is clipped to [0, zmax] first, as in every figure of the paper."""
    return np.clip(np.expm1(LOG_RAIN_SCALE * np.clip(z, 0, zmax)), 0, None)


def tiles(x: np.ndarray, p: int = P) -> np.ndarray:
    """(C, H, W) frame -> (nr * nc, C, p, p) non-overlapping blocks, row-major;
    the remainder rows and columns are left out, as in the data split."""
    c, h, w = x.shape
    nr, nc = h // p, w // p
    t = x[:, :nr * p, :nc * p].reshape(c, nr, p, nc, p)
    return np.ascontiguousarray(t.transpose(1, 3, 0, 2, 4).reshape(nr * nc, c, p, p))


def mosaic(t: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """(nr * nc, 1, p, p) blocks -> (nr * p, nc * p) field; inverse of `tiles`."""
    p = t.shape[-1]
    return t.reshape(nr, nc, p, p).transpose(0, 2, 1, 3).reshape(nr * p, nc * p)
