"""Fine-grained truth-vs-predicted contingency heatmap, one panel per
track, across ALL FIVE model tracks + THREE baselines at a fixed lead --
accumulated over each track's ACTUAL held-out test scope, not a single
piece/few frames:

  - regression (old split): the day-level held-out test pieces
    (run.config.test_piece_ids, e.g. [2,12,22,32] for mseuv_lead*).
  - regression (new/patch split), two-head, diffusion, and all three
    baselines: all 40 pieces, restricted per-piece to the actual
    patch-split test blocks (latentsky.patch_split), same scope as the
    object-detection scale-decomposition eval.

Accumulates a fixed-bin np.histogram2d per piece/frame batch and sums
across pieces (never holds the full test set in memory at once), so
this scales to the full domain x 40 pieces without OOM.

Cost note: diffusion inference is `t_start+1` sequential forward passes
per patch per frame per ensemble draw -- the same order of magnitude as
its earlier full-coverage scale-decomposition eval (~9h across all 6
leads total). Use --diffusion-pieces/--diffusion-max-frames to bound
diffusion's scope independently of the other (cheap) tracks if compute
is scarce; this is printed clearly in the figure title so a bounded
diffusion scope is never mistaken for full coverage.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR
from latentsky.patch_split import blocks_for_piece, compute_patch_grid, make_patch_split
from latentsky.plotting import build_contingency_table
from latentsky.predict_dispatch import predict_piece_track, tracks_for_lead
from latentsky.registry import load_run

N_TOTAL_PIECES = 40
MAX_FRAMES_PER_PIECE = 16
VMAX = 25.0  # mm/15min, shared bin range across every track/lead
NBINS = 30


def compute_test_blocks(H: int, W: int, patch_size: int = 128, seed: int = 1919810,
                         train_frac: float = 0.7, val_frac: float = 0.15):
    nrows, ncols = compute_patch_grid(H, W, patch_size)
    _, _, test_blocks = make_patch_split(N_TOTAL_PIECES, nrows, ncols, seed, train_frac, val_frac)
    return test_blocks


def patch_test_mask(H: int, W: int, patch_size: int, test_blocks, piece_id: int) -> np.ndarray:
    blocks = blocks_for_piece(test_blocks, piece_id)
    mask = np.zeros((H, W), dtype=bool)
    for r, c in blocks:
        i, j = r * patch_size, c * patch_size
        mask[i:i + patch_size, j:j + patch_size] = True
    return mask


def accumulate_track(track: dict, bins: np.ndarray, device, diffusion_pieces=None,
                      diffusion_max_frames=None) -> tuple[np.ndarray, int]:
    """Return (summed confusion matrix, total pixel count) for a track's
    full held-out test scope."""
    kind = track["kind"]
    conf_total = np.zeros((len(bins) - 1, len(bins) - 1), dtype=np.int64)
    n_pixels = 0

    if kind == "regression" and "onehead" not in track["run"] and "twohead" not in track["run"] \
            and "patchsplit" not in track["run"]:
        # old-split: day-level held-out test pieces, no patch masking needed.
        run = load_run(track["run"])
        piece_ids = run.config.test_piece_ids
        for pid in piece_ids:
            _, Y, Yp = predict_piece_track(track, pid, MAX_FRAMES_PER_PIECE, device=device)
            if Y.shape[0] == 0:
                continue
            _, conf, _ = build_contingency_table(Y.ravel(), Yp.ravel(), bins=bins)
            conf_total += conf
            n_pixels += Y.size
        return conf_total, n_pixels

    # patch-split tracks (regression new-split, two-head, diffusion) and all baselines:
    # full 40-piece scan, restricted per-piece to the actual test blocks.
    is_diffusion = kind == "diffusion"
    piece_ids = (diffusion_pieces if (is_diffusion and diffusion_pieces is not None)
                 else range(N_TOTAL_PIECES))
    max_frames = (diffusion_max_frames if (is_diffusion and diffusion_max_frames is not None)
                  else MAX_FRAMES_PER_PIECE)

    test_blocks = None
    for pid in piece_ids:
        try:
            _, Y, Yp = predict_piece_track(track, pid, max_frames, device=device)
        except Exception as e:
            print(f"  piece {pid} failed for {track['label']!r}: {e}")
            continue
        if Y.shape[0] == 0:
            continue
        H, W = Y.shape[-2], Y.shape[-1]
        if test_blocks is None:
            test_blocks = compute_test_blocks(H, W)
        mask = patch_test_mask(H, W, patch_size=128, test_blocks=test_blocks, piece_id=pid)
        yt = Y[:, 0][:, mask]
        yp = Yp[:, 0][:, mask]
        if yt.size == 0:
            continue
        _, conf, _ = build_contingency_table(yt.ravel(), yp.ravel(), bins=bins)
        conf_total += conf
        n_pixels += yt.size
    return conf_total, n_pixels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=float, required=True)
    p.add_argument("--nbins", type=int, default=NBINS)
    p.add_argument("--vmax", type=float, default=VMAX)
    p.add_argument("--diffusion-pieces", type=int, nargs="+", default=None,
                    help="Bound diffusion's piece scope (expensive); default: all 40, "
                         "same as every other track.")
    p.add_argument("--diffusion-max-frames", type=int, default=None,
                    help="Bound diffusion's frames/piece; default: 16, same as every "
                         "other track.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    shared_bins = np.linspace(0.0, args.vmax, args.nbins + 1)
    n_bins = args.nbins
    tick_idx = np.unique(np.linspace(0, n_bins, 7).astype(int))

    tracks = tracks_for_lead(args.lead)
    results = []
    for t in tracks:
        print(f"accumulating {t['label']!r} ...", flush=True)
        try:
            conf, n_pixels = accumulate_track(
                t, shared_bins, device,
                diffusion_pieces=args.diffusion_pieces,
                diffusion_max_frames=args.diffusion_max_frames,
            )
        except Exception as e:
            print(f"skip {t['label']!r}: {e}")
            continue
        if n_pixels == 0:
            print(f"skip {t['label']!r}: no pixels accumulated")
            continue
        print(f"  {t['label']!r}: {n_pixels} pixels", flush=True)
        results.append((t, conf))

    if not results:
        raise SystemExit(f"no tracks available at lead={args.lead:g}h")

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 5),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, (t, conf) in zip(axes, results):
        with np.errstate(divide="ignore"):
            log_conf = np.log(np.where(conf > 0, conf, np.nan))
        im = ax.imshow(log_conf, origin="upper", aspect="auto")
        ax.plot(np.arange(n_bins), np.arange(n_bins), "o", mfc="none", mec="red", ms=4)
        ax.set_xticks(tick_idx - 0.5)
        ax.set_yticks(tick_idx - 0.5)
        ax.set_xticklabels([f"{shared_bins[i]:.2g}" for i in tick_idx], rotation=45, fontsize=7)
        ax.set_yticklabels([f"{shared_bins[i]:.2g}" for i in tick_idx], fontsize=7)
        ax.set_title(t["label"], fontsize=10)
        ax.set_xlabel("predicted (mm/15min)", fontsize=8)
    axes[0].set_ylabel("truth (mm/15min)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="log(count)")

    scope_note = "full test-set scan" if args.diffusion_pieces is None and args.diffusion_max_frames is None \
        else f"full test-set scan (diffusion bounded: pieces={args.diffusion_pieces}, max_frames={args.diffusion_max_frames})"
    fig.suptitle(f"Contingency (truth vs predicted rain rate), all tracks -- "
                 f"lead={args.lead:g}h ({scope_note})", fontsize=11)

    out = pathlib.Path(args.out) if args.out else (
        FIGURES_DIR / f"2026-07-22_full_comparison/confusion_matrix_compare_lead{args.lead:g}h.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
