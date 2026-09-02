"""GPU job: run inference with every mseuv_lead* checkpoint on ONE fixed
input snapshot (piece 2, frame 75 -- same as figs 01/02), full domain
(320x1440, so convolutions see full context, no patch-edge artifacts).
Also runs the same for a West-Africa-cropped region at lead=3h for the
method-illustration figure.

Caches everything to an .npz so figs 03 and 05 can be built (fast, CPU
only) without re-running inference.
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch

from latentsky import load_run
from latentsky.data import load_piece_arrays
from latentsky.inference import _load_model, _infer
from latentsky.paths import PAPER_FIGURES_DIR

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = PAPER_FIGURES_DIR / "_inference_cache.npz"

PIECE_ID = 2
FRAME = 60  # same snapshot used for figs 01, 02, 03 (must satisfy FRAME+delta_index<96 for lead6h)
LEADS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# West Africa crop, index space (lat_extent=(-40,40), lon_extent=(0,360),
# 320x1440 grid, 0.25 deg/cell): rows 160:256 -> lat 0..24N,
# cols 0:96 -> lon 0..24E. 96 is divisible by 32 (UNet depth=5 needs it),
# but we crop AFTER inference on the full frame, so this constraint
# doesn't actually bind -- kept anyway for a clean, round region.
WA_LAT_IDX = (160, 256)
WA_LON_IDX = (0, 96)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    preds_full = {}
    X_full = None
    Y_full = None

    for lead in LEADS:
        name = f"mseuv_lead{lead:g}h"
        print(f"=== {name} ===", flush=True)
        run = load_run(name)
        cfg = run.config
        X, Y = load_piece_arrays(cfg.pieces_dir, PIECE_ID, cfg.input_vars,
                                 cfg.target_prefix, cfg.delta_index,
                                 n_frames=FRAME + 1)
        X1 = X[FRAME:FRAME + 1]
        Y1 = Y[FRAME:FRAME + 1]
        model = _load_model(run, device)
        Yp = _infer(model, X1, Y1, batch_size=1, device=device)
        preds_full[name] = Yp[0, 0]  # (H, W)
        if X_full is None:
            X_full = X1[0]  # (C, H, W) -- input channels identical index-wise across leads
            Y_full = Y1[0, 0]  # truth at lead 0's target time... but we want TRUTH
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Yp range: {Yp.min():.3f} {Yp.max():.3f}", flush=True)

    # Truth for the multi-lead panel: use lead=0h's target (== input state
    # itself, since delta=1 raw step only) as the reference "t=0" truth
    # panel; each lead's own true FUTURE state is Y at that lead's config.
    # For fig03 we mainly need the predictions; also grab true future state
    # per lead for context.
    truths_full = {}
    for lead in LEADS:
        name = f"mseuv_lead{lead:g}h"
        run = load_run(name)
        cfg = run.config
        _, Y = load_piece_arrays(cfg.pieces_dir, PIECE_ID, cfg.input_vars,
                                 cfg.target_prefix, cfg.delta_index,
                                 n_frames=FRAME + 1)
        truths_full[name] = Y[FRAME, 0]

    # --- West Africa crop, lead=3h, for the method-illustration figure ---
    name3h = "mseuv_lead3h"
    run3h = load_run(name3h)
    yt_wa = truths_full[name3h][WA_LAT_IDX[0]:WA_LAT_IDX[1], WA_LON_IDX[0]:WA_LON_IDX[1]]
    yp_wa = preds_full[name3h][WA_LAT_IDX[0]:WA_LAT_IDX[1], WA_LON_IDX[0]:WA_LON_IDX[1]]

    save_dict = {
        "X_full": X_full,  # (C,H,W)
        "input_vars": np.array(load_run("mseuv_lead0h").config.input_vars),
        "leads": np.array(LEADS),
        "piece_id": PIECE_ID,
        "frame": FRAME,
        "wa_lat_idx": np.array(WA_LAT_IDX),
        "wa_lon_idx": np.array(WA_LON_IDX),
        "yt_wa": yt_wa,
        "yp_wa": yp_wa,
    }
    for lead in LEADS:
        name = f"mseuv_lead{lead:g}h"
        save_dict[f"pred_{lead:g}h"] = preds_full[name]
        save_dict[f"truth_{lead:g}h"] = truths_full[name]

    np.savez_compressed(OUT, **save_dict)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
