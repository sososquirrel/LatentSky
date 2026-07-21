"""GPU job: true training-patch-sized (64x64 px = 16x16 deg) inference.

Unlike run_inference_cache.py (which infers on the FULL domain then crops
the output for display), this slices a genuine 64x64 window from the
already-cached full-domain input FIRST, then runs each lead checkpoint on
exactly that small crop -- a real single forward pass at the same spatial
size the model trains on (patch_size=64 in the configs).
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch

from latentsky import load_run
from latentsky.inference import _load_model, _infer

REPO = pathlib.Path(__file__).resolve().parents[2]
FULL_CACHE = REPO / "docs/paper_figures/_inference_cache.npz"
OUT = REPO / "docs/paper_figures/_inference_cache_patch64.npz"

LEADS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# 64x64 px = 16x16 deg window, centered on the active cluster around
# lat 6-10N / lon 5-10E seen in the West Africa figures.
P64_LAT_IDX = (160, 224)  # lat 0..16N
P64_LON_IDX = (0, 64)     # lon 0..16E


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    d = np.load(FULL_CACHE, allow_pickle=True)
    r0, r1 = P64_LAT_IDX
    c0, c1 = P64_LON_IDX
    X_patch = d["X_full"][:, r0:r1, c0:c1]  # (C, 64, 64) -- genuine patch input
    print(f"X_patch shape: {X_patch.shape}", flush=True)

    preds_patch = {}
    truths_patch = {}
    for lead in LEADS:
        name = f"mseuv_lead{lead:g}h"
        print(f"=== {name} ===", flush=True)
        run = load_run(name)
        model = _load_model(run, device)

        X1 = X_patch[None, ...]  # (1, C, 64, 64)
        Y1 = np.zeros((1, 1, X_patch.shape[1], X_patch.shape[2]), dtype=np.float32)  # dummy, unused
        Yp = _infer(model, X1, Y1, batch_size=1, device=device)
        preds_patch[name] = Yp[0, 0]
        truths_patch[name] = d[f"truth_{lead:g}h"][r0:r1, c0:c1]

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Yp range: {Yp.min():.3f} {Yp.max():.3f}", flush=True)

    save_dict = {
        "X_patch": X_patch,  # (C, 64, 64) -- the genuine patch input, nothing cropped after
        "input_vars": d["input_vars"],
        "leads": np.array(LEADS),
        "piece_id": d["piece_id"],
        "frame": d["frame"],
        # indices into the FULL 320x1440 domain grid -- needed to compute
        # correct lat/lon for this patch (X_patch's own shape is just 64x64
        # and carries no coordinate information by itself).
        "patch_lat_idx_in_full_domain": np.array(P64_LAT_IDX),
        "patch_lon_idx_in_full_domain": np.array(P64_LON_IDX),
    }
    for lead in LEADS:
        name = f"mseuv_lead{lead:g}h"
        save_dict[f"pred_{lead:g}h"] = preds_patch[name]
        save_dict[f"truth_{lead:g}h"] = truths_patch[name]

    np.savez_compressed(OUT, **save_dict)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
