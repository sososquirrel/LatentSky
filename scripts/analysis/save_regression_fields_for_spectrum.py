"""Save full (T,1,H,W) truth/prediction fields for regression tracks (plain
one-head, two-head), on the same piece_ids/lead set already used for the
diffusion power-spectrum figure (scripts/analysis/plot_power_spectrum_diffusion.py),
so their spectra can be overlaid on it.

Cheap by construction: a regression forward pass is a single pass per frame
(no iterative diffusion sampling), so this runs in minutes on one GPU for
8 pieces x 6 leads, unlike the diffusion evaluation jobs.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from latentsky.inference import predict_piece, _load_model
from latentsky.registry import load_run

_LEADS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]
_PIECES = list(range(8))  # matches pieces 00-07 already used in the diffusion spectrum


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-template", required=True,
                    help="format string with a {lead} placeholder, e.g. "
                         "'patchsplit_onehead_lead{lead}h' or "
                         "'patchsplit_twohead_olr_fss_lead{lead}h'")
    p.add_argument("--tag", required=True, help="output subdirectory name")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--out-dir", default="/glade/derecho/scratch/sabramian/LatentSky_regression_fields")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    out_dir = pathlib.Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for lead_label in _LEADS:
        lead_num = lead_label.rstrip("h")
        run_name = args.run_template.format(lead=lead_num)
        run = load_run(run_name)
        model = _load_model(run, device)
        for pid in _PIECES:
            X, Y, Yp = predict_piece(run, pid, max_frames=args.max_frames, device=device, model=model)
            out_path = out_dir / f"{run_name}_piece{pid:02d}.npz"
            np.savez_compressed(out_path, Y=Y.astype(np.float32), Yp=Yp.astype(np.float32),
                                 piece_id=pid, lead_h=run.config.lead_hours)
            print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
