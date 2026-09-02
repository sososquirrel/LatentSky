"""Tropic-wide truth-vs-pred map for the diffusion track (fig 04 of the
final_12_diffusion suite), rebuilt via latentsky.predict_dispatch at the
standard t_start=100 (not the earlier diffusion_final_figures.py's
t_start=150) so it's consistent with every other diffusion evaluation
done today. Single-piece qualitative figure -- no full-scan needed.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from latentsky.paths import FIGURES_DIR
from latentsky.predict_dispatch import predict_piece_track
from latentsky.scale_analysis import _DEFAULT_LAT_EXTENT, _DEFAULT_LON_EXTENT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["diffusion_lead0.5h", "diffusion_lead1h"])
    p.add_argument("--labels", nargs="+", default=["t+30min", "t+1h"])
    p.add_argument("--piece", type=int, default=32)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--t-start", type=int, default=100)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = {}
    for name, label in zip(args.runs, args.labels):
        track = {"kind": "diffusion", "run": name, "t_start": args.t_start,
                  "ensemble_size": args.ensemble_size}
        X, Y, Yp = predict_piece_track(track, args.piece, args.max_frames, device=None)
        results[name] = (label, Y, Yp)
        print(f"{name}: truth mean={Y.mean():.4f} pred mean={Yp.mean():.4f}", flush=True)

    # Full 360-degree domain -- unlike a regional crop that might cross the
    # dateline/prime-meridian (see plot_method_illustration.py), the full
    # span must NOT be wrapped into -180..180: doing so collapses both
    # edges to the same longitude (0 and 360 both map to 0), which makes
    # cartopy's transform singular (identical xlims) instead of drawing
    # the whole globe. Keep everything in the data's native 0-360 range.
    proj = ccrs.PlateCarree()
    lat0, lat1 = _DEFAULT_LAT_EXTENT
    lon0, lon1 = _DEFAULT_LON_EXTENT
    extent_xy = (lon0, lon1, lat0, lat1)

    fig = plt.figure(figsize=(8 * len(results), 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, len(results))
    axes = [[fig.add_subplot(gs[r, c], projection=proj) for c in range(len(results))]
            for r in range(2)]
    for col, (name, (label, Y, Yp)) in enumerate(results.items()):
        vmax = max(np.percentile(Y[0, 0], 99.5), 1.0)
        axes[0][col].imshow(Y[0, 0], cmap="Blues", vmin=0, vmax=vmax, aspect="auto",
                             origin="lower", extent=extent_xy, transform=proj)
        axes[0][col].set_title(f"{label}: truth", fontsize=10)
        axes[1][col].imshow(Yp[0, 0], cmap="Blues", vmin=0, vmax=vmax, aspect="auto",
                             origin="lower", extent=extent_xy, transform=proj)
        axes[1][col].set_title(f"{label}: diffusion pred (t_start={args.t_start})", fontsize=10)
        for ax in (axes[0][col], axes[1][col]):
            # set_extent with a full 360-degree span is a known cartopy
            # trap (0 and 360 both normalize to the same x, so it reads as
            # "identical xlims" and cartopy's "automatic expansion" fallback
            # blows the view out far past the actual data, squeezing the
            # real image into a sliver -- confirmed by looking at the
            # actual rendered output, not just the warning text). set_global
            # is the correct, cartopy-blessed way to say "show the whole
            # globe" without triggering that edge case.
            ax.set_global()
            ax.set_ylim(lat0, lat1)
            ax.coastlines(resolution="50m", linewidth=0.8, color="black", zorder=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Diffusion -- tropic-wide truth vs pred (piece {args.piece}, frame 0)", y=1.03)
    out = pathlib.Path(args.out) if args.out else (
        FIGURES_DIR / "final_12_diffusion/04_map_truth_vs_pred.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
