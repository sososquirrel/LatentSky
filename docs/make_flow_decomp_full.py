"""The flow's version of the §5 decomposition figure (eval_method_decomp):
left — the standard example frame's flow-draw matched pairs on the dlnA/dlnrho
plane (diamonds, numbered, colored by scale, sized by precip volume);
right — the full held-out test record (systems_alltest_flow_lead3h.csv.gz,
125k matched pairs) with precip-weighted band barycenters. Same axes, limits
and color scale conventions as the two-head figure for direct comparison.
CPU, ~40 s."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm

from latentsky.metrics import cell_area_km2, match_systems
from latentsky.models import VelocityUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = REPO / "docs" / "assets_model_reference"
INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
THR, IOU_MIN, SCALE, SIGMA = 0.1, 0.1, 2.0, 0.1876

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def frame_flow_matches() -> pd.DataFrame:
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    area_row = cell_area_km2(320, 1440)[r * P:(r + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME], dtype=np.float32)[sl]
                  for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13], dtype=np.float32)[sl]
    with torch.no_grad():
        twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
        twohead.load_state_dict(torch.load(
            RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
            map_location="cpu", weights_only=True))
        twohead.eval()
        cond = torch.from_numpy(x[None])
        det_t, _ = twohead(cond)
        net = VelocityUNet(cond_ch=10, base_ch=64, depth=5,
                                       time_emb_dim=128)
        net.load_state_dict(torch.load(
            RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
            map_location="cpu", weights_only=True))
        net.eval()
        rf = RectifiedFlow(sigma=SIGMA)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        g = torch.Generator().manual_seed(7)
        z = rf.sample(net, cond, zf, n_steps=32, generator=g)
        flow = np.clip(np.expm1(SCALE * np.clip(z.numpy()[0, 0], 0, 2.5)), 0, None)
    mm = match_systems(truth, flow, THR, area_row, iou_min=IOU_MIN)
    return mm[mm.role == "match"].reset_index(drop=True)


def main():
    m = frame_flow_matches()
    full = pd.read_csv(EVAL / "systems_alltest_flow_lead3h.csv.gz")
    fullm = full[full.role == "match"].reset_index(drop=True)
    print(f"frame matches {len(m)}; full record matches {len(fullm)}")

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.3), sharex=True, sharey=True)
    # shared conventions with eval_method_decomp (two-head): same range, same
    # color normalization, same cloud size/alpha - the panels compare directly
    lim = 3.2
    cnorm = LogNorm(vmin=28.0, vmax=1500.0)

    def decorate(ax):
        ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
        xs = np.linspace(-lim, lim, 2)
        ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("dlnA  (area error)")

    ax = axs[0]
    ssize = 12 + 260 * (m["P_t"] / m["P_t"].max())
    ax.scatter(m["dlnA"], m["dlnrho"], s=ssize, c=m["scale"], cmap="viridis",
               norm=cnorm, alpha=0.85, edgecolors="0.2", linewidths=0.5,
               marker="D")
    for k, r in m.iterrows():
        ax.annotate(str(k + 1), (r["dlnA"], r["dlnrho"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    decorate(ax)
    ax.set_ylabel(r"dln$\rho$  (intensity error)")
    ax.set_title("one frame, one flow draw (diamonds)", fontsize=9)
    ax.text(0.6 * lim, -0.5 * lim, "dlnP = 0", fontsize=7, color="#b2182b",
            ha="center")

    ax = axs[1]
    npts = len(fullm)
    sc = ax.scatter(fullm["dlnA"], fullm["dlnrho"], s=1.5,
                    c=fullm["scale"], cmap="viridis", norm=cnorm,
                    alpha=0.25, linewidths=0, rasterized=True)
    vir = plt.get_cmap("viridis")
    bands = [("small < 65 km", 0, 65, 40), ("65–200 km", 65, 200, 115),
             ("large > 200 km", 200, np.inf, 350)]
    for name, lo, hi, lrep in bands:
        g = fullm[(fullm["scale"] >= lo) & (fullm["scale"] < hi)]
        wg = g["P_t"].to_numpy()
        bA = float((wg * g["dlnA"]).sum() / wg.sum())
        bR = float((wg * g["dlnrho"]).sum() / wg.sum())
        ax.plot(bA, bR, marker="X", ms=12, mec="0.05", mew=1.1,
                mfc=vir(cnorm(lrep)), zorder=6, ls="none",
                label=f"{name} (n={len(g):,})")
        print(f"BARY {name}: ({bA:+.2f}, {bR:+.2f}), n={len(g)}")
    ax.legend(loc="upper left", fontsize=7, frameon=False,
              title="precip-weighted barycenter", title_fontsize=7)
    decorate(ax)
    ax.set_title(f"full held-out test record — {npts:,} matched flow pairs",
                 fontsize=9)
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.015)
    cb.set_label("system scale L = √A (km)", fontsize=8)
    fig.savefig(OUT / "flow_decomp_full.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved flow_decomp_full.png")


if __name__ == "__main__":
    main()
