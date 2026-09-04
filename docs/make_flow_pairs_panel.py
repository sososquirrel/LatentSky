"""Three-panel matched-pairs figure: truth / two-head / flow draw (+3 h) on
the standard example block, pair colors keyed to TRUTH-system identity so a
system recovered by only one model shows its color only there. CPU (~30 s)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from latentsky.metrics import cell_area_km2, label_systems, match_systems
from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"
INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
THR, IOU_MIN, SCALE = 0.1, 0.1, 2.0

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def main():
    torch.set_num_threads(4)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    area_row = cell_area_km2(320, 1440)[r * P:(r + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME], dtype=np.float32)[sl]
                  for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13], dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])

    with torch.no_grad():
        twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
        twohead.load_state_dict(torch.load(
            RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
            map_location="cpu", weights_only=True))
        twohead.eval()
        det_t, _ = twohead(cond)
        det = np.clip(det_t.numpy()[0, 0], 0, None)
        net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
        net.load_state_dict(torch.load(
            RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
            map_location="cpu", weights_only=True))
        net.eval()
        rf = RectifiedFlow(sigma=0.1876)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        g = torch.Generator().manual_seed(7)
        z = rf.sample(net, cond, zf, n_steps=32, generator=g)
        flow = np.clip(np.expm1(SCALE * np.clip(z.numpy()[0, 0], 0, 2.5)), 0, None)

    lt, tt = label_systems(truth, THR, area_row)
    preds = {"two-head": det, "flow draw": flow}
    labels, tables, matches = {}, {}, {}
    for name, f in preds.items():
        labels[name], tables[name] = label_systems(f, THR, area_row)
        matches[name] = match_systems(truth, f, THR, area_row, iou_min=IOU_MIN)

    lut_t = {(round(a, 6), round(b, 6)): int(l)
             for a, b, l in zip(tt["row"], tt["col"], tt["label"])}
    luts_p = {n: {(round(a, 6), round(b, 6)): int(l)
                  for a, b, l in zip(tables[n]["row"], tables[n]["col"], tables[n]["label"])}
              for n in preds}

    # assign a color index per truth system matched by >=1 model
    truth_match = {}
    for n in preds:
        for _, rr in matches[n][matches[n].role == "match"].iterrows():
            tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
            truth_match.setdefault(tid, {})[n] = rr
    order = sorted(truth_match, key=lambda tid: -tt.set_index("label").loc[tid, "A"])
    _t20 = plt.get_cmap("tab20")
    pair_colors = [_t20(i) for i in range(20) if i not in (6, 7)]
    color_of = {tid: pair_colors[k % len(pair_colors)] for k, tid in enumerate(order)}
    num_of = {tid: k + 1 for k, tid in enumerate(order)}

    both = sum(1 for v in truth_match.values() if len(v) == 2)
    only = {n: sum(1 for v in truth_match.values() if list(v) == [n]) for n in preds}
    n_truth = len(tt)
    neither = n_truth - len(truth_match)
    print(f"truth systems {n_truth}: matched by both {both}, "
          f"two-head only {only['two-head']}, flow only {only['flow draw']}, "
          f"neither {neither}")

    vmax = max(truth.max(), det.max(), flow.max(), 0.5)
    norm = LogNorm(vmin=THR, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")

    def outline(ax, lab, lid, color, ls="-", lw=1.6):
        ax.contour(lab == lid, levels=[0.5], colors=[color], linewidths=lw,
                   linestyles=ls)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    panels = [("truth (+3 h)", truth, None), ("two-head (+3 h)", det, "two-head"),
              ("flow draw (+3 h)", flow, "flow draw")]
    for ax, (ttl, f, model) in zip(axes, panels):
        im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if model is None:
            for tid, v in truth_match.items():
                outline(ax, lt, tid, color_of[tid], lw=2.0)
                rrow = tt.set_index("label").loc[tid]
                ax.text(rrow["col"], rrow["row"], str(num_of[tid]), fontsize=6.5,
                        fontweight="bold", color="0.1", ha="center", va="center")
            for _, rr in matches["two-head"][matches["two-head"].role == "miss"].iterrows():
                tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
                if tid not in truth_match:
                    outline(ax, lt, tid, "0.05", ls="--", lw=0.9)
        else:
            mm = matches[model]
            for _, rr in mm[mm.role == "match"].iterrows():
                tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
                pid = luts_p[model][(round(rr["row_p"], 6), round(rr["col_p"], 6))]
                outline(ax, labels[model], pid, color_of[tid], lw=2.0)
                ax.text(rr["col_p"], rr["row_p"], str(num_of[tid]), fontsize=6.5,
                        fontweight="bold", color="0.1", ha="center", va="center")
            for _, rr in mm[mm.role == "false_alarm"].iterrows():
                pid = luts_p[model][(round(rr["row"], 6), round(rr["col"], 6))]
                outline(ax, labels[model], pid, "#d62728", ls="--", lw=1.0)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012)
    cb.set_label("rain (mm)", fontsize=8)
    handles = [
        Line2D([], [], color=pair_colors[0], lw=2.0,
               label="matched truth system — same color & number wherever a model found it"),
        Line2D([], [], color="0.05", lw=1.0, ls="--",
               label=f"missed by both ({neither + sum(only.values()) - sum(only.values())})"),
        Line2D([], [], color="#d62728", lw=1.0, ls="--", label="false alarm (per panel)"),
    ]
    handles[1] = Line2D([], [], color="0.05", lw=1.0, ls="--",
                        label=f"missed by both models ({neither})")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(f"pair recognition across models — matched by both: {both}, "
                 f"two-head only: {only['two-head']}, flow only: {only['flow draw']}, "
                 f"neither: {neither} (of {n_truth} truth systems)", fontsize=9.5)
    fig.savefig(OUT / "flow_pairs_panel.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved flow_pairs_panel.png")


if __name__ == "__main__":
    main()
