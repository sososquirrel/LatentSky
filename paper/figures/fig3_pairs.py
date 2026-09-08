"""Paper Fig 3 — pair recognition on one frame: truth / two-head / flow draw.

Caption sentence: a matched truth system keeps its color and number wherever
a model found it; the flow recovers small systems the two-head misses, at the
price of small false alarms. Restyled from docs/make_flow_pairs_panel.py:
no titles, in-panel labels, thin rain colorbar below. CPU ~1 min.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from paperfig import W2

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "docs"))
from make_flow_pairs_panel import (  # noqa: E402
    BLOCK, FRAME, INPUT_VARS, IOU_MIN, P, PIECE, PIECES, RUNS, SCALE, THR)

from latentsky.metrics import cell_area_km2, label_systems, match_systems  # noqa: E402
from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead  # noqa: E402
from latentsky.models.flow import RectifiedFlow  # noqa: E402

matplotlib.rcParams["savefig.bbox"] = "tight"
OUT = REPO / "paper" / "figures"


def main():
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    area_row = cell_area_km2(320, 1440)[r * P:(r + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME],
                             dtype=np.float32)[sl] for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13],
                       dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])

    with torch.no_grad():
        twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5,
                                olr_channel_idx=2)
        twohead.load_state_dict(torch.load(
            RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
            map_location="cpu", weights_only=True))
        twohead.eval()
        det_t, _ = twohead(cond)
        det = np.clip(det_t.numpy()[0, 0], 0, None)
        net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5,
                                       time_emb_dim=128)
        net.load_state_dict(torch.load(
            RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
            map_location="cpu", weights_only=True))
        net.eval()
        rf = RectifiedFlow(sigma=0.1876)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        g = torch.Generator().manual_seed(7)
        z = rf.sample(net, cond, zf, n_steps=32, generator=g)
        flow = np.clip(np.expm1(SCALE * np.clip(z.numpy()[0, 0], 0, 2.5)),
                       0, None)

    lt, tt = label_systems(truth, THR, area_row)
    preds = {"two-head": det, "flow draw": flow}
    labels, tables, matches = {}, {}, {}
    for name, f in preds.items():
        labels[name], tables[name] = label_systems(f, THR, area_row)
        matches[name] = match_systems(truth, f, THR, area_row,
                                      iou_min=IOU_MIN)

    lut_t = {(round(a, 6), round(b, 6)): int(l)
             for a, b, l in zip(tt["row"], tt["col"], tt["label"])}
    luts_p = {n: {(round(a, 6), round(b, 6)): int(l)
                  for a, b, l in zip(tables[n]["row"], tables[n]["col"],
                                     tables[n]["label"])} for n in preds}

    truth_match = {}
    for n in preds:
        for _, rr in matches[n][matches[n].role == "match"].iterrows():
            tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
            truth_match.setdefault(tid, {})[n] = rr
    order = sorted(truth_match,
                   key=lambda tid: -tt.set_index("label").loc[tid, "A"])
    _t20 = plt.get_cmap("tab20")
    pair_colors = [_t20(i) for i in range(20) if i not in (6, 7)]
    color_of = {tid: pair_colors[k % len(pair_colors)]
                for k, tid in enumerate(order)}
    num_of = {tid: k + 1 for k, tid in enumerate(order)}
    neither = len(tt) - len(truth_match)

    vmax = max(truth.max(), det.max(), flow.max(), 0.5)
    norm = LogNorm(vmin=THR, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#f7f9fb")

    def outline(ax, lab, lid, color, ls="-", lw=1.4):
        ax.contour(lab == lid, levels=[0.5], colors=[color], linewidths=lw,
                   linestyles=ls)

    fig, axes = plt.subplots(1, 3, figsize=(W2, W2 * 0.40))
    panels = [("A. truth", truth, None),
              ("B. two-head", det, "two-head"),
              ("C. flow draw", flow, "flow draw")]
    im = None
    for ax, (lab, f, model) in zip(axes, panels):
        im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="bottom")
        if model is None:
            for tid in truth_match:
                outline(ax, lt, tid, color_of[tid], lw=1.6)
                rrow = tt.set_index("label").loc[tid]
                ax.text(rrow["col"], rrow["row"], str(num_of[tid]),
                        fontsize=5.5, fontweight="bold", color="0.1",
                        ha="center", va="center")
            mm = matches["two-head"]
            for _, rr in mm[mm.role == "miss"].iterrows():
                tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
                if tid not in truth_match:
                    outline(ax, lt, tid, "0.05", ls="--", lw=0.7)
        else:
            mm = matches[model]
            for _, rr in mm[mm.role == "match"].iterrows():
                tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
                pid = luts_p[model][(round(rr["row_p"], 6),
                                    round(rr["col_p"], 6))]
                outline(ax, labels[model], pid, color_of[tid], lw=1.6)
                ax.text(rr["col_p"], rr["row_p"], str(num_of[tid]),
                        fontsize=5.5, fontweight="bold", color="0.1",
                        ha="center", va="center")
            for _, rr in mm[mm.role == "false_alarm"].iterrows():
                pid = luts_p[model][(round(rr["row"], 6),
                                    round(rr["col"], 6))]
                outline(ax, labels[model], pid, "#d62728", ls="--", lw=0.8)

    handles = [
        Line2D([], [], color=pair_colors[0], lw=1.6,
               label="matched truth system (same color and number)"),
        Line2D([], [], color="0.05", lw=0.8, ls="--",
               label=f"missed by both ({neither})"),
        Line2D([], [], color="#d62728", lw=0.8, ls="--",
               label="false alarm"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(wspace=0.04)
    cb = fig.colorbar(im, ax=list(axes), orientation="horizontal",
                      fraction=0.05, pad=0.16, aspect=60)
    cb.set_label("rain (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    fig.savefig(OUT / "fig3_pairs.pdf", dpi=300)
    print("saved fig3_pairs.pdf")


if __name__ == "__main__":
    main()
