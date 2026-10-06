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
    INPUT_VARS, IOU_MIN, P, PIECES, RUNS, SCALE, THR)

# the standard example block of the paper (West Pacific warm pool), as in Fig. 2
PIECE, FRAME, BLOCK = 8, 64, (1, 4)
# FIG3_EXAMPLE="piece,row,col,frame[,outfile]" tries another frame without editing the script
import os  # noqa: E402
if os.environ.get("FIG3_EXAMPLE"):
    _e = os.environ["FIG3_EXAMPLE"].split(",")
    PIECE, BLOCK, FRAME = int(_e[0]), (int(_e[1]), int(_e[2])), int(_e[3])
    OUT_FILE = Path(_e[4]) if len(_e) > 4 else None
else:
    OUT_FILE = None

from latentsky.metrics import cell_area_km2, label_systems, match_systems  # noqa: E402
from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead  # noqa: E402
from latentsky.models.flow import RectifiedFlow  # noqa: E402

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

    coast = np.load(REPO / "data/eval_legacy/_landmask_320x1440_-40_40_50m.npy")[sl].astype(float)
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
    pair_colors = ["#1f78b4", "#33a02c", "#e6ab02", "#7570b3", "#e7298a", "#66a61e",
                   "#a6761d", "#1b9e77", "#d95f02", "#386cb0", "#f0027f", "#bf5b17",
                   "#666666", "#8da0cb", "#fc8d62", "#a6d854", "#ffd92f", "#e5c494"]
    HIT = "#2a7f62"                 # the outline says "matched"; the fill says which pair
    color_of = {tid: pair_colors[k % len(pair_colors)] for k, tid in enumerate(order)}
    num_of = {tid: k + 1 for k, tid in enumerate(order)}
    neither = len(tt) - len(truth_match)

    vmax = max(truth.max(), det.max(), flow.max(), 0.5)
    norm = LogNorm(vmin=THR, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#ffffff")
    cmap.set_bad("#ffffff")
    FA = "#b03a2e"
    nA = sum(1 for d in truth_match.values() if len(d) == 2)
    nB = sum(1 for d in truth_match.values() if list(d) == ["two-head"])
    nC = sum(1 for d in truth_match.values() if list(d) == ["flow draw"])
    print(f"truth systems {len(tt)}; matched by both {nA}, two-head only {nB}, "
          f"flow only {nC}, neither {neither}")
    for n_ in preds:
        mm_ = matches[n_]
        print(f"  {n_}: matched {(mm_.role == 'match').sum()}, missed {(mm_.role == 'miss').sum()}, "
              f"false alarms {(mm_.role == 'false_alarm').sum()}")

    def outline(ax, lab, lid, color, ls="-", lw=1.2):
        ax.contour(lab == lid, levels=[0.5], colors=[color], linewidths=lw,
                   linestyles=ls)

    def fill(ax, lab, lid, color, alpha):
        ax.contourf(lab == lid, levels=[0.5, 1.5], colors=[color], alpha=alpha)

    def matched(ax, lab, lid, tid):  # a system found by the forecast (row 2: one colour)
        fill(ax, lab, lid, HIT, 0.30)
        outline(ax, lab, lid, HIT, lw=0.9)

    def pair(ax, lab, lid, tid, num_xy):  # row 3: the pair's own colour and number
        fill(ax, lab, lid, color_of[tid], 0.55)
        outline(ax, lab, lid, color_of[tid], lw=1.1)
        ax.text(num_xy[0], num_xy[1], str(num_of[tid]), fontsize=5.5, fontweight="bold",
                color="0.1", ha="center", va="center")

    def missed(ax, tid):            # a truth system no forecast found
        fill(ax, lt, tid, "0.15", 0.55)
        outline(ax, lt, tid, "0.05", lw=0.7)

    def false_alarm(ax, lab, pid):  # a forecast system that matches nothing
        fill(ax, lab, pid, FA, 0.35)
        outline(ax, lab, pid, FA, lw=0.6)

    def title(ax, letter, text, sub=None):
        y = 1.10 if sub else 1.03      # with a second line of counts, the title moves up
        ax.text(0.0, y, letter, transform=ax.transAxes, fontsize=9,
                fontfamily="cmb10", ha="left", va="bottom")
        ax.text(0.10, y, text, transform=ax.transAxes, fontsize=8.5,
                ha="left", va="bottom")
        if sub:
            ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=6.5,
                    color="0.35", ha="left", va="bottom")

    # the paper's text font
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})
    # two rows (rain fields; matched systems), the truth set apart from the two forecasts
    fig = plt.figure(figsize=(W2, W2 * 0.78))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 0.10, 1, 1], left=0.005, right=0.945,
                          top=0.905, bottom=0.12, wspace=0.05, hspace=0.32)
    cols = [0, 2, 3]
    ax_rain = [fig.add_subplot(gs[0, c]) for c in cols]
    ax_sys = [fig.add_subplot(gs[1, c]) for c in cols]
    for ax in ax_rain + ax_sys:
        ax.contour(coast, levels=[0.5], colors=["0.35"], linewidths=0.5, zorder=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("0.3")
    # the separation: a thin rule between the truth column and the forecasts
    x_sep = 0.5 * (ax_rain[0].get_position().x1 + ax_rain[1].get_position().x0)
    fig.add_artist(Line2D([x_sep, x_sep], [0.125, 0.985], color="0.6", lw=0.6,
                          transform=fig.transFigure))
    fig.text(0.5 * (ax_rain[0].get_position().x0 + ax_rain[0].get_position().x1), 0.985,
             "reference", ha="center", va="top", fontsize=8, color="0.35")
    fig.text(0.5 * (ax_rain[1].get_position().x0 + ax_rain[2].get_position().x1), 0.985,
             "forecasts, +3 h", ha="center", va="top", fontsize=8, color="0.35")

    # ---- top row: the rain fields ------------------------------------------
    fields = [("A.", "Truth", truth), ("B.", "Two-head U-Net", det), ("C.", "Flow, one draw", flow)]
    im = None
    for ax, (letter, name, f) in zip(ax_rain, fields):
        im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
        title(ax, letter, name)
    for tid in truth_match:                        # the pairs: one colour per truth system
        outline(ax_rain[0], lt, tid, color_of[tid], lw=1.1)
    for ax, model in zip(ax_rain[1:], ("two-head", "flow draw")):
        mm = matches[model]
        for _, rr in mm[mm.role == "match"].iterrows():
            tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
            pid = luts_p[model][(round(rr["row_p"], 6), round(rr["col_p"], 6))]
            outline(ax, labels[model], pid, color_of[tid], lw=1.1)
    # 500 km scale bar on the truth (pixel width by cos(lat))
    ybar = 8
    latbar = -40 + 0.25 * (r * P + ybar)
    nbar = 500.0 / (27.83 * np.cos(np.deg2rad(latbar)))
    ax_rain[0].plot([5, 5 + nbar], [ybar, ybar], "-", color="0.1", lw=2.0,
                    solid_capstyle="butt", zorder=6)
    ax_rain[0].text(5 + nbar / 2, ybar + 3, "500 km", fontsize=6.5, ha="center", color="0.1")

    # ---- bottom row: the systems and their matches, on white (grey = missed) ------
    for ax in ax_sys:
        ax.set_xlim(ax_rain[0].get_xlim())
        ax.set_ylim(ax_rain[0].get_ylim())
        ax.set_aspect("equal")
    title(ax_sys[0], "D.", "Truth systems", f"{len(tt)}, of which {neither} missed by both")
    for tid in truth_match:
        matched(ax_sys[0], lt, tid, tid)
    mm = matches["two-head"]
    for _, rr in mm[mm.role == "miss"].iterrows():
        tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
        if tid not in truth_match:
            missed(ax_sys[0], tid)
    for ax, letter, model in zip(ax_sys[1:], ("E.", "F."), ("two-head", "flow draw")):
        mm = matches[model]
        n_match = int((mm.role == "match").sum())
        n_miss = int((mm.role == "miss").sum())
        n_fa = int((mm.role == "false_alarm").sum())
        title(ax, letter, "Two-head U-Net" if model == "two-head" else "Flow, one draw",
              f"{n_match} matched, {n_fa} false alarms")
        for _, rr in mm[mm.role == "match"].iterrows():
            tid = lut_t[(round(rr["row"], 6), round(rr["col"], 6))]
            pid = luts_p[model][(round(rr["row_p"], 6), round(rr["col_p"], 6))]
            matched(ax, labels[model], pid, tid)
        for _, rr in mm[mm.role == "false_alarm"].iterrows():
            pid = luts_p[model][(round(rr["row"], 6), round(rr["col"], 6))]
            false_alarm(ax, labels[model], pid)

    # rain colorbar to the right of the top row; legend under the whole figure
    p = ax_rain[2].get_position()
    cax = fig.add_axes([p.x1 + 0.012, p.y0, 0.012, p.y1 - p.y0])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("rain in 15 min (mm)", fontsize=8)
    ticks = [0.1, 0.3, 1, 3, 10]
    cb.set_ticks([t for t in ticks if t <= vmax])
    cb.set_ticklabels([f"{t:g}" for t in ticks if t <= vmax])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=pair_colors[0], lw=1.1, label="top: matched pair, same colour in A to C"),
        Patch(fc=HIT, alpha=0.30, ec=HIT, lw=0.9, label="matched: found by the forecast"),
        Patch(fc="0.15", alpha=0.55, ec="0.05", lw=0.7, label="missed by both models (in D)"),
        Patch(fc=FA, alpha=0.35, ec=FA, lw=0.6,
              label="false alarm: forecast system matching nothing"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.475, 0.0), handlelength=1.8, columnspacing=1.4)
    fig.savefig(OUT_FILE or OUT / "fig3_pairs.pdf", dpi=300)
    print("saved", OUT_FILE or "fig3_pairs.pdf")


if __name__ == "__main__":
    main()
