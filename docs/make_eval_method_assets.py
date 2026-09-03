"""Worked example of the cluster detection + P = rho*A decomposition for
docs/model_reference.html, on the standard example block (truth vs two-head,
+3 h, from example_fields.npz). Pure CPU — login-node safe."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from latentsky.metrics import (cell_area_km2, default_scale_edges, label_systems,
                               match_systems, summarize_by_scale)

REPO = Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "docs" / "assets_model_reference"
P, BLOCK_ROW = 128, 1  # block (1,4): domain rows 128..255
THR, IOU_MIN = 0.1, 0.1

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def main():
    d = np.load(OUT / "example_fields.npz")
    truth, det = d["truth"], d["det"]
    area_row = cell_area_km2(320, 1440)[BLOCK_ROW * P:(BLOCK_ROW + 1) * P]

    lt, tt = label_systems(truth, THR, area_row)
    lp, tp = label_systems(det, THR, area_row)
    df = match_systems(truth, det, THR, area_row, iou_min=IOU_MIN)
    m = df[df.role == "match"].reset_index(drop=True)
    miss = df[df.role == "miss"]
    fa = df[df.role == "false_alarm"]
    print(f"truth systems: {len(tt)}, pred systems: {len(tp)}, "
          f"matched: {len(m)}, miss: {len(miss)}, false alarm: {len(fa)}")

    # --- figure 1: fields + roles (top) and the detection pipeline on a zoom (bottom)
    vmax = max(truth.max(), det.max(), 0.5)
    norm = LogNorm(vmin=THR, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

    def outline(ax, labels, label_id, color, ls="-", lw=1.4):
        ax.contour(labels == label_id, levels=[0.5], colors=[color],
                   linewidths=lw, linestyles=ls)

    # centroid-safe label lookup: map (row, col) back to the object table
    # (a concave system's centroid can fall OUTSIDE the blob, so indexing the
    # label image at the centroid can return background or a neighbor)
    lut_t = {(round(r, 6), round(c, 6)): int(l)
             for r, c, l in zip(tt["row"], tt["col"], tt["label"])}
    lut_p = {(round(r, 6), round(c, 6)): int(l)
             for r, c, l in zip(tp["row"], tp["col"], tp["label"])}

    def find_label(labels, row, col):
        lut = lut_t if labels is lt else lut_p
        return lut[(round(float(row), 6), round(float(col), 6))]

    for ax, f, ttl in [(axes[0], truth, "truth (+3 h)"),
                       (axes[1], det, "two-head (+3 h)")]:
        im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for _, r in miss.iterrows():
        outline(axes[0], lt, find_label(lt, r["row"], r["col"]), "0.05",
                ls="--", lw=1.0)
    for _, r in fa.iterrows():
        outline(axes[1], lp, find_label(lp, r["row"], r["col"]), "#d62728",
                ls="--", lw=1.0)
    _t20 = plt.get_cmap("tab20")
    pair_colors = [_t20(i) for i in range(20) if i not in (6, 7)]  # reds reserved for false alarms
    for k, r in m.iterrows():
        col_k = pair_colors[k % len(pair_colors)]
        lid_t = find_label(lt, r["row"], r["col"])
        lid_p = find_label(lp, r["row_p"], r["col_p"])
        outline(axes[0], lt, lid_t, col_k, lw=2.2)
        outline(axes[1], lp, lid_p, col_k, lw=2.2)
        for ax, rr, cc in [(axes[0], r["row"], r["col"]),
                           (axes[1], r["row_p"], r["col_p"])]:
            ax.text(cc, rr, str(k + 1), fontsize=7, fontweight="bold",
                    color="0.1", ha="center", va="center")

    # 500 km scalebar (meridional: 27.8 km/px)
    px500 = 500.0 / 27.8
    axes[0].plot([4, 4 + px500], [4, 4], color="0.1", lw=2.5)
    axes[0].text(4 + px500 / 2, 8, "500 km", fontsize=7, ha="center", color="0.1")

    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.012)
    cb.set_label("rain (mm)", fontsize=8)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=pair_colors[0], lw=2.2,
               label=f"matched pair ({len(m)}) — one color per pair, both panels"),
        Line2D([], [], color="0.05", lw=1.2, ls="--",
               label=f"miss ({len(miss)}) — real system, no prediction"),
        Line2D([], [], color="#d62728", lw=1.2, ls="--",
               label=f"false alarm ({len(fa)}) — predicted, not real"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(f"cluster detection (> {THR} mm, 8-connected) and greedy "
                 f"IoU ≥ {IOU_MIN} matching", fontsize=9, y=0.985)
    fig.savefig(OUT / "eval_method_clusters.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- all-frames matched pairs for the same block (cached; CPU two-head)
    systems_csv = OUT / "eval_method_systems_lead3h.csv"
    if systems_csv.exists():
        import pandas as pd
        alldf = pd.read_csv(systems_csv)
    else:
        import pandas as pd
        import torch
        torch.set_num_threads(4)
        from latentsky.models import UNet2DTwoHead
        RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
        twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
        twohead.load_state_dict(torch.load(
            RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
            map_location="cpu", weights_only=True))
        twohead.eval()
        PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
        INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
                      "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
                      "umeridional_850hPa", "rain_log1p"]
        DELTA = 13
        sl = np.s_[BLOCK_ROW * P:(BLOCK_ROW + 1) * P, 4 * P:5 * P]  # block (1,4)
        ytab = np.load(PIECES / "rain_total_piece08.npy", mmap_mode="r")
        rows = []
        with torch.no_grad():
            for t in range(96 - DELTA):
                x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece08.npy",
                                                 mmap_mode="r")[t], dtype=np.float32)[sl]
                              for v in INPUT_VARS], 0)
                pred, _ = twohead(torch.from_numpy(x[None]))
                pred = np.clip(pred.numpy()[0, 0], 0, None)
                yt = np.asarray(ytab[t + DELTA], dtype=np.float32)[sl]
                dfr = match_systems(yt, pred, THR, area_row, iou_min=IOU_MIN)
                dfr.insert(0, "frame", t)
                rows.append(dfr)
        alldf = pd.concat(rows, ignore_index=True)
        alldf.to_csv(systems_csv, index=False)
    allm = alldf[alldf.role == "match"].reset_index(drop=True)
    print(f"all-frames systems: {len(alldf)} (matched {len(allm)}) over 83 frames")

    # --- figure 2: the decomposition for the matched pairs
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.3), sharex=True, sharey=True)
    lim = 1.05 * max(np.abs(allm["dlnA"]).max(), np.abs(allm["dlnrho"]).max(),
                     np.abs(m["dlnA"]).max(), np.abs(m["dlnrho"]).max())
    from matplotlib.colors import LogNorm as _LN
    cnorm = _LN(vmin=min(m["scale"].min(), allm["scale"].min()),
                vmax=max(m["scale"].max(), allm["scale"].max()))

    def decorate(ax):
        ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
        xs = np.linspace(-lim, lim, 2)
        ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("dlnA  (area error)")

    ax = axs[0]
    ssize = 12 + 260 * (m["P_t"] / m["P_t"].max())
    sc = ax.scatter(m["dlnA"], m["dlnrho"], s=ssize, c=m["scale"], cmap="viridis",
                    norm=cnorm, alpha=0.85, edgecolors="0.2", linewidths=0.5)
    for k, r in m.iterrows():
        ax.annotate(str(k + 1), (r["dlnA"], r["dlnrho"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    decorate(ax)
    ax.set_ylabel(r"dln$\rho$  (intensity error)")
    ax.set_title("one frame (pairs numbered as above)", fontsize=9)
    ax.text(0.6 * lim, -0.5 * lim, "dlnP = 0", fontsize=7, color="#b2182b",
            ha="center")

    ax = axs[1]
    w = allm["P_t"].to_numpy()
    sc = ax.scatter(allm["dlnA"], allm["dlnrho"], s=9, c=allm["scale"],
                    cmap="viridis", norm=cnorm, alpha=0.45, linewidths=0)
    vir = plt.get_cmap("viridis")
    bands = [("small < 65 km", 0, 65, 40), ("65–200 km", 65, 200, 115),
             ("large > 200 km", 200, np.inf, 350)]
    txt = []
    for j, (name, lo, hi, lrep) in enumerate(bands):
        g = allm[(allm["scale"] >= lo) & (allm["scale"] < hi)]
        wg = g["P_t"].to_numpy()
        bA = float((wg * g["dlnA"]).sum() / wg.sum())
        bR = float((wg * g["dlnrho"]).sum() / wg.sum())
        ax.plot(bA, bR, marker="X", ms=12, mec="0.05", mew=1.1,
                mfc=vir(cnorm(lrep)), zorder=6, ls="none",
                label=f"{name} (n={len(g)})")
        txt.append(f"{name}: ({bA:+.2f}, {bR:+.2f}), n={len(g)}")
    ax.legend(loc="upper left", fontsize=7, frameon=False,
              title="precip-weighted barycenter", title_fontsize=7)
    print("barycenters:", "; ".join(txt))
    decorate(ax)
    ax.set_title(f"all frames of the day — {len(allm)} matched pairs", fontsize=9)
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.015)
    cb.set_label("system scale L = √A (km)", fontsize=8)
    fig.savefig(OUT / "eval_method_decomp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 3: object-based POD / FAR by scale over the day
    summ = summarize_by_scale(alldf, default_scale_edges(25, 600, 10)).by_scale
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(summ["scale_mid"], summ["POD"], "o-", color="#2166ac", label="POD")
    ax.plot(summ["scale_mid"], summ["FAR"], "s--", color="#d62728", label="FAR")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("system scale L = √A (km)")
    ax.set_ylabel("object-based rate")
    ax.set_title("detection skill by scale — one day, two-head (+3 h)", fontsize=9)
    for _, r in summ.iterrows():
        ax.annotate(f"{int(r['n_truth'])}", (r["scale_mid"], 1.0), fontsize=6,
                    color="0.45", ha="center", va="bottom", annotation_clip=False)
    ax.text(0.02, 0.93, "counts: truth systems per bin", transform=ax.transAxes,
            fontsize=6.5, color="0.45")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.savefig(OUT / "eval_method_podfar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(summ[["scale_mid", "n_truth", "POD", "FAR"]].round(3).to_string(index=False))

    w = m["P_t"].to_numpy()
    print(dict(dlnA_w=float((w * m["dlnA"]).sum() / w.sum()),
               dlnrho_w=float((w * m["dlnrho"]).sum() / w.sum()),
               dlnP_w=float((w * m["dlnP"]).sum() / w.sum()),
               pod=len(m) / (len(m) + len(miss)),
               far=len(fa) / (len(m) + len(fa))))


if __name__ == "__main__":
    main()
