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

from latentsky.metrics import cell_area_km2, label_systems, match_systems

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

    # annotations: L for the biggest pair and one small miss; IoU for the biggest pair
    big = m.loc[m["P_t"].idxmax()]
    tx = float(np.clip(big["col"] - 58, 2, 70))
    ty = float(np.clip(big["row"] + 20, 10, 118))
    axes[0].annotate(f"L = {big['L_t']:.0f} km, IoU = {big['iou']:.2f}",
                     xy=(big["col"], big["row"]), xytext=(tx, ty),
                     fontsize=7, color="#12335c",
                     arrowprops=dict(arrowstyle="-", color="#12335c", lw=0.7))
    small = miss.loc[miss["L_t"].idxmin()]
    axes[0].annotate(f"miss, L = {small['L_t']:.0f} km",
                     xy=(small["col"], small["row"]), xytext=(small["col"] + 8, small["row"] + 10),
                     fontsize=7, color="0.05",
                     arrowprops=dict(arrowstyle="-", color="0.05", lw=0.7))
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

    # --- figure 2: the decomposition for the matched pairs
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    s = 12 + 260 * (m["P_t"] / m["P_t"].max())
    sc = ax.scatter(m["dlnA"], m["dlnrho"], s=s, c=m["scale"],
                    cmap="viridis", norm=LogNorm(), alpha=0.85,
                    edgecolors="0.2", linewidths=0.5)
    for k, r in m.iterrows():
        ax.annotate(str(k + 1), (r["dlnA"], r["dlnrho"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    lim = 1.05 * max(np.abs(m["dlnA"]).max(), np.abs(m["dlnrho"]).max())
    ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
    xs = np.linspace(-lim, lim, 2)
    ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.2)
    ax.text(0.62 * lim, -0.52 * lim, "dlnP = 0\n(compensation)", fontsize=7,
            color="#b2182b", ha="center")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("dlnA  (area error)"); ax.set_ylabel(r"dln$\rho$  (intensity error)")
    ax.set_title("matched pairs: the exact split dlnP = dlnA + dln" + r"$\rho$", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("system scale L = √A (km)", fontsize=8)
    fig.savefig(OUT / "eval_method_decomp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    w = m["P_t"].to_numpy()
    print(dict(dlnA_w=float((w * m["dlnA"]).sum() / w.sum()),
               dlnrho_w=float((w * m["dlnrho"]).sum() / w.sum()),
               dlnP_w=float((w * m["dlnP"]).sum() / w.sum()),
               pod=len(m) / (len(m) + len(miss)),
               far=len(fa) / (len(m) + len(fa))))


if __name__ == "__main__":
    main()
