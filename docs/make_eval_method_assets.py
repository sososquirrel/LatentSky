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

    # --- figure 1: the two fields with detected systems, roles and pair ids
    vmax = max(truth.max(), det.max(), 0.5)
    norm = LogNorm(vmin=THR, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))

    def outline(ax, labels, label_id, color, ls="-", lw=1.4):
        ax.contour(labels == label_id, levels=[0.5], colors=[color],
                   linewidths=lw, linestyles=ls)

    def find_label(labels, row, col):
        return labels[int(round(row)), int(round(col))]

    for ax, f, ttl in [(axes[0], truth, "truth (+3 h)"),
                       (axes[1], det, "two-head (+3 h)")]:
        ax.imshow(f, origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for k, r in m.iterrows():
        lid_t = find_label(lt, r["row"], r["col"])
        lid_p = find_label(lp, r["row_p"], r["col_p"])
        outline(axes[0], lt, lid_t, "#2166ac")
        outline(axes[1], lp, lid_p, "#2166ac")
        for ax, rr, cc in [(axes[0], r["row"], r["col"]),
                           (axes[1], r["row_p"], r["col_p"])]:
            ax.text(cc, rr, str(k + 1), fontsize=7, fontweight="bold",
                    color="#12335c", ha="center", va="center")
    for _, r in miss.iterrows():
        outline(axes[0], lt, find_label(lt, r["row"], r["col"]), "#e08214", lw=1.6)
    for _, r in fa.iterrows():
        outline(axes[1], lp, find_label(lp, r["row"], r["col"]), "#e08214",
                ls="--", lw=1.6)
    fig.suptitle(f"cluster detection (>{THR} mm, 8-connected) and greedy IoU ≥ {IOU_MIN} "
                 f"matching — {len(m)} matched pairs (blue, numbered), "
                 f"{len(miss)} misses / {len(fa)} false alarms (amber)", fontsize=9)
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
