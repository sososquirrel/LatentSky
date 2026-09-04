"""Pair recognition for GraphCast at +6 h (2020-08-15 00 UTC init, valid 06 UTC):
truth (ERA5) vs forecast over the full tropical belt, plus a zoom on the most
active 60-degree window. Conventions: matched pairs share a color across panels
(filled overlay on the belt, outlines on the zoom), misses black, false alarms
red. Also prints the POD/FAR-by-scale threshold sensitivity for GraphCast at
0.1 / 1.0 / 2.4 mm per 6 h (2.4 = LatentSky's 0.1 mm/15 min rate equivalent).
Uses the cached fetch of i1_pod6h_august.py."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from latentsky.metrics import cell_area_km2, label_systems, match_systems

REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = REPO / "wb2_benchmark"
IOU_MIN, THR = 0.1, 2.4  # rate-equivalent threshold (see sensitivity)
BANDS = np.geomspace(30, 3000, 11)

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def lut(table):
    return {(round(a, 6), round(b, 6)): int(l)
            for a, b, l in zip(table["row"], table["col"], table["label"])}


def main():
    ds = xr.open_dataset(CACHE / "wb2_6h_graphcast_2020-08-15.nc")
    truth = ds.truth.values[0] * 1000.0        # m -> mm
    fc = ds.forecast.values[0] * 1000.0
    ny, nx = truth.shape
    area_row = cell_area_km2(ny, nx, lat_extent=(float(ds.lat[0]) - 0.125,
                                                 float(ds.lat[-1]) + 0.125))

    lt, tt = label_systems(truth, THR, area_row)
    lp, tp = label_systems(fc, THR, area_row)
    mm = match_systems(truth, fc, THR, area_row, iou_min=IOU_MIN)
    lut_t, lut_p = lut(tt), lut(tp)

    matches = mm[mm.role == "match"]
    misses = mm[mm.role == "miss"]
    fas = mm[mm.role == "false_alarm"]
    npod = len(matches) / (len(matches) + len(misses))
    nfar = len(fas) / (len(matches) + len(fas))
    print(f"systems: truth {len(tt)}, forecast {len(tp)}; match {len(matches)}, "
          f"miss {len(misses)}, FA {len(fas)}; POD {npod:.3f} FAR {nfar:.3f}")

    # pair colors, biggest first
    order = matches.sort_values("A_t", ascending=False)
    t20 = plt.get_cmap("tab20")
    pair_colors = [t20(i) for i in range(20) if i not in (6, 7)]
    color_of_t, color_of_p = {}, {}
    for k, (_, rr) in enumerate(order.iterrows()):
        col = pair_colors[k % len(pair_colors)]
        color_of_t[lut_t[(round(rr["row"], 6), round(rr["col"], 6))]] = col
        color_of_p[lut_p[(round(rr["row_p"], 6), round(rr["col_p"], 6))]] = col
    miss_ids = {lut_t[(round(r["row"], 6), round(r["col"], 6))]
                for _, r in misses.iterrows()}
    fa_ids = {lut_p[(round(r["row"], 6), round(r["col"], 6))]
              for _, r in fas.iterrows()}

    def overlay(lab, color_of, other_ids, other_col):
        rgba = np.zeros((*lab.shape, 4), np.float32)
        for lid, col in color_of.items():
            rgba[lab == lid] = (*col[:3], 0.9)
        for lid in other_ids:
            rgba[lab == lid] = (*other_col, 0.55)
        return rgba

    ov_t = overlay(lt, color_of_t, miss_ids, (0.05, 0.05, 0.05))
    ov_p = overlay(lp, color_of_p, fa_ids, (0.84, 0.15, 0.16))

    # most active 60-degree window (240 px) by truth precip volume
    vol = (truth * area_row[:, None]).sum(0)
    W = 240
    csum = np.concatenate([[0], np.cumsum(vol)])
    j0 = int(np.argmax(csum[W:] - csum[:-W]))
    zoom = np.s_[:, j0:j0 + W]
    lon0, lon1 = float(ds.lon[j0]), float(ds.lon[j0 + W - 1])

    norm = LogNorm(vmin=THR, vmax=max(truth.max(), fc.max()))
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_under("#ffffff")

    fig = plt.figure(figsize=(13.6, 9.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.55, 1.55], hspace=0.28)
    panels = [("truth (ERA5, valid 2020-08-15 06 UTC)", truth, ov_t, None),
              ("GraphCast +6 h", fc, ov_p, None),
              (f"truth — zoom {lon0:.0f}–{lon1:.0f}°E", truth, ov_t, zoom),
              ("GraphCast +6 h — zoom", fc, ov_p, zoom)]
    for ax_i, (ttl, f, ov, z) in enumerate(panels):
        ax = fig.add_subplot(gs[ax_i])
        ff, oo = (f, ov) if z is None else (f[z], ov[z])
        ax.imshow(ff, origin="lower", norm=norm, cmap=cmap, alpha=0.45,
                  interpolation="nearest", aspect="auto")
        ax.imshow(oo, origin="lower", interpolation="nearest", aspect="auto")
        if z is not None:
            labz = (lt if "truth" in ttl else lp)[z]
            ids = (color_of_t if "truth" in ttl else color_of_p)
            for lid, col in ids.items():
                if (labz == lid).any():
                    ax.contour(labz == lid, levels=[0.5], colors=[col],
                               linewidths=1.4)
        else:
            ax.axvspan(j0, j0 + W, color="none", ec="0.2", lw=1.2, ls="--")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[
        Line2D([], [], color=pair_colors[0], lw=4,
               label=f"matched pair — same color both panels ({len(matches)})"),
        Line2D([], [], color="0.05", lw=4, alpha=0.55,
               label=f"miss ({len(misses)})"),
        Line2D([], [], color="#d62728", lw=4, alpha=0.55,
               label=f"false alarm ({len(fas)})")],
        loc="lower center", ncol=3, fontsize=8.5, frameon=False,
        bbox_to_anchor=(0.5, 0.055))
    fig.suptitle(f"GraphCast pair recognition, +6 h, tropics — POD {npod:.2f}, "
                 f"FAR {nfar:.2f} (threshold {THR} mm/6 h, IoU ≥ {IOU_MIN})",
                 fontsize=10.5, y=0.945)
    fig.savefig(OUT / "fig_pairs_graphcast.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_pairs_graphcast.png")

    # ---- threshold sensitivity: POD/FAR by band at 0.1 / 1.0 / 2.4 mm per 6 h
    print("\nthreshold sensitivity (GraphCast, both inits of 2020-08-15):")
    import pandas as pd
    mids = np.sqrt(BANDS[:-1] * BANDS[1:])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    for thr, colr in [(0.1, "#1b7837"), (1.0, "#b05e0a"), (2.4, "#b2182b")]:
        rows = []
        for i in range(ds.init.size):
            yt = ds.truth.values[i] * 1000.0
            pf = ds.forecast.values[i] * 1000.0
            rows.append(match_systems(yt, pf, thr, area_row, iou_min=IOU_MIN))
        df = pd.concat(rows, ignore_index=True)
        b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
        pod = np.full(len(mids), np.nan)
        far = np.full(len(mids), np.nan)
        ns = np.zeros(len(mids), int)
        for k in range(len(mids)):
            g = df[(b == k) & df.role.isin(["match", "miss"])]
            gf = df[(b == k) & df.role.isin(["match", "false_alarm"])]
            if len(g):
                pod[k] = (g.role == "match").mean(); ns[k] = len(g)
            if len(gf):
                far[k] = (gf.role == "false_alarm").mean()
        ok = ns >= 20
        axs[0].plot(mids[ok], pod[ok], "o-", ms=4, color=colr,
                    label=f"{thr} mm/6 h")
        axs[1].plot(mids[ok], far[ok], "s--", ms=4, color=colr)
        nm = (df.role == "match").sum(); nmi = (df.role == "miss").sum()
        nfa = (df.role == "false_alarm").sum()
        print(f"  thr {thr:>4} mm/6h: truth systems {nm+nmi:6d}, POD "
              f"{nm/(nm+nmi):.3f}, FAR {nfa/(nm+nfa):.3f}")
    for ax, t in [(axs[0], "POD"), (axs[1], "FAR")]:
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_xlabel("system scale L = √A (km)"); ax.set_title(t + " (+6 h)",
                                                                fontsize=9)
    axs[0].set_ylabel("object-based rate")
    axs[0].legend(fontsize=8, frameon=False, title="wet threshold",
                  title_fontsize=8, loc="upper left")
    fig.suptitle("GraphCast: threshold sensitivity (2.4 mm/6 h = LatentSky's "
                 "0.1 mm/15 min rate equivalent)", fontsize=9.5)
    fig.savefig(OUT / "fig_thr_sensitivity_graphcast.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig_thr_sensitivity_graphcast.png")


if __name__ == "__main__":
    main()
