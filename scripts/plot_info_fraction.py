"""Agreement fraction vs lead time — a POD-like bounded (0,1] index.

dlnX = ln(pred/truth) passes through 0 exactly at perfect agreement, so map
it directly into a bounded index the same way POD is bounded (hits over
total):

    info_X(t, L) = exp(-|dlnX_med(t, L)|)  =  min(pred,truth) / max(pred,truth)

1 = perfect agreement, -> 0 as the bias grows. No reference lead time is
needed (unlike normalizing by the lead-6h value, which is itself noisy and
forces info(0)=1 even when there's already real bias at lead 0 — there
often is, e.g. dlnrho at the smallest scale is already ~-0.3 at lead 0).

Two panels: info_A(t,L) and info_rho(t,L), one line per scale, same
n_match alpha/size weighting convention as plot_dln_vs_lead.py.

Reads docs/scale_summary_v3.csv by default (latest 9var mse+uv re-eval).
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

N_MATCH_FLOOR = 20
N_MATCH_CEIL = 2000
MIN_ALPHA = 0.12
MIN_MS, MAX_MS = 3, 11
Y_SHARED = (0.0, 1.05)


def _n_weight(n):
    n = np.clip(n, N_MATCH_FLOOR, N_MATCH_CEIL)
    return (np.log(n) - np.log(N_MATCH_FLOOR)) / (np.log(N_MATCH_CEIL) - np.log(N_MATCH_FLOOR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--out", default=None,
                   help="Default docs/figures/info_frac_<quantity>_<predictor>.png")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo / args.summary)
    g = df[df["predictor_set"] == args.predictor_set]
    if g.empty:
        opts = sorted(df["predictor_set"].unique())
        raise SystemExit(f"no rows for predictor set {args.predictor_set!r}; "
                         f"available: {opts}")

    scales = sorted(g["scale_mid"].unique())
    norm = LogNorm(vmin=min(scales), vmax=max(scales))
    cmap = cm.viridis

    panels = [
        ("dlnA_med", "info_A  =  exp(-|dln A|)  —  area agreement"),
        ("dlnrho_med", "info_ρ  =  exp(-|dln ρ|)  —  intensity agreement"),
    ]

    for col, title in panels:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        for scale in scales:
            sub = g[g["scale_mid"] == scale].sort_values("lead_h")
            sub = sub[sub[col].notna() & sub["n_match"].notna()]
            t = sub["lead_h"].to_numpy()
            n = sub["n_match"].to_numpy()
            info = np.exp(-np.abs(sub[col].to_numpy()))
            if len(t) < 1:
                continue
            base_rgba = np.array(cmap(norm(scale)))

            if len(t) >= 2:
                pts = np.array([t, info]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                seg_w = _n_weight((n[:-1] + n[1:]) / 2)
                seg_colors = np.tile(base_rgba, (len(segs), 1))
                seg_colors[:, 3] = MIN_ALPHA + (1 - MIN_ALPHA) * seg_w
                ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=1.8))

            pt_w = _n_weight(n)
            pt_colors = np.tile(base_rgba, (len(t), 1))
            pt_colors[:, 3] = MIN_ALPHA + (1 - MIN_ALPHA) * pt_w
            sizes = (MIN_MS + (MAX_MS - MIN_MS) * pt_w) ** 2
            ax.scatter(t, info, s=sizes, color=pt_colors, edgecolors="none", zorder=3)

        ax.axhline(1, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.35)
        ax.margins(x=0.03)
        ax.autoscale_view(scaley=False)
        ax.set_ylim(*Y_SHARED)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel("agreement  (1 = perfect, 0 = none)")
        ax.set_title(f"{title} — {args.predictor_set}")
        ax.grid(True, alpha=0.3)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("system scale L = √A  (km)")

        n_handles = [
            Line2D([], [], marker="o", linestyle="none", color="0.3",
                   alpha=MIN_ALPHA + (1 - MIN_ALPHA) * _n_weight(nv),
                   markersize=(MIN_MS + (MAX_MS - MIN_MS) * _n_weight(nv)),
                   label=f"n_match {'≤' if nv == N_MATCH_FLOOR else ('≥' if nv == N_MATCH_CEIL else '~')}{nv}")
            for nv in [N_MATCH_FLOOR, 200, N_MATCH_CEIL]
        ]
        ax.legend(handles=n_handles, title="matched systems\n(size/opacity)",
                  loc="upper left", bbox_to_anchor=(1.18, 1.0),
                  fontsize=8, title_fontsize=8, framealpha=0.9)

        fig.tight_layout()

        safe_pred = args.predictor_set.split()[0]
        safe_col = col.replace("_med", "").replace("dln", "")
        default_name = f"docs/figures/info_frac_{safe_col}_{safe_pred}.png"
        out = (repo / (args.out.replace("info_frac", f"info_frac_{safe_col}")
                        if args.out else default_name)).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
