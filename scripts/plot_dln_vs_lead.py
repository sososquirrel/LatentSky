"""Log-additive decomposition vs forecast horizon, one curve per scale.

Companion to scripts/plot_scale_leads.py (which puts scale on the x-axis and
colour-codes lead). Here lead time is the x-axis and each system-scale bin
is its own curve, so you can read off how dlnP/dlnA/dlnrho drift with
horizon at a fixed scale.

Three panels, one per term of dlnP = dlnA + dlnrho:
  (a) dlnP_med vs lead_h, one line per scale
  (b) dlnA_med vs lead_h, one line per scale
  (c) dlnrho_med vs lead_h, one line per scale

Greedy IoU matching yields fewer matched systems at small scale + long lead
(as few as ~75 vs ~16k at lead 0 for the smallest bin in the 9var mse+uv
v3 re-eval) — medians over that few systems are noisy, not necessarily a
real signal. Line/marker alpha and marker size are scaled by n_match (log,
floored) so low-confidence stretches fade rather than being silently
dropped or shown at full weight.

Shaded bands (median ± IQR/2, i.e. the per-system P25-P75 spread within the
bin) are drawn for the smallest and largest scale only, so the noisiest and
the most-sampled ends of the size range are legible without cluttering the
mid-range curves. dlnA/dlnrho IQR come straight from the summary table;
dlnP has no precomputed IQR (only its median is stored), so it's derived
here from the per-system table (docs/scale_systems_v3.csv.gz).

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
from matplotlib.patches import Patch

N_MATCH_FLOOR = 20   # n_match at/below this renders at MIN_ALPHA
N_MATCH_CEIL = 2000  # n_match at/above this renders at full weight
MIN_ALPHA = 0.12
MIN_MS, MAX_MS = 3, 11
BAND_ALPHA = 0.18
Y_SHARED = (-2.3, 2.3)  # shared y-range across dlnP/dlnA/dlnrho panels


def _n_weight(n):
    """Map n_match -> [0, 1] on a log scale, clipped at the floor/ceiling."""
    n = np.clip(n, N_MATCH_FLOOR, N_MATCH_CEIL)
    return (np.log(n) - np.log(N_MATCH_FLOOR)) / (np.log(N_MATCH_CEIL) - np.log(N_MATCH_FLOOR))


def _dlnP_iqr_by_lead(repo, summary_path, predictor_set, g, scale_mid):
    """Per-lead dlnP IQR for one scale bin, from the per-system table.

    dlnP has no precomputed *_iqr column in the summary (only dlnA/dlnrho
    do), so recompute it from the matched-system rows the summary was
    itself built from, restricted to this one scale bin.
    """
    sys_path = pathlib.Path(str(summary_path).replace("scale_summary", "scale_systems"))
    if not sys_path.suffix == ".gz":
        gz = sys_path.with_suffix(sys_path.suffix + ".gz")
        if (repo / gz).exists():
            sys_path = gz
    if not (repo / sys_path).exists():
        print(f"note: no per-system table at {sys_path}, skipping dln P IQR band")
        return {}

    lo, hi = g.loc[g["scale_mid"] == scale_mid, ["scale_lo", "scale_hi"]].iloc[0]
    sysdf = pd.read_csv(repo / sys_path,
                        usecols=["predictor_set", "lead_h", "role", "scale", "dlnP"])
    m = sysdf[(sysdf["predictor_set"] == predictor_set) & (sysdf["role"] == "match")
              & (sysdf["scale"] >= lo) & (sysdf["scale"] <= hi)]
    return (m.groupby("lead_h")["dlnP"]
            .apply(lambda s: s.quantile(0.75) - s.quantile(0.25)).to_dict())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--out", default=None,
                   help="Default docs/figures/dln_vs_lead_<predictor>.png")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    df = pd.read_csv(repo / args.summary)

    g = df[df["predictor_set"] == args.predictor_set]
    if g.empty:
        opts = sorted(df["predictor_set"].unique())
        raise SystemExit(f"no rows for predictor set {args.predictor_set!r}; "
                         f"available: {opts}")

    scales = sorted(g["scale_mid"].unique())
    band_scales = {scales[0], scales[-1]}
    norm = LogNorm(vmin=min(scales), vmax=max(scales))
    cmap = cm.viridis

    panels = [
        ("dlnP_med", "dln P  —  total precip log-bias", "median dln P"),
        ("dlnA_med", "dln A  —  area (size) log-bias", "median dln A"),
        ("dlnrho_med", "dln ρ  —  intensity log-bias", "median dln ρ"),
    ]

    dlnP_iqr_by_scale = {
        scale: _dlnP_iqr_by_lead(repo, args.summary, args.predictor_set, g, scale)
        for scale in band_scales
    }

    n_legend_vals = [N_MATCH_FLOOR, 200, N_MATCH_CEIL]

    for col, title, ylab in panels:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        for scale in scales:
            sub = g[g["scale_mid"] == scale].sort_values("lead_h")
            sub = sub[sub[col].notna() & sub["n_match"].notna()]
            x = sub["lead_h"].to_numpy()
            y = sub[col].to_numpy()
            n = sub["n_match"].to_numpy()
            if len(x) < 1:
                continue
            base_rgba = np.array(cmap(norm(scale)))

            if scale in band_scales:
                if col == "dlnP_med":
                    iqr = np.array([dlnP_iqr_by_scale.get(scale, {}).get(lh, np.nan)
                                    for lh in x])
                else:
                    iqr_col = col.replace("_med", "_iqr")
                    iqr = sub[iqr_col].to_numpy()
                band = np.isfinite(iqr)
                if band.any():
                    ax.fill_between(x[band], y[band] - iqr[band] / 2,
                                    y[band] + iqr[band] / 2,
                                    color=base_rgba[:3], alpha=BAND_ALPHA,
                                    linewidth=0, zorder=1)

            if len(x) >= 2:
                pts = np.array([x, y]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                seg_w = _n_weight((n[:-1] + n[1:]) / 2)
                seg_colors = np.tile(base_rgba, (len(segs), 1))
                seg_colors[:, 3] = MIN_ALPHA + (1 - MIN_ALPHA) * seg_w
                ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=1.8))

            pt_w = _n_weight(n)
            pt_colors = np.tile(base_rgba, (len(x), 1))
            pt_colors[:, 3] = MIN_ALPHA + (1 - MIN_ALPHA) * pt_w
            sizes = (MIN_MS + (MAX_MS - MIN_MS) * pt_w) ** 2
            ax.scatter(x, y, s=sizes, color=pt_colors, edgecolors="none", zorder=3)

        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.margins(x=0.03)
        ax.autoscale_view(scaley=False)
        ax.set_ylim(*Y_SHARED)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(ylab)
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
            for nv in n_legend_vals
        ]
        n_handles.append(Patch(facecolor="0.3", alpha=BAND_ALPHA,
                                label="IQR/2 band\n(smallest & largest\nscale only)"))
        ax.legend(handles=n_handles, title="matched systems\n(size/opacity)",
                  loc="upper left", bbox_to_anchor=(1.18, 1.0),
                  fontsize=8, title_fontsize=8, framealpha=0.9)

        fig.tight_layout()

        safe_pred = args.predictor_set.split()[0]
        safe_col = col.replace("_med", "")
        default_name = f"docs/figures/dln_vs_lead_{safe_col}_{safe_pred}.png"
        out = (repo / (args.out.replace("dln_vs_lead", f"dln_vs_lead_{safe_col}")
                        if args.out else default_name)).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
