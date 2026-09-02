"""tau(L) scaling law, one panel per model, each overlaid with persistence
for direct reference -- same fit method/markers as
plot_pod_tau_track_compare.py (marker color = actual POD at that scale
bin's earliest available lead), split into separate panels per model.

Bins: the standard log-spaced 16-bin scheme (default_scale_edges) used
throughout this session's tau(L) work, read from the pre-aggregated
rebinned16 summary CSVs, capped at --max-km (default 750km) -- beyond
that, truth systems are too sparse per lead for a reliable per-bin POD
(see the many-to-one/decomposition investigation earlier this session).

Equal-count (quantile) bins were tried here first and discarded: object-
size distributions are heavily right-skewed (many small systems, very few
large), so equal-POPULATION bins pack almost all of them into a tiny
small-scale sliver and leave terrible coverage of the rest of the range --
fine for a per-bin median (the decomposition plots), bad for a power-law
fit that needs the L-range sampled evenly in log-space. R2 dropped from
0.93-0.96 (log-spaced) to 0.44-0.54 (equal-count) on this exact data.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latentsky.fits import fit_exp_decay, fit_powerlaw
from latentsky.paths import FIGURES_DIR
from latentsky.registry import REPO_ROOT as REPO

_SIX_LEADS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]

MODELS = [
    dict(label="regression (new split)",
         summary="docs/scale_summary_onehead_patchsplit_rebinned16.csv",
         runs=[f"patchsplit_onehead_lead{L}" for L in _SIX_LEADS], color="tab:purple", marker="o"),
    dict(label="two-head",
         summary="docs/scale_summary_modelcompare_rebinned16.csv",
         runs=[f"patchsplit_twohead_olr_fss_lead{L}" for L in _SIX_LEADS], color="tab:orange", marker="s"),
    dict(label="two-head+diffusion",
         summary="docs/scale_summary_diffusion_manytoone_rebinned16.csv",
         runs=[f"diffusion_lead{L}" for L in _SIX_LEADS], color="tab:green", marker="D"),
    dict(label="two-head+diffusion (plain-regression warm-start)",
         summary="docs/scale_summary_diffusion_oneheadbackbone_rebinned16.csv",
         runs=[f"diffusion_lead{L}" for L in _SIX_LEADS], color="tab:red", marker="P"),
]
PERSISTENCE = dict(label="persistence", summary="docs/scale_summary_baselines_patchsplit.csv",
                    runs=[f"persistence_lead{L}" for L in _SIX_LEADS], color="gray", marker="v")

pod_cmap = plt.get_cmap("RdYlGn")
pod_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)


def load_and_bin(spec, max_km):
    """Pre-aggregated log-spaced rebinned16 summary, filtered to this
    source's runs and capped at max_km."""
    df = pd.read_csv(REPO / spec["summary"])
    df = df[df["name"].isin(spec["runs"]) & (df["scale_mid"] <= max_km)]
    return df


def powerlaw_crossover(pl_a, pl_b):
    """Scale L where two tau=tau_ref*(L/L_ref)^beta power-law fits cross,
    solved analytically (a straight line in log-log space for each), or
    None if their slopes are ~equal (no crossover, lines stay parallel)."""
    if pl_a["beta"] == pl_b["beta"]:
        return None
    ln_a = np.log(pl_a["tau_ref"]) - pl_a["beta"] * np.log(pl_a["L_ref"])
    ln_b = np.log(pl_b["tau_ref"]) - pl_b["beta"] * np.log(pl_b["L_ref"])
    lnL = (ln_b - ln_a) / (pl_a["beta"] - pl_b["beta"])
    return float(np.exp(lnL))


def fit_track(df):
    Ls, taus, pod0s = [], [], []
    for L, g in df.groupby("scale_mid"):
        g = g.sort_values("lead_h")
        if len(g) < 2 or g["POD"].iloc[0] <= 0:
            continue
        p0 = g["POD"].iloc[0]
        norm = g["POD"].to_numpy() / p0
        fit = fit_exp_decay(g["lead_h"].to_numpy(), norm)
        if np.isnan(fit["tau"]):
            continue
        Ls.append(L)
        taus.append(fit["tau"])
        pod0s.append(p0)
    return np.array(Ls), np.array(taus), np.array(pod0s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-km", type=float, default=750.0,
                    help="drop scale bins above this -- too sparse per lead beyond here")
    p.add_argument("--out", default=str(FIGURES_DIR / "2026-07-22_full_comparison/track_compare_4panel_vs_persistence.png"))
    args = p.parse_args()

    pr_df = load_and_bin(PERSISTENCE, args.max_km)
    pr_L, pr_tau, pr_pod0 = fit_track(pr_df)
    pr_pl = fit_powerlaw(pr_L, pr_tau) if len(pr_L) >= 2 else None

    pod_results = [dict(label=PERSISTENCE["label"], color=PERSISTENCE["color"],
                       marker=PERSISTENCE["marker"], L=pr_L, pod0=pr_pod0, tau=pr_tau)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    for ax, spec in zip(axes.ravel(), MODELS):
        df = load_and_bin(spec, args.max_km)
        L, tau, pod0 = fit_track(df)
        pl = fit_powerlaw(L, tau) if len(L) >= 2 else None
        pod_results.append(dict(label=spec["label"], color=spec["color"],
                               marker=spec["marker"], L=L, pod0=pod0, tau=tau))

        # persistence reference, same panel
        ax.scatter(pr_L, pr_tau, c=pr_pod0, cmap=pod_cmap, norm=pod_norm,
                  marker=PERSISTENCE["marker"], s=70, edgecolors="k", linewidths=0.6,
                  alpha=0.55, zorder=3, label="persistence")
        if pr_pl and np.isfinite(pr_pl["beta"]):
            xs = np.geomspace(pr_L.min(), pr_L.max(), 50)
            ys = pr_pl["tau_ref"] * (xs / pr_pl["L_ref"]) ** pr_pl["beta"]
            ax.plot(xs, ys, ":", color="0.4", lw=1.5, alpha=0.8, zorder=1,
                    label=f"persistence: beta={pr_pl['beta']:.2f}")

        # this model
        ax.scatter(L, tau, c=pod0, cmap=pod_cmap, norm=pod_norm,
                  marker=spec["marker"], s=80, edgecolors="k", linewidths=0.6,
                  zorder=4, label=spec["label"])
        if pl and np.isfinite(pl["beta"]):
            xs = np.geomspace(L.min(), L.max(), 50)
            ys = pl["tau_ref"] * (xs / pl["L_ref"]) ** pl["beta"]
            ax.plot(xs, ys, "--", color=spec["color"], lw=1.8, zorder=2,
                    label=f"{spec['label']}: beta={pl['beta']:.2f} (R2={pl['R2']:.2f})")

        ax.set_xscale("log")
        ax.set_yscale("log")

        if pl and pr_pl and np.isfinite(pl["beta"]) and np.isfinite(pr_pl["beta"]):
            Lx = powerlaw_crossover(pl, pr_pl)
            all_L = np.concatenate([L, pr_L])
            if Lx is not None and all_L.min() <= Lx <= all_L.max():
                ax.axvline(Lx, color="0.3", lw=1.2, ls=":", zorder=1)
                ax.text(Lx, 0.03, f"  {Lx:.0f} km", transform=ax.get_xaxis_transform(),
                       rotation=90, fontsize=7.5, color="0.3", va="bottom", ha="center")
        ax.set_title(spec["label"], fontsize=10.5)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, frameon=False, loc="upper left")

    for ax in axes[-1]:
        ax.set_xlabel("scale L (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("tau (h)")

    sm = matplotlib.cm.ScalarMappable(norm=pod_norm, cmap=pod_cmap)
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("POD at earliest available lead\n(the tau fit's own normalization anchor)")
    fig.suptitle(f"tau(L) scaling law -- each model vs persistence (gray, faded)\n"
                 f"log-spaced 16-bin scheme, capped at {args.max_km:g}km", y=0.995)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    # ---- companion plot: POD(L) and POD(tau), all tracks together, one
    # color per track. Left: raw detection skill each model starts with,
    # by scale (no tau/decay involved). Right: the SAME anchor-lead POD
    # values, but plotted against each bin's fitted predictability
    # timescale instead of its scale -- since tau itself is monotonic in
    # L for every track, this looks similar in shape, but it directly
    # answers "does a longer predictability window correlate with higher
    # starting detection skill across ALL tracks/scales pooled together,"
    # independent of scale binning.
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))
    for r in pod_results:
        order = np.argsort(r["L"])
        ax2.plot(r["L"][order], r["pod0"][order], "-", color=r["color"], lw=1.3, alpha=0.6, zorder=2)
        ax2.scatter(r["L"], r["pod0"], color=r["color"], marker=r["marker"], s=80,
                   edgecolors="k", linewidths=0.6, zorder=3, label=r["label"])

        order_p = np.argsort(r["pod0"])
        ax3.plot(r["pod0"][order_p], r["tau"][order_p], "-", color=r["color"], lw=1.3, alpha=0.6, zorder=2)
        ax3.scatter(r["pod0"], r["tau"], color=r["color"], marker=r["marker"], s=80,
                   edgecolors="k", linewidths=0.6, zorder=3, label=r["label"])

    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("scale L (km)")
    ax2.set_ylabel("POD at earliest available lead")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.set_title("POD(L)")

    ax3.set_yscale("log")
    ax3.set_xlim(0, 1.05)
    ax3.set_xlabel("POD at earliest available lead")
    ax3.set_ylabel("tau (h)")
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=8, frameon=False, loc="upper left")
    ax3.set_title("tau(POD)")

    fig2.suptitle(f"POD vs scale and vs predictability timescale, all tracks -- capped at {args.max_km:g}km")
    out_pod = args.out.replace("4panel_vs_persistence.png", "pod_vs_scale_all_models.png")
    fig2.savefig(out_pod, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pod}")


if __name__ == "__main__":
    main()
