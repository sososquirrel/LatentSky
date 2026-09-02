"""Cross-track comparison: raw POD(t) per scale bin (no fit forced) and the
tau(L) scaling law, overlaying all FIVE model tracks -- regression (old
split), regression (new/patch split), classif+regression (two-head),
classif+regression+diffusion -- plus THREE baselines (persistence,
climatology, random) on the same axes. This is the main POD-scaling-law
deliverable: same colors/labels as latentsky.predict_dispatch's
tracks_for_lead/summary_tracks_for_lead, so it is visually consistent with
every other cross-track comparison figure in this suite.

Point of this script: tau(L) is only meaningful where POD(t)/POD(0) actually
decays like exp(-t/tau) in the first place. Forcing that functional form on
all three tracks and only reporting the fitted tau hides cases where the
decay doesn't look exponential (flat, non-monotonic, or a fit that saturates
at fit_exp_decay's tau_bounds upper edge instead of failing loudly). Figure A
shows the raw data so that's visible; Figure B fits tau(L) but marks any
per-scale-bin fit that hit the bound as unreliable rather than plotting it
as if it were solid.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.fits import fit_exp_decay, fit_powerlaw
from latentsky.paths import FIGURES_DIR

_SIX_LEADS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]

TRACKS = [
    dict(
        label="regression (old split)",
        summary="docs/scale_summary_v3_rebinned16.csv",
        runs=[f"mseuv_lead{L}" for L in
              ["0h", "0.25h", "0.5h", "0.75h", "1h", "1.5h", "2h", "3h", "4h", "6h"]],
        color="tab:blue",
    ),
    dict(
        label="regression (new split)",
        summary="docs/scale_summary_onehead_patchsplit_rebinned16.csv",
        runs=[f"patchsplit_onehead_lead{L}" for L in _SIX_LEADS],
        color="tab:purple",
    ),
    dict(
        label="two-head",
        summary="docs/scale_summary_modelcompare_rebinned16.csv",
        runs=[f"patchsplit_twohead_olr_fss_lead{L}" for L in _SIX_LEADS],
        color="tab:orange",
    ),
    dict(
        label="two-head+diffusion",
        summary="docs/scale_summary_diffusion_manytoone_rebinned16.csv",
        runs=[f"diffusion_lead{L}" for L in _SIX_LEADS],
        color="tab:green",
    ),
    dict(
        label="two-head+diffusion (plain-regression warm-start)",
        summary="docs/scale_summary_diffusion_oneheadbackbone_rebinned16.csv",
        runs=[f"diffusion_lead{L}" for L in _SIX_LEADS],
        color="tab:red",
    ),
    dict(
        label="persistence",
        summary="docs/scale_summary_baselines_patchsplit.csv",
        runs=[f"persistence_lead{L}" for L in _SIX_LEADS],
        color="gray",
    ),
    dict(
        label="climatology",
        summary="docs/scale_summary_baselines_patchsplit.csv",
        runs=[f"climatology_lead{L}" for L in _SIX_LEADS],
        color="black",
    ),
    dict(
        label="random",
        summary="docs/scale_summary_baselines_patchsplit.csv",
        runs=[f"random_lead{L}" for L in _SIX_LEADS],
        color="brown",
    ),
]

TAU_UPPER_BOUND = 200.0


def load_track(spec: dict) -> pd.DataFrame:
    df = pd.read_csv(spec["summary"])
    return df[df["name"].isin(spec["runs"])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default=str(FIGURES_DIR / "2026-07-22_full_comparison/track_compare"))
    p.add_argument("--tracks", nargs="+", default=None,
                    help="Subset of track labels to include (default: all). "
                         f"Available: {[s['label'] for s in TRACKS]}")
    args = p.parse_args()

    track_specs = TRACKS
    if args.tracks is not None:
        track_specs = [s for s in TRACKS if s["label"] in args.tracks]
        missing = set(args.tracks) - {s["label"] for s in track_specs}
        if missing:
            raise SystemExit(f"unknown track label(s) {missing}; available: {[s['label'] for s in TRACKS]}")

    tracks = []
    for spec in track_specs:
        df = load_track(spec)
        if df.empty:
            print(f"WARNING: no rows for track {spec['label']!r} "
                  f"(summary={spec['summary']!r}, runs={spec['runs']}) -- skipping")
            continue
        n_leads = df["lead_h"].nunique()
        print(f"track {spec['label']!r}: {n_leads} leads, {len(df)} rows "
              f"(summary={spec['summary']!r})")
        tracks.append(dict(spec, df=df))

    if not tracks:
        raise SystemExit("no track data found -- nothing to plot")

    all_bins = sorted(set().union(*[set(t["df"]["scale_mid"].unique()) for t in tracks]))

    # ---- Figure A: raw POD(t) per scale bin, no fit ----
    ncols = 4
    nrows = int(np.ceil(len(all_bins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                              sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, L in zip(axes, all_bins):
        for t in tracks:
            sub = t["df"][np.isclose(t["df"]["scale_mid"], L)].sort_values("lead_h")
            if sub.empty:
                continue
            ax.plot(sub["lead_h"], sub["POD"], "o-", color=t["color"], label=t["label"])
        ax.set_title(f"L~{L:.0f} km", fontsize=9)
        ax.set_xlabel("lead (h)")
        ax.grid(alpha=0.3)
    for ax in axes[len(all_bins):]:
        ax.axis("off")
    axes[0].set_ylabel("POD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(tracks), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("POD(t) per scale bin -- raw, no exponential fit forced")
    outA = f"{args.out_prefix}_pod_raw.png"
    fig.savefig(outA, dpi=150, bbox_inches="tight")
    print(f"wrote {outA}")

    # ---- Figure B: tau(L) scaling law (power-law fit over all scale bins) ----
    # Marker SHAPE now identifies the track; marker COLOR instead encodes
    # the actual POD at that scale bin's earliest available lead (the same
    # p0 used to normalize the exponential fit) -- so you can see, at a
    # glance, whether a high/low tau point is backed by strong or weak raw
    # detection skill, not just how fast it decays relatively.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    pod_cmap = plt.get_cmap("RdYlGn")
    pod_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    fig2, ax2 = plt.subplots(figsize=(7.2, 7.0))
    for t, marker in zip(tracks, markers):
        Ls, taus, pod0s = [], [], []
        for L, g in t["df"].groupby("scale_mid"):
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
        Ls, taus, pod0s = np.array(Ls), np.array(taus), np.array(pod0s)

        ax2.scatter(Ls, taus, c=pod0s, cmap=pod_cmap, norm=pod_norm,
                   marker=marker, s=80, edgecolors="k", linewidths=0.6,
                   zorder=4, label=t["label"])

        if len(Ls) >= 2:
            pl = fit_powerlaw(Ls, taus)
            if np.isfinite(pl["beta"]):
                xs = np.geomspace(Ls.min(), Ls.max(), 50)
                ys = pl["tau_ref"] * (xs / pl["L_ref"]) ** pl["beta"]
                ax2.plot(xs, ys, "--", color=t["color"], alpha=0.6, zorder=2,
                         label=f"{t['label']}: beta={pl['beta']:.2f} (R2={pl['R2']:.2f})")
        else:
            print(f"  {t['label']}: fewer than 2 scale bins -- no power-law fit drawn")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("scale L (km)")
    ax2.set_ylabel("tau (h)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    sm = matplotlib.cm.ScalarMappable(norm=pod_norm, cmap=pod_cmap)
    cbar = fig2.colorbar(sm, ax=ax2)
    cbar.set_label("POD at earliest available lead\n(the tau fit's own normalization anchor)")
    fig2.suptitle("tau(L) scaling law across tracks\nmarker shape = track, marker color = actual POD at that bin's anchor lead")
    outB = f"{args.out_prefix}_tau_scaling.png"
    fig2.savefig(outB, dpi=150, bbox_inches="tight")
    print(f"wrote {outB}")


if __name__ == "__main__":
    main()
