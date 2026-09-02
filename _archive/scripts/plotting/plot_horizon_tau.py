"""Headline figure: predictability horizon tau(L), defined as the first lead
time at which |ln(pred/truth)| crosses ln(2) (a factor-of-2 error), computed
independently per scale bin / component (area, intensity, total precip) /
model track -- replacing the 3x3 dln_3panel grid as the punchline figure.

The point: which horizon you report depends entirely on which functional you
score. Area and intensity each cross factor-of-2 error on their own,
component-specific schedule; total precip (P = rho*A) can look good for far
longer than either component alone whenever area/intensity errors have
opposite sign and partially cancel in the product -- an "illusory" horizon
that overstates real predictability. tau_upper = min(tau_area, tau_intensity)
per scale bin is a defensible, quotable upper bound on how far the model's
structure is actually trustworthy, as opposed to how far the aggregate
metric happens to still look fine.

tau is found by linear interpolation of the lead time at which the median
|dlnX| curve crosses ln(2) (see latentsky.fits.first_crossing_tau); if it
never crosses within the observed lead range (0.25-6h), that's reported as
right-censored (tau > 6h, drawn as an upward arrow -- matplotlib's lolims
convention -- not a real crossing). CIs are a nonparametric bootstrap over
the raw per-system dln values at each lead (latentsky.fits.
bootstrap_horizon_tau), not just the point-estimate median curve -- plain
medians alone don't support a horizon claim, especially for the sparsest
bins (e.g. 450-700km, n~2,900 total across all 6 leads).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.fits import bootstrap_horizon_tau
from latentsky.paths import FIGURES_DIR
from latentsky.registry import REPO_ROOT as REPO

_LEADS = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0]
_LEAD_LABELS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]

TRACK_PRESETS = {
    "onehead": dict(
        systems="docs/scale_systems_onehead_patchsplit.csv.gz",
        runs=[f"patchsplit_onehead_lead{L}" for L in _LEAD_LABELS],
        label="regression (new split)", color="tab:purple"),
    "two-head": dict(
        systems="docs/scale_systems_modelcompare.csv.gz",
        runs=[f"patchsplit_twohead_olr_fss_lead{L}" for L in _LEAD_LABELS],
        label="two-head", color="tab:orange"),
    "diffusion": dict(
        systems="docs/scale_systems_diffusion.csv.gz",
        runs=[f"diffusion_lead{L}" for L in _LEAD_LABELS],
        label="two-head+diffusion", color="tab:green"),
}

DEFAULT_EDGES = [60, 62.0, 71.4, 80, 100, 140, 200, 300, 450, 700]
COMPONENTS = [("dlnA", "area", "-", "o"), ("dlnrho", "intensity", "--", "s"),
              ("dlnP", "total precip (illusory)", ":", "^")]
THRESH = float(np.log(2.0))


def _load(spec, edges, floor_km, repo):
    df = pd.read_csv(repo / spec["systems"])
    df = df[df["name"].isin(spec["runs"]) & (df["role"] == "match")].copy()
    df = df[(df["scale"] >= max(floor_km, edges[0])) & (df["scale"] < edges[-1])]
    b = np.clip(np.digitize(df["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    df["bin_idx"] = b
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", nargs="+", default=["onehead", "two-head", "diffusion"],
                    choices=list(TRACK_PRESETS))
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--floor-km", type=float, default=60.0)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--out", default=str(FIGURES_DIR / "horizon_tau.png"))
    args = p.parse_args()

    edges = np.array(args.edges)
    n_bins = len(edges) - 1
    mids = np.sqrt(edges[:-1] * edges[1:])  # geometric-mean bin midpoint, for log-x placement

    tracks = [TRACK_PRESETS[n] for n in args.tracks]
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.2 * len(tracks), 6.0),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, spec in zip(axes, tracks):
        df = _load(spec, edges, args.floor_km, REPO)
        comp_results = {}
        for col, comp_label, ls, marker in COMPONENTS:
            taus, los, his, cens_flags = [], [], [], []
            for bi in range(n_bins):
                sub = df[df["bin_idx"] == bi]
                raw_by_lead = [sub.loc[sub["lead_h"] == L, col].to_numpy() for L in _LEADS]
                res = bootstrap_horizon_tau(_LEADS, raw_by_lead, thresh=THRESH,
                                             n_boot=args.n_boot, seed=bi)
                taus.append(res["tau"])
                los.append(res["lo"])
                his.append(res["hi"])
                cens_flags.append(res["frac_censored"] > 0.5)
            taus, los, his = np.array(taus), np.array(los), np.array(his)
            cens_flags = np.array(cens_flags)
            comp_results[col] = dict(taus=taus, los=los, his=his, cens=cens_flags)

            ok = np.isfinite(taus)
            yerr_lo = np.clip(taus - los, 0, None)
            yerr_hi = np.clip(his - taus, 0, None)
            ax.errorbar(mids[ok], taus[ok], yerr=[yerr_lo[ok], yerr_hi[ok]],
                        lolims=cens_flags[ok], fmt=marker + ls, color=spec["color"],
                        alpha=0.55 if col == "dlnP" else 1.0,
                        lw=2.2 if col != "dlnP" else 1.6,
                        ms=7, capsize=3, label=comp_label, zorder=3 if col != "dlnP" else 2)

        # tau_upper = min(tau_area, tau_intensity) per bin -- bold overlay
        a, r = comp_results["dlnA"]["taus"], comp_results["dlnrho"]["taus"]
        upper = np.nanmin(np.stack([a, r]), axis=0)
        ok = np.isfinite(upper)
        ax.plot(mids[ok], upper[ok], "-", color="k", lw=3.0, alpha=0.85, zorder=4,
                label="tau_upper = min(area, intensity)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(edges[0] * 0.85, edges[-1] * 1.2)
        ax.set_ylim(0.2, 9.0)
        ax.axhline(6.0, color="0.5", lw=0.8, ls=":", zorder=1)
        ax.set_title(spec["label"], fontsize=11.5)
        ax.set_xlabel("scale L (km)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, frameon=False, loc="lower right")

    axes[0].set_ylabel("horizon tau (h) -- lead at first |ln ratio| = ln(2)\n(bootstrap 68% CI; ↑ = right-censored, tau > 6h)")
    fig.suptitle("Predictability horizon by functional: area vs intensity vs total precip (P=rho·A)\n"
                 "the horizon you measure depends on which functional you score", y=1.02)

    out = args.out
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
