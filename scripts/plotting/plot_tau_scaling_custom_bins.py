"""tau(L) predictability-scaling law, computed from POD(t)/POD(0) exponential
decay per scale bin, using the SAME custom scale-bin edges as
plot_pod_far_vs_lead_by_scale.py / plot_dln_3panel.py (the precip
decomposition figure) -- so this is the direct scaling-law companion to
that POD/FAR figure, not the earlier rebinned16-based tau(L) figures.

Per (track, scale bin): POD(t) computed from raw systems CSVs at each lead,
normalized by POD at the earliest available lead, then fit tau via
latentsky.fits.fit_exp_decay (POD(t)/POD(0) = exp(-t/tau)). tau(L) is then
fit as a power law (latentsky.fits.fit_powerlaw) across bins, per track.
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

_LEADS = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0]
_LEAD_LABELS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]

TRACKS = [
    dict(key="onehead", systems="docs/scale_systems_onehead_patchsplit.csv.gz",
         runs=[f"patchsplit_onehead_lead{L}" for L in _LEAD_LABELS],
         label="regression (new split)", color="tab:purple", marker="o"),
    dict(key="two-head", systems="docs/scale_systems_modelcompare.csv.gz",
         runs=[f"patchsplit_twohead_olr_fss_lead{L}" for L in _LEAD_LABELS],
         label="two-head", color="tab:orange", marker="s"),
    dict(key="diffusion", systems="docs/scale_systems_diffusion.csv.gz",
         runs=[f"diffusion_lead{L}" for L in _LEAD_LABELS],
         label="two-head+diffusion", color="tab:green", marker="D"),
    dict(key="persistence", systems="docs/scale_systems_baselines_patchsplit.csv.gz",
         runs=[f"persistence_lead{L}" for L in _LEAD_LABELS],
         label="persistence", color="gray", marker="v"),
]
DEFAULT_EDGES = [60, 62.0, 71.4, 80, 100, 140, 200, 300, 450, 700]


def _labels(edges):
    e = np.round(edges).astype(int)
    return [f"<{e[1]}km" if i == 0 else f"{e[i]}-{e[i+1]}km" for i in range(len(e) - 1)]


def _pod_by_bin_lead(systems_path, runs, edges, floor_km, repo):
    df = pd.read_csv(repo / systems_path, usecols=["name", "lead_h", "role", "scale"])
    df = df[df["name"].isin(runs)].copy()
    df = df[(df["scale"] >= max(floor_km, edges[0])) & (df["scale"] < edges[-1])]
    b = np.clip(np.digitize(df["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    df["bin"] = np.array(_labels(edges))[b]

    rows = []
    for (lead_h, lbl), g in df.groupby(["lead_h", "bin"]):
        n_match = int((g["role"] == "match").sum())
        n_miss = int((g["role"] == "miss").sum())
        n_truth = n_match + n_miss
        rows.append(dict(lead_h=lead_h, bin=lbl, POD=n_match / n_truth if n_truth else np.nan,
                          n_truth=n_truth))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--floor-km", type=float, default=60.0)
    p.add_argument("--out", default=str(FIGURES_DIR / "tau_scaling_custom_bins.png"))
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    edges = np.array(args.edges)
    labels = _labels(edges)
    mids = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    for t in TRACKS:
        pf = _pod_by_bin_lead(t["systems"], t["runs"], edges, args.floor_km, repo)
        Ls, taus, tau_ses = [], [], []
        for lbl, Lmid in zip(labels, mids):
            g = pf[pf["bin"] == lbl].sort_values("lead_h")
            if len(g) < 2 or g["POD"].iloc[0] <= 0 or np.isnan(g["POD"].iloc[0]):
                continue
            p0 = g["POD"].iloc[0]
            norm = g["POD"].to_numpy() / p0
            fit = fit_exp_decay(g["lead_h"].to_numpy(), norm)
            if np.isnan(fit["tau"]):
                continue
            Ls.append(Lmid)
            taus.append(fit["tau"])
            tau_ses.append(fit["tau_se"])
        Ls, taus, tau_ses = np.array(Ls), np.array(taus), np.array(tau_ses)

        ax.errorbar(Ls, taus, yerr=tau_ses, fmt=t["marker"], ms=8, color=t["color"],
                    capsize=3, markeredgecolor="k", markeredgewidth=0.6, zorder=4,
                    label=t["label"])
        if len(Ls) >= 2:
            pl = fit_powerlaw(Ls, taus)
            if np.isfinite(pl["beta"]):
                xs = np.geomspace(Ls.min(), Ls.max(), 50)
                ys = pl["tau_ref"] * (xs / pl["L_ref"]) ** pl["beta"]
                ax.plot(xs, ys, "--", color=t["color"], lw=1.8, alpha=0.7, zorder=2,
                        label=f"{t['label']}: beta={pl['beta']:.2f} (R2={pl['R2']:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scale L (km)")
    ax.set_ylabel("tau (h) -- from POD(t)/POD(0) = exp(-t/tau)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("tau(L) scaling law, POD normalized by initial value\n"
                 "same scale bins/tracks as the POD/FAR-vs-lead figure")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
