"""Predictability timescale tau vs system scale L -- from exponential POD
decay fits, per model/run family (--tag).

Genericized (2026-07) from a script hardcoded to docs/scale_summary_v3.csv
+ "9var (mse+uv)" into a --summary/--predictor-set/--tag CLI tool, using
the shared fit helpers in latentsky/fits.py instead of a local copy of the
curve_fit + polyfit logic -- see latentsky/fits.py's docstring for why.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.fits import fit_exp_decay, fit_powerlaw
from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import FIGURES_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)",
                    help="Ignored if --runs is given -- see plot_pod_decay.py for why.")
    p.add_argument("--runs", nargs="+", default=None,
                    help="Filter to these exact `name` values instead of --predictor-set.")
    p.add_argument("--tag", default="v3", help="short model-family label, used in output filename")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df_all = pd.read_csv(REPO / args.summary)
    if args.runs is not None:
        df = df_all[df_all["name"].isin(args.runs)]
        if df.empty:
            opts = sorted(df_all["name"].unique())
            raise SystemExit(f"no rows for runs {args.runs!r}; available: {opts}")
    else:
        df = df_all[df_all["predictor_set"] == args.predictor_set]
        if df.empty:
            opts = sorted(df_all["predictor_set"].unique())
            raise SystemExit(f"no rows for predictor set {args.predictor_set!r}; available: {opts}")

    scales = sorted(df["scale_mid"].unique())
    has_lead0 = bool((df["lead_h"] == 0.0).any())
    if has_lead0:
        pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]
    else:
        pod0 = None
        print(f"note: no lead=0h data for {args.tag} -- normalizing each scale by its "
              f"own earliest available lead instead of assuming POD(0h)=1", flush=True)

    TAU_UPPER_BOUND = 200.0  # matches latentsky.fits.fit_exp_decay's default tau_bounds
    taus, non_monotonic = [], []
    for scale in scales:
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(scale, np.nan) if has_lead0 else sub["POD"].iloc[0]
        if not np.isfinite(p0) or p0 <= 0:
            taus.append(np.nan)
            non_monotonic.append(False)
            continue
        norm = (sub["POD"] / p0).values
        taus.append(fit_exp_decay(t, norm)["tau"])
        # A real single-exponential decay is monotonically non-increasing; POD
        # rising with lead (small numerical wobble aside) means the data
        # doesn't actually look exponential, regardless of what tau curve_fit
        # happened to converge to (same convention as
        # plot_pod_tau_track_compare.py -- keep both scripts consistent).
        non_monotonic.append(bool(np.any(np.diff(norm) > 1e-3)))

    scale_arr = np.array(scales)
    tau_arr = np.array(taus)
    non_monotonic = np.array(non_monotonic)
    # B=0 means 1 parameter, always constrained -- but a fit that saturates at the
    # tau_bounds upper edge (curve_fit ran out of room, not a real decay), or one
    # fit to data that doesn't actually decay monotonically, is not a reliable
    # data point for the power-law fit; exclude rather than plot as if solid.
    saturated = (tau_arr >= TAU_UPPER_BOUND - 1.0) | non_monotonic
    reliable = np.isfinite(tau_arr) & ~saturated
    if saturated.any():
        print(f"excluding {saturated.sum()} scale bin(s) (saturated fit or non-monotonic "
              f"POD decay) from the power-law: {scale_arr[saturated].tolist()}", flush=True)

    pl = fit_powerlaw(scale_arr[reliable], tau_arr[reliable])
    L_fit = np.geomspace(scale_arr[0], scale_arr[-1], 200)
    tau_fit = pl["tau_ref"] * (L_fit / pl["L_ref"]) ** pl["beta"]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    finite = np.isfinite(tau_arr)
    plotted_reliable = finite & ~saturated
    ax.scatter(scale_arr[plotted_reliable], tau_arr[plotted_reliable], color="#1f77b4", s=60,
               zorder=4, label="fitted tau  (B=0 forced)")
    if saturated.any():
        ax.scatter(scale_arr[saturated], tau_arr[saturated], color="#1f77b4",
                   s=60, alpha=0.3, zorder=4,
                   label="fit saturated or non-monotonic POD -- excluded from fit")
    ax.plot(L_fit, tau_fit, color="#d62728", lw=1.8, ls="--",
            label=f"power law:  tau = {pl['tau_ref']:.2f}h * (L/{pl['L_ref']:.0f}km)^{pl['beta']:.2f}")

    ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
    ax.text(35, 6.2, "6 h lead", fontsize=8, color="0.4")
    ax.axvline(56, color="0.55", lw=1.0, ls=":", zorder=1)
    ax.text(58, 0.35, "ERA5\nNyquist\n56 km", fontsize=7, color="0.55")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("system scale  L = sqrt(area)  [km]", fontsize=11)
    ax.set_ylabel("predictability timescale  tau  [h]", fontsize=11)
    ax.set_title(f"Scale-dependent predictability -- {args.tag}\n"
                 "Model: exp(-t/tau), B=0 forced.", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")

    out = REPO / args.out if args.out else FIGURES_DIR / f"tau_vs_scale_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"power law: tau = {pl['tau_ref']:.3f}h * (L/{pl['L_ref']:.1f}km)^{pl['beta']:.3f}  "
          f"[R2={pl['R2']:.3f}, {pl['n_used']}/{pl['n_total']} scale bins used]")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
