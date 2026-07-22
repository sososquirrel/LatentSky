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
    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]

    taus = []
    for scale in scales:
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(scale, 1.0)
        if p0 <= 0:
            taus.append(np.nan)
            continue
        norm = (sub["POD"] / p0).values
        taus.append(fit_exp_decay(t, norm)["tau"])

    scale_arr = np.array(scales)
    tau_arr = np.array(taus)
    reliable = np.isfinite(tau_arr)  # B=0 means 1 parameter, always constrained

    pl = fit_powerlaw(scale_arr[reliable], tau_arr[reliable])
    L_fit = np.geomspace(scale_arr[0], scale_arr[-1], 200)
    tau_fit = pl["tau_ref"] * (L_fit / pl["L_ref"]) ** pl["beta"]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    ax.scatter(scale_arr, tau_arr, color="#1f77b4", s=60, zorder=4,
               label="fitted tau  (B=0 forced)")
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
