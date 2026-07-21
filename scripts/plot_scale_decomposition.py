"""Scale-aware precipitation decomposition figure.

Reads the two artifacts written by scripts/eval_scale_decomposition.py
(docs/scale_systems.parquet, docs/scale_summary.csv) and produces, for one
run, a six-panel physically-interpretable view:

  (a) system-size distribution        truth vs pred, log-log
  (b) precip carried by each scale    truth vs pred  (where the water is)
  (c) object detection vs scale       POD and FAR by system size
  (d) per-scale error split           median dlnA vs dlnrho (+IQR band):
                                      is the model wrong on size or rate?
  (e) precip-bias budget vs scale     area vs intensity fraction of the
                                      matched-system bias, plus the
                                      miss/false-alarm detection term
  (f) attribute PDFs                  truth vs pred for A, rho, P

This replaces bulk RMSE/correlation framing with P = rho * A,
dlnP = dlnA + dlnrho, stratified by precipitating-system scale.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _logbins(*arrays, n=30):
    vals = np.concatenate([a[a > 0] for a in arrays if a.size])
    return np.geomspace(vals.min(), vals.max(), n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems.csv.gz",
                   help="Per-system table written by the eval script "
                        "(.csv/.csv.gz or .parquet).")
    p.add_argument("--summary", default="docs/scale_summary.csv")
    p.add_argument("--run", default="mseuv_lead3h")
    p.add_argument("--out", default=None,
                   help="Default docs/figures/scale_decomp_<run>.png")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    sys_path = repo / args.systems
    sysdf = (pd.read_parquet(sys_path) if str(sys_path).endswith(".parquet")
             else pd.read_csv(sys_path))
    summ = pd.read_csv(repo / args.summary)

    s = sysdf[sysdf["name"] == args.run]
    g = summ[summ["name"] == args.run].sort_values("scale_mid")
    if s.empty or g.empty:
        raise SystemExit(f"no rows for run {args.run!r} — check eval output")

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))

    # (a) system-size (characteristic length L = sqrt(A)) distribution
    Lt = s.loc[s["L_t"].notna(), "L_t"].to_numpy()
    Lp = s.loc[s["L_p"].notna(), "L_p"].to_numpy()
    be = _logbins(Lt, Lp, n=30)
    ax[0, 0].hist(Lt, bins=be, histtype="step", color="k", label="truth", lw=1.8)
    ax[0, 0].hist(Lp, bins=be, histtype="step", color="C3", label="pred", lw=1.8)
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("system scale L = √A  (km)")
    ax[0, 0].set_ylabel("# systems")
    ax[0, 0].set_title("(a) system-size distribution")
    ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.25)

    # (b) precip carried by each scale bin
    bin_edges = np.r_[g["scale_lo"].to_numpy(), g["scale_hi"].to_numpy()[-1]]
    cats = pd.cut(s["scale"], bins=bin_edges)
    Pt_bin = s.groupby(cats, observed=False)["P_t"].sum()
    Pp_bin = s.groupby(cats, observed=False)["P_p"].sum()
    x = g["scale_mid"].to_numpy()
    ax[0, 1].plot(x, Pt_bin.to_numpy(), "o-", color="k", label="truth")
    ax[0, 1].plot(x, Pp_bin.to_numpy(), "s--", color="C3", label="pred")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("system scale (km)")
    ax[0, 1].set_ylabel("Σ P  (mm·km²)")
    ax[0, 1].set_title("(b) precip by scale — where the water is")
    ax[0, 1].legend(); ax[0, 1].grid(True, which="both", alpha=0.25)

    # (c) detection skill vs scale
    ax[0, 2].plot(x, g["POD"], "o-", color="C0", label="POD")
    ax[0, 2].plot(x, g["FAR"], "s--", color="C1", label="FAR")
    ax[0, 2].set_xscale("log"); ax[0, 2].set_ylim(0, 1)
    ax[0, 2].set_xlabel("system scale (km)")
    ax[0, 2].set_ylabel("object skill")
    ax[0, 2].set_title("(c) detection vs scale")
    ax[0, 2].legend(); ax[0, 2].grid(True, alpha=0.3)

    # (d) per-scale error split: median dlnA vs dlnrho with IQR band
    ax[1, 0].axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax[1, 0].plot(x, g["dlnA_med"], "o-", color="C0", label="dln A (area)")
    ax[1, 0].fill_between(x, g["dlnA_med"] - g["dlnA_iqr"] / 2,
                          g["dlnA_med"] + g["dlnA_iqr"] / 2,
                          color="C0", alpha=0.15)
    ax[1, 0].plot(x, g["dlnrho_med"], "s-", color="C3", label="dln ρ (intensity)")
    ax[1, 0].fill_between(x, g["dlnrho_med"] - g["dlnrho_iqr"] / 2,
                          g["dlnrho_med"] + g["dlnrho_iqr"] / 2,
                          color="C3", alpha=0.15)
    ax[1, 0].plot(x, g["dlnP_med"], "^:", color="0.4", label="dln P (total)")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("system scale (km)")
    ax[1, 0].set_ylabel("median log error (pred/truth)")
    ax[1, 0].set_title("(d) area vs intensity error by scale")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, which="both", alpha=0.25)

    # (e) precip-bias budget: matched area/intensity fractions + detection term
    ax[1, 1].plot(x, g["area_frac"], "o-", color="C0", label="area share")
    ax[1, 1].plot(x, g["int_frac"], "s-", color="C3", label="intensity share")
    ax[1, 1].plot(x, g["miss_P_frac"], "v--", color="C2",
                  label="missed-precip frac")
    ax[1, 1].plot(x, g["fa_P_frac"], "^--", color="C1",
                  label="false-precip frac")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_xlabel("system scale (km)")
    ax[1, 1].set_ylabel("fraction")
    ax[1, 1].set_title("(e) precip-bias budget by scale")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, alpha=0.3)

    # (f) attribute PDFs (rho shown; A and P as twin-scaled insets via steps)
    rt = s.loc[s["rho_t"].notna(), "rho_t"].to_numpy()
    rp = s.loc[s["rho_p"].notna(), "rho_p"].to_numpy()
    rb = _logbins(rt, rp, n=40)
    ax[1, 2].hist(rt, bins=rb, density=True, histtype="step", color="k",
                  lw=1.8, label="truth ρ")
    ax[1, 2].hist(rp, bins=rb, density=True, histtype="step", color="C3",
                  lw=1.8, label="pred ρ")
    ax[1, 2].set_xscale("log")
    ax[1, 2].set_xlabel("system mean intensity ρ  (mm)")
    ax[1, 2].set_ylabel("density")
    ax[1, 2].set_title("(f) intensity distribution")
    ax[1, 2].legend(); ax[1, 2].grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"Scale-aware decomposition — {args.run}   "
        f"(P = ρ·A,  dlnP = dlnA + dlnρ)", fontsize=14)
    fig.tight_layout()

    out = (repo / (args.out or f"docs/figures/scale_decomp_{args.run}.png")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
