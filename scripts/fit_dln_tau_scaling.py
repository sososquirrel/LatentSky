"""Predictability-timescale scaling law: tau(L) for dlnA and dlnrho.

Fits an exponential decay to the bounded agreement index from
scripts/plot_info_fraction.py:

    info_X(t, L) = exp(-|dlnX_med(t, L)|) = I0(L) * exp(-t / tau(L))

Taking logs linearizes this exactly (no nonlinear solver needed):

    -|dlnX_med(t, L)| = ln(I0(L)) - t / tau(L)

so tau(L) is recovered per scale bin by a single n_match-weighted linear
regression of -|dlnX_med(t)| against lead time t; tau = -1/slope.

tau(L) is then regressed as a power law tau = tau_ref * (L / L_ref)^beta
via a weighted log-log linear fit (weight = fit precision, 1/se_ln_tau^2).

dlnP is not analyzed here, same reasoning as plot_info_fraction.py: it's a
sum of two such decays with different signs/timescales and is often
non-monotonic, so a single tau isn't a meaningful summary for it.

Outputs:
  docs/dln_tau_fits_<tag>.csv          one row per (quantity, scale bin):
                                        I0, tau, tau_se, R2
  docs/figures/tau_vs_scale_<tag>.png  tau_A(L) and tau_rho(L) + power-law
                                        fit lines, log-log axes

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
from matplotlib.colors import TwoSlopeNorm


def _weighted_linreg(x, y, w):
    """OLS slope/intercept weighted by w, with slope std error and R2."""
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    Sxx = np.sum(w * (x - xbar) ** 2)
    slope = np.sum(w * (x - xbar) * (y - ybar)) / Sxx
    intercept = ybar - slope * xbar

    yhat = intercept + slope * x
    resid = y - yhat
    dof = max(len(x) - 2, 1)
    reduced_chi2 = np.sum(w * resid ** 2) / dof
    slope_se = np.sqrt(reduced_chi2 / Sxx)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    r2 = 1 - np.sum(w * resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, slope_se, r2


def _fit_tau(t, dlnX, n):
    """tau from a weighted linear fit of -|dlnX(t)| vs t (see module docstring)."""
    y = -np.abs(dlnX)
    slope, intercept, slope_se, r2 = _weighted_linreg(t, y, n)
    if slope >= 0:
        return dict(I0=np.exp(intercept), tau=np.nan, tau_se=np.nan, R2=r2)
    tau = -1 / slope
    tau_se = tau ** 2 * slope_se  # delta method: d(tau)/d(slope) = 1/slope^2
    return dict(I0=np.exp(intercept), tau=tau, tau_se=tau_se, R2=r2)


def _powerlaw_fit(scale, tau, tau_se):
    """tau = tau_ref * (scale / L_ref)^beta via weighted log-log linear fit."""
    m = np.isfinite(tau) & np.isfinite(tau_se) & (tau > 0) & (tau_se > 0)
    if m.sum() < 2:
        return dict(L_ref=np.nan, tau_ref=np.nan, beta=np.nan, beta_se=np.nan,
                    R2=np.nan, n_used=int(m.sum()), n_total=len(scale))
    x = np.log(scale[m])
    y = np.log(tau[m])
    w = 1 / (tau_se[m] / tau[m]) ** 2
    beta, _intercept, beta_se, r2 = _weighted_linreg(x, y, w)
    L_ref = np.exp(np.average(x, weights=w))
    tau_ref = np.exp(np.average(y, weights=w))  # fitted value at x = xbar = log(L_ref)
    return dict(L_ref=L_ref, tau_ref=tau_ref, beta=beta, beta_se=beta_se,
                R2=r2, n_used=int(m.sum()), n_total=len(scale))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--tag", default="v3")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo / args.summary)
    g = df[df["predictor_set"] == args.predictor_set]
    if g.empty:
        opts = sorted(df["predictor_set"].unique())
        raise SystemExit(f"no rows for predictor set {args.predictor_set!r}; "
                         f"available: {opts}")

    quantities = {"A": "dlnA_med", "rho": "dlnrho_med"}
    scales = np.array(sorted(g["scale_mid"].unique()))

    fit_rows = []
    for qname, col in quantities.items():
        for scale in scales:
            sub = g[g["scale_mid"] == scale].sort_values("lead_h")
            sub = sub[sub[col].notna() & sub["n_match"].notna()]
            t = sub["lead_h"].to_numpy()
            y = sub[col].to_numpy()
            n = sub["n_match"].to_numpy()
            fit = _fit_tau(t, y, n)
            fit_rows.append(dict(quantity=qname, scale_mid=scale, **fit))

    fits = pd.DataFrame(fit_rows)
    out_table = repo / f"docs/dln_tau_fits_{args.tag}.csv"
    fits.to_csv(out_table, index=False)
    print(f"wrote {out_table}")

    print("\ninfo_X(t) = I0 * exp(-t/tau), fit via weighted linear regression "
          "of -|dlnX_med(t)| vs t")
    print(fits.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    powerlaws = {}
    for qname in quantities:
        sub = fits[fits["quantity"] == qname].sort_values("scale_mid")
        pl = _powerlaw_fit(sub["scale_mid"].to_numpy(), sub["tau"].to_numpy(),
                           sub["tau_se"].to_numpy())
        powerlaws[qname] = pl
        if np.isfinite(pl["beta"]):
            print(f"\ntau_{qname}(L) = {pl['tau_ref']:.2f} h * "
                  f"(L / {pl['L_ref']:.0f} km)^({pl['beta']:.2f} +/- {pl['beta_se']:.2f})"
                  f"   [R2={pl['R2']:.2f}, {pl['n_used']}/{pl['n_total']} scale bins used]")
        else:
            print(f"\ntau_{qname}(L): fewer than 2 usable scale bins "
                  f"({pl['n_used']}/{pl['n_total']}) — no power law fit")

    # dlnP at lead 0 per scale — the total-precip gap already present at
    # nowcast time, used to color the tau markers (not fit to anything).
    dlnP0 = (g[g["lead_h"] == 0.0].set_index("scale_mid")["dlnP_med"]
             .reindex(scales).to_numpy())
    vmax = max(abs(np.nanmin(dlnP0)), abs(np.nanmax(dlnP0)), 1e-6)
    cnorm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = cm.RdYlGn

    # --- headline plot: tau_A(L) and tau_rho(L) ---------------------------
    fig, ax = plt.subplots(figsize=(8.2, 6))
    styles = {"A": dict(marker="o", line_color="C0", label="tau_A  (area)"),
             "rho": dict(marker="s", line_color="C3", label="tau_ρ  (intensity)")}
    Lgrid = np.geomspace(scales.min(), scales.max(), 100)
    dlnP0_by_scale = dict(zip(scales, dlnP0))

    for qname, style in styles.items():
        sub = fits[fits["quantity"] == qname].sort_values("scale_mid")
        L = sub["scale_mid"].to_numpy()
        tau = sub["tau"].to_numpy()
        tau_se = sub["tau_se"].to_numpy()
        m = np.isfinite(tau)
        c = np.array([dlnP0_by_scale[l] for l in L])

        ax.errorbar(L[m], tau[m], yerr=tau_se[m], fmt="none",
                    ecolor="0.6", capsize=3, zorder=2)
        sc = ax.scatter(L[m], tau[m], c=c[m], cmap=cmap, norm=cnorm,
                        marker=style["marker"], s=70, edgecolors="k",
                        linewidths=0.6, zorder=3, label=style["label"])

        pl = powerlaws[qname]
        if np.isfinite(pl["beta"]):
            tau_fit = pl["tau_ref"] * (Lgrid / pl["L_ref"]) ** pl["beta"]
            ax.plot(Lgrid, tau_fit, "--", color=style["line_color"], lw=1.5, alpha=0.8,
                    label=f"beta_{qname} = {pl['beta']:.2f} +/- {pl['beta_se']:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("system scale L = √A  (km)")
    ax.set_ylabel("tau  (h)   [info_X(t) = I0 exp(-t/tau)]")
    ax.set_title(f"Agreement-decay timescale vs scale — {args.predictor_set}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    cbar = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax)
    cbar.set_label("dln P at lead 0h  (total-precip gap already\npresent at nowcast time)")

    fig.tight_layout()
    out_fig = repo / f"docs/figures/tau_vs_scale_{args.tag}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_fig}")


if __name__ == "__main__":
    main()
