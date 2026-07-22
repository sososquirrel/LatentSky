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

from latentsky.fits import fit_tau_from_neg_abs_dln, fit_powerlaw
from latentsky.paths import FIGURES_DIR


def _fit_tau(t, dlnX, n):
    """tau from a weighted linear fit of -|dlnX(t)| vs t (see module docstring).
    Thin wrapper over latentsky.fits.fit_tau_from_neg_abs_dln to keep this
    script's existing dict-key names (I0/tau/tau_se/R2)."""
    fit = fit_tau_from_neg_abs_dln(t, dlnX, n)
    return dict(I0=fit["I0"], tau=fit["tau"], tau_se=fit["tau_se"], R2=fit["R2"])


def _powerlaw_fit(scale, tau, tau_se):
    """tau = tau_ref * (scale / L_ref)^beta via weighted log-log linear fit,
    weight = fit precision (1/relative_se^2). Thin wrapper over
    latentsky.fits.fit_powerlaw."""
    tau, tau_se = np.asarray(tau), np.asarray(tau_se)
    m = np.isfinite(tau) & np.isfinite(tau_se) & (tau > 0) & (tau_se > 0)
    w = np.full_like(tau, np.nan)
    w[m] = 1 / (tau_se[m] / tau[m]) ** 2
    return fit_powerlaw(scale, tau, weights=w)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)",
                    help="Ignored if --runs is given -- see plot_pod_decay.py for why.")
    p.add_argument("--runs", nargs="+", default=None,
                    help="Filter to these exact `name` values instead of --predictor-set.")
    p.add_argument("--tag", default="v3")
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    df = pd.read_csv(repo / args.summary)
    if args.runs is not None:
        g = df[df["name"].isin(args.runs)]
        label = args.tag
        if g.empty:
            opts = sorted(df["name"].unique())
            raise SystemExit(f"no rows for runs {args.runs!r}; available: {opts}")
    else:
        g = df[df["predictor_set"] == args.predictor_set]
        label = args.predictor_set
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
    # Some run families (e.g. old-LatentSky, only ever trained/evaluated at
    # 3h/6h) have no lead=0h row at all -- reindex() against an empty frame
    # then silently gives all-NaN, and matplotlib renders NaN-colored
    # scatter points as fully transparent (cmap "bad" color defaults to
    # alpha=0), which looks like "nothing plotted" even though the
    # errorbar dashes still render. Fall back to a plain fixed color and
    # skip the colorbar rather than silently producing invisible markers.
    has_lead0 = (g["lead_h"] == 0.0).any()
    if has_lead0:
        dlnP0 = (g[g["lead_h"] == 0.0].set_index("scale_mid")["dlnP_med"]
                 .reindex(scales).to_numpy())
        vmax = max(abs(np.nanmin(dlnP0)), abs(np.nanmax(dlnP0)), 1e-6)
        cnorm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap = cm.RdYlGn
    else:
        print(f"note: no lead=0h data for {label} -- markers won't be colored by "
              "dln P at lead 0h (doesn't exist for this run family)")
        dlnP0_by_scale = {}
        cnorm = None
        cmap = None

    # --- headline plot: tau_A(L) and tau_rho(L) ---------------------------
    fig, ax = plt.subplots(figsize=(8.2, 6))
    styles = {"A": dict(marker="o", line_color="C0", label="tau_A  (area)"),
             "rho": dict(marker="s", line_color="C3", label="tau_ρ  (intensity)")}
    Lgrid = np.geomspace(scales.min(), scales.max(), 100)
    if has_lead0:
        dlnP0_by_scale = dict(zip(scales, dlnP0))

    for qname, style in styles.items():
        sub = fits[fits["quantity"] == qname].sort_values("scale_mid")
        L = sub["scale_mid"].to_numpy()
        tau = sub["tau"].to_numpy()
        tau_se = sub["tau_se"].to_numpy()
        m = np.isfinite(tau)

        ax.errorbar(L[m], tau[m], yerr=tau_se[m], fmt="none",
                    ecolor="0.6", capsize=3, zorder=2)
        if has_lead0:
            c = np.array([dlnP0_by_scale[l] for l in L])
            sc = ax.scatter(L[m], tau[m], c=c[m], cmap=cmap, norm=cnorm,
                            marker=style["marker"], s=70, edgecolors="k",
                            linewidths=0.6, zorder=3, label=style["label"])
        else:
            ax.scatter(L[m], tau[m], color=style["line_color"],
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
    ax.set_title(f"Agreement-decay timescale vs scale — {label}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    if has_lead0:
        cbar = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax)
        cbar.set_label("dln P at lead 0h  (total-precip gap already\npresent at nowcast time)")

    fig.tight_layout()
    out_fig = FIGURES_DIR / f"tau_vs_scale_{args.tag}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_fig}")


if __name__ == "__main__":
    main()
