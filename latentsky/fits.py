"""Shared fitting helpers for the scale-decomposition figure suite.

Centralizes two patterns that were independently copy-pasted across 5+
plotting scripts (plot_tau_vs_scale*.py, plot_pod_decay*.py,
plot_tau_pod_far_vs_scale.py, plot_tau_far_vs_scale.py, and the linearized
variant in fit_dln_tau_scaling.py):

1. Exponential-decay skill fit: value(t)/value(0) = exp(-t/tau), B=0 forced
   (skill assumed to eventually vanish at long lead). Used for POD(t) and
   1-FAR(t).
2. Weighted-linear-regression tau fit for a quantity that's already
   log-linear in lead time (dlnA/dlnrho's bounded agreement index), plus a
   weighted (or unweighted) log-log power-law fit tau(L) = tau_ref *
   (L/L_ref)^beta, used for both flavors above.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def fit_exp_decay(t, norm_value, tau_bounds=(0.1, 200.0), p0=2.0) -> dict:
    """norm_value(t) = exp(-t/tau) via curve_fit, B=0 forced.

    `norm_value` must already be pre-normalized (e.g. POD(t)/POD(0)) --
    this function only fits the single tau parameter, it does not do the
    normalization itself (callers vary in what they normalize by).
    Returns {"tau": float, "tau_se": float} (nan/nan on fit failure).
    """
    def decay(tt, tau):
        return np.exp(-tt / tau)

    try:
        popt, pcov = curve_fit(decay, t, norm_value, p0=[p0],
                                bounds=([tau_bounds[0]], [tau_bounds[1]]))
        tau = float(popt[0])
        tau_se = float(np.sqrt(pcov[0, 0])) if pcov.size else float("nan")
        return {"tau": tau, "tau_se": tau_se}
    except Exception:
        return {"tau": float("nan"), "tau_se": float("nan")}


def fit_weighted_linreg(x, y, w) -> dict:
    """OLS slope/intercept weighted by w, with slope std error and R2."""
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
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
    return {"slope": slope, "intercept": intercept, "slope_se": slope_se, "r2": r2}


def fit_tau_from_neg_abs_dln(t, dln, n) -> dict:
    """tau from a weighted linear fit of -|dln(t)| vs t (the
    info_X(t)=I0*exp(-t/tau) linearization used for dlnA_med/dlnrho_med).
    Returns {I0, tau, tau_se, R2}; tau is nan if the fitted slope isn't
    negative (no decay detected)."""
    y = -np.abs(np.asarray(dln))
    fit = fit_weighted_linreg(t, y, n)
    if fit["slope"] >= 0:
        return {"I0": np.exp(fit["intercept"]), "tau": np.nan, "tau_se": np.nan, "R2": fit["r2"]}
    tau = -1 / fit["slope"]
    tau_se = tau ** 2 * fit["slope_se"]  # delta method: d(tau)/d(slope) = 1/slope^2
    return {"I0": np.exp(fit["intercept"]), "tau": tau, "tau_se": tau_se, "R2": fit["r2"]}


def fit_powerlaw(x, y, weights=None) -> dict:
    """tau = tau_ref * (x/x_ref)^beta via a (optionally weighted) log-log
    linear fit. weights=None -> ordinary (unweighted) least squares.
    x_ref/tau_ref are reported at the (weighted) mean of log(x), not at
    x=1, for numerical stability when x spans orders of magnitude.
    Returns {L_ref, tau_ref, beta, beta_se, R2, n_used, n_total} (all nan
    except n_used/n_total if fewer than 2 usable points)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (x > 0) & (y > 0) & (w > 0)
    if m.sum() < 2:
        return dict(L_ref=np.nan, tau_ref=np.nan, beta=np.nan, beta_se=np.nan,
                    R2=np.nan, n_used=int(m.sum()), n_total=len(x))
    lx, ly, lw = np.log(x[m]), np.log(y[m]), w[m]
    fit = fit_weighted_linreg(lx, ly, lw)
    L_ref = np.exp(np.average(lx, weights=lw))
    tau_ref = np.exp(np.average(ly, weights=lw))
    return dict(L_ref=L_ref, tau_ref=tau_ref, beta=fit["slope"], beta_se=fit["slope_se"],
                R2=fit["r2"], n_used=int(m.sum()), n_total=len(x))
