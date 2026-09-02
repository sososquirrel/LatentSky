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
from scipy.optimize import curve_fit, least_squares


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


_NAN_LINREG = {"slope": float("nan"), "intercept": float("nan"),
               "slope_se": float("nan"), "r2": float("nan")}


def fit_weighted_linreg(x, y, w) -> dict:
    """OLS slope/intercept weighted by w, with slope std error and R2.
    Returns all-NaN if there's fewer than 2 points, weights sum to zero
    (e.g. a scale bin with zero matched systems in a sparse lead sweep --
    encountered on old-LatentSky's 2-lead coverage), or x has no spread
    (Sxx==0) -- same "insufficient data -> NaN" convention used by
    fit_exp_decay/fit_powerlaw rather than letting np.average raise."""
    x, y, w = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    if len(x) < 2 or np.sum(w) <= 0:
        return dict(_NAN_LINREG)
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    Sxx = np.sum(w * (x - xbar) ** 2)
    if Sxx <= 0:
        return dict(_NAN_LINREG)
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


def first_crossing_tau(leads, values, thresh=np.log(2.0)) -> tuple[float, bool]:
    """Lead time at which |values(t)| first crosses `thresh`, linearly
    interpolated between the bracketing observed leads (in lead-time, not
    log-lead, matching fit_exp_decay's linear-time decay convention).

    `leads` must be sorted ascending, `values` already non-negative
    (e.g. |median dlnA| per lead). If the first point is already at/above
    threshold, returns that lead (no left-extrapolation). If the curve
    never reaches threshold within the observed range, returns
    (leads[-1], True) -- a right-censored lower bound: the true horizon is
    unknown but at least `leads[-1]`, so plot it as such rather than as a
    real crossing.
    """
    leads = np.asarray(leads, dtype=float)
    values = np.asarray(values, dtype=float)
    if values[0] >= thresh:
        return float(leads[0]), False
    above = values >= thresh
    if not above.any():
        return float(leads[-1]), True
    i = int(np.argmax(above))
    l0, l1 = leads[i - 1], leads[i]
    v0, v1 = values[i - 1], values[i]
    if v1 == v0:
        return float(l1), False
    frac = (thresh - v0) / (v1 - v0)
    return float(l0 + frac * (l1 - l0)), False


def bootstrap_horizon_tau(leads, raw_by_lead, thresh=np.log(2.0), n_boot=300, seed=0) -> dict:
    """Bootstrap CI for first_crossing_tau, resampling raw per-system dln
    values (with replacement) independently at each lead before taking the
    median -- so the CI reflects per-lead sampling noise in the median,
    propagated through the crossing-lead interpolation.

    `raw_by_lead`: list of 1D arrays (one per entry in `leads`, same order),
    each the raw (signed) dln values for that scale bin/lead/model/
    component; only their absolute value is used. A lead with zero
    observations makes every bootstrap replicate for it NaN, which is
    dropped (matches "insufficient data" convention elsewhere in this
    module) rather than raising.

    Returns {tau, lo, hi, frac_censored, n_boot_used}: `tau`/`lo`/`hi` are
    the median/16th/84th percentile of the bootstrap tau distribution
    (roughly a 68% CI, i.e. "1-sigma" for a normal distribution -- matches
    this module's tau_se convention elsewhere more closely than a 95% CI
    would). `frac_censored` is the fraction of bootstrap replicates that
    never crossed threshold (right-censored at leads[-1]).
    """
    rng = np.random.default_rng(seed)
    arrs = [np.abs(np.asarray(a, dtype=float)) for a in raw_by_lead]
    taus, cens = [], []
    for _ in range(n_boot):
        med = np.full(len(leads), np.nan)
        for j, a in enumerate(arrs):
            if len(a) == 0:
                continue
            med[j] = np.median(rng.choice(a, size=len(a), replace=True))
        if np.any(np.isnan(med)):
            continue
        tau, c = first_crossing_tau(leads, med, thresh)
        taus.append(tau)
        cens.append(c)
    if not taus:
        return dict(tau=np.nan, lo=np.nan, hi=np.nan, frac_censored=np.nan, n_boot_used=0)
    taus = np.array(taus)
    return dict(tau=float(np.median(taus)), lo=float(np.percentile(taus, 16)),
                hi=float(np.percentile(taus, 84)), frac_censored=float(np.mean(cens)),
                n_boot_used=len(taus))


def fit_joint_tau_powerlaw(t, L, dln, n) -> dict:
    """Joint nonlinear fit of info_X(t,L) = I0(L) * exp(-t / tau(L)),
    tau(L) = tau_ref * (L/L_ref)^beta, over ALL (t, L) observations at
    once -- weighted by n_match -- instead of the two-stage approach of
    fitting tau per scale bin first (fit_tau_from_neg_abs_dln) and then
    power-law-fitting those noisy point estimates (fit_powerlaw).

    The two-stage estimate lets scale bins with few matched systems (per-
    bin R2 as low as 0.01, seen at small/quantized L and long leads where
    n_match collapses) dominate or derail the across-bin fit even under
    inverse-variance weighting; excluding them via an R2 cutoff throws
    away real (if noisy) data rather than using it properly. Fitting
    tau_ref/beta directly against every (t, L, dlnX) triple -- with one
    nuisance intercept ln(I0) per scale bin, not a free tau per bin --
    pools all n_match-weighted residuals into 2 shared parameters, which
    is much better-conditioned than 7 independent per-bin fits followed by
    a fit of those noisy estimates.

    Returns {L_ref, tau_ref, beta, beta_se, I0 (dict scale->I0), R2,
    n_obs, n_bins}; tau_ref/beta/beta_se/R2 are nan if there are fewer
    than 2 scale bins or the fit fails to converge.
    """
    t = np.asarray(t, dtype=float)
    L = np.asarray(L, dtype=float)
    y = -np.abs(np.asarray(dln, dtype=float))
    w = np.asarray(n, dtype=float)
    m = np.isfinite(t) & np.isfinite(L) & np.isfinite(y) & np.isfinite(w) & (L > 0) & (w > 0)
    t, L, y, w = t[m], L[m], y[m], w[m]
    bins = np.unique(L)
    nan_out = dict(L_ref=np.nan, tau_ref=np.nan, beta=np.nan, beta_se=np.nan,
                    I0={}, R2=np.nan, n_obs=int(len(t)), n_bins=int(len(bins)))
    if len(bins) < 2 or len(t) < len(bins) + 2:
        return nan_out

    L_ref = float(np.exp(np.average(np.log(L), weights=w)))
    bin_idx = np.searchsorted(bins, L)
    sw = np.sqrt(w)

    def residuals(params):
        log_tau_ref, beta = params[0], params[1]
        ln_I0 = np.asarray(params[2:])
        tau_L = np.exp(log_tau_ref) * (L / L_ref) ** beta
        pred = ln_I0[bin_idx] - t / tau_L
        return sw * (y - pred)

    tau0 = max(float(np.median(t[t > 0])) if np.any(t > 0) else 1.0, 0.5)
    ln_I0_0 = [float(np.average(y[bin_idx == i], weights=w[bin_idx == i]))
               for i in range(len(bins))]
    p0 = [np.log(tau0), 1.0] + ln_I0_0

    try:
        res = least_squares(residuals, p0, method="lm", max_nfev=10000)
    except Exception:
        return nan_out
    if not np.all(np.isfinite(res.x)):
        return nan_out

    log_tau_ref, beta = res.x[0], res.x[1]
    ln_I0_fit = res.x[2:]
    tau_ref = float(np.exp(log_tau_ref))

    J = res.jac
    dof = max(len(t) - len(p0), 1)
    chi2_red = float(np.sum(res.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * chi2_red
        beta_se = float(np.sqrt(cov[1, 1])) if cov[1, 1] >= 0 else float("nan")
    except np.linalg.LinAlgError:
        beta_se = float("nan")

    tau_L_best = tau_ref * (L / L_ref) ** beta
    yhat = ln_I0_fit[bin_idx] - t / tau_L_best
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return dict(L_ref=L_ref, tau_ref=tau_ref, beta=float(beta), beta_se=beta_se,
                I0=dict(zip(bins.tolist(), np.exp(ln_I0_fit).tolist())),
                R2=R2, n_obs=int(len(t)), n_bins=int(len(bins)))
