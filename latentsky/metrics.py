"""Standard metrics used in all four canonical notebooks.

Cherry-picked from tzhou/LatentSky/util_LS.py:
    r2_all_pixels_safe, r2_rain_only, js_divergence_binned, compute_metrics.
"""

from __future__ import annotations

import numpy as np


def r2_all_pixels_safe(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + eps
    return float(1.0 - ss_res / ss_tot)


def r2_rain_only(y_true, y_pred, threshold: float = 0.1, eps: float = 1e-12) -> float:
    mask = y_true > threshold
    return r2_all_pixels_safe(y_true[mask], y_pred[mask], eps)


def _contingency_counts(y_true, y_pred, threshold: float):
    t = np.asarray(y_true).ravel() > threshold
    p = np.asarray(y_pred).ravel() > threshold
    hits = int(np.sum(t & p))
    misses = int(np.sum(t & ~p))
    fa = int(np.sum(~t & p))
    correct_neg = int(np.sum(~t & ~p))
    return hits, misses, fa, correct_neg


def csi(y_true, y_pred, threshold: float = 0.1) -> float:
    """Critical Success Index — hits / (hits + misses + false_alarms)."""
    hits, misses, fa, _ = _contingency_counts(y_true, y_pred, threshold)
    denom = hits + misses + fa
    return float(hits / denom) if denom > 0 else float("nan")


def pod(y_true, y_pred, threshold: float = 0.1) -> float:
    """Probability of Detection (hit rate) — hits / (hits + misses)."""
    hits, misses, _, _ = _contingency_counts(y_true, y_pred, threshold)
    denom = hits + misses
    return float(hits / denom) if denom > 0 else float("nan")


def far(y_true, y_pred, threshold: float = 0.1) -> float:
    """False Alarm Ratio — false_alarms / (hits + false_alarms)."""
    hits, _, fa, _ = _contingency_counts(y_true, y_pred, threshold)
    denom = hits + fa
    return float(fa / denom) if denom > 0 else float("nan")


def freq_bias(y_true, y_pred, threshold: float = 0.1) -> float:
    """Frequency bias — predicted-rain area / observed-rain area. 1.0 = perfect."""
    hits, misses, fa, _ = _contingency_counts(y_true, y_pred, threshold)
    denom = hits + misses
    return float((hits + fa) / denom) if denom > 0 else float("nan")


def freq_intensity_scalars(y_true, y_pred, threshold: float = 0.1) -> dict:
    """Per-field scalars for the closed mean-bias decomposition.

    rho_t, rho_p : rain frequency (fraction of pixels above threshold)
    mu_t,  mu_p  : mean conditional intensity (mean of values above threshold)

    Mean precip exactly factors as rho * mu, so error decomposes as
        eps_freq = (rho_p - rho_t) * mu_t
        eps_int  = rho_p * (mu_p - mu_t)
        eps_freq + eps_int = rho_p mu_p - rho_t mu_t  (total mean-precip bias)
    """
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    t_mask = t > threshold
    p_mask = p > threshold
    rho_t = float(t_mask.mean())
    rho_p = float(p_mask.mean())
    mu_t = float(t[t_mask].mean()) if t_mask.any() else float("nan")
    mu_p = float(p[p_mask].mean()) if p_mask.any() else float("nan")
    return {"rho_truth": rho_t, "rho_pred": rho_p, "mu_truth": mu_t, "mu_pred": mu_p}


def js_divergence_binned(x, y, bins: int = 100, base: float = 2.0) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    data_range = (min(x.min(), y.min()), max(x.max(), y.max()))
    p, edges = np.histogram(x, bins=bins, range=data_range)
    q, _ = np.histogram(y, bins=edges)
    p = p.astype(float); q = q.astype(float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * (np.log(a[mask] / b[mask]) / np.log(base))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    truth = truth.astype(float)
    pred = pred.astype(float)

    mse = float(np.mean((pred - truth) ** 2))
    mae = float(np.mean(np.abs(pred - truth)))
    max_err = float(np.max(np.abs(pred - truth)))
    corr = float(np.corrcoef(truth.flatten(), pred.flatten())[0, 1])
    fsd = float(np.std(pred) / (np.std(truth) + 1e-12))
    spectral_ratio = float(
        np.abs(np.fft.fft2(pred)).sum() ** 2 / (np.abs(np.fft.fft2(truth)).sum() ** 2 + 1e-12)
    )
    return {
        "MSE": mse,
        "MAE": mae,
        "MaxError": max_err,
        "Correlation": corr,
        "FracStdDev": fsd,
        "SpectralRatio": spectral_ratio,
        "R2_all": r2_all_pixels_safe(truth, pred),
        "R2_rain": r2_rain_only(truth, pred),
        "JS_div": js_divergence_binned(truth, pred),
        "CSI": csi(truth, pred),
        "POD": pod(truth, pred),
        "FAR": far(truth, pred),
        "FreqBias": freq_bias(truth, pred),
        **freq_intensity_scalars(truth, pred),
    }
