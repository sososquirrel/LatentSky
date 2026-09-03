"""Radially averaged power spectra for rain fields (promoted from
experiments/tstart_protocol/run.py, 2026-09-03 — the experiment keeps its
local copy for provenance; new code imports from here).

Spectral decomposition is an R3 representation metric (FRAME.md): band_power
ratios to truth quantify how much small-scale variance a model carries.
"""
from __future__ import annotations

import numpy as np

KM_PER_PIXEL = 27.8
SMALL_SCALE_KM = 1000.0

def radial_spectrum(field):
    f = np.fft.rfft2(field)
    p2 = np.abs(f) ** 2
    ky = np.fft.fftfreq(field.shape[0], d=KM_PER_PIXEL)
    kx = np.fft.rfftfreq(field.shape[1], d=KM_PER_PIXEL)
    kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kbins = np.linspace(kk[kk > 0].min(), kk.max(), 60)
    idx = np.digitize(kk.ravel(), kbins)
    ps = np.bincount(idx, weights=p2.ravel(), minlength=len(kbins) + 1)
    ns = np.bincount(idx, minlength=len(kbins) + 1)
    with np.errstate(invalid="ignore"):
        prof = ps / np.maximum(ns, 1)
    kmid = 0.5 * (kbins[:-1] + kbins[1:])
    return kmid, prof[1:len(kbins)]


def band_power(field, lam_max_km=SMALL_SCALE_KM, lam_min_km=2 * KM_PER_PIXEL):
    k, p = radial_spectrum(field)
    sel = (k > 1.0 / lam_max_km) & (k < 1.0 / lam_min_km)
    return float(np.nansum(p[sel]))
