"""Tests for the recovered P = rho*A scale decomposition (metrics/scale.py)."""
import numpy as np
import pytest

from latentsky.metrics import (
    cell_area_km2, default_scale_edges, label_systems, match_systems,
    match_systems_many_to_one, summarize_by_scale, system_table,
)


def test_cell_area_latitude_weighting():
    area = cell_area_km2(320, 1440)  # tropical crop, 0.25 deg
    # cells shrink by ~cos(lat): edge/equator ratio ~ cos(39.875 deg)
    ratio = area[0] / area.max()
    assert abs(ratio - np.cos(np.deg2rad(39.875))) < 1e-3
    # a 0.25-deg equatorial cell is ~27.8 km x 27.8 km
    assert abs(np.sqrt(area.max()) - 27.8) < 0.2


def test_identity_P_equals_rho_A_exact():
    rng = np.random.default_rng(0)
    field = np.where(rng.random((40, 60)) > 0.7, rng.gamma(2, 1, (40, 60)), 0.0)
    area_row = cell_area_km2(40, 60)
    _, tab = label_systems(field, 0.1, area_row)
    assert len(tab) > 0
    assert np.allclose(tab["P"], tab["rho"] * tab["A"], rtol=1e-12)
    assert np.allclose(tab["L"], np.sqrt(tab["A"]), rtol=1e-12)


def _disk(h, w, r0, c0, rad, value=1.0):
    rr, cc = np.mgrid[:h, :w]
    return np.where((rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2, value, 0.0)


def test_match_roles_and_decomposition():
    h, w = 64, 64
    area_row = cell_area_km2(h, w)
    truth = _disk(h, w, 20, 20, 8, 2.0) + _disk(h, w, 50, 50, 5, 1.0)
    # pred: first disk slightly shifted and bigger+weaker; second missing;
    # plus a spurious blob truth doesn't have
    pred = _disk(h, w, 22, 22, 10, 1.0) + _disk(h, w, 50, 10, 4, 1.5)
    df = match_systems(truth, pred, 0.1, area_row, iou_min=0.1)
    roles = df["role"].value_counts().to_dict()
    assert roles == {"match": 1, "miss": 1, "false_alarm": 1}
    m = df[df["role"] == "match"].iloc[0]
    assert m["dlnA"] > 0        # predicted bigger
    assert m["dlnrho"] < 0      # predicted weaker
    assert np.isclose(m["dlnP"], m["dlnA"] + m["dlnrho"], rtol=1e-12)


def test_many_to_one_merges_fragments():
    h, w = 64, 64
    area_row = cell_area_km2(h, w)
    truth = _disk(h, w, 32, 32, 12, 1.0)
    # prediction fragments the storm into two smaller blobs inside it
    pred = _disk(h, w, 28, 28, 6, 1.0) + _disk(h, w, 38, 38, 6, 1.0)
    greedy = match_systems(truth, pred, 0.1, area_row, iou_min=0.1)
    merged = match_systems_many_to_one(truth, pred, 0.1, area_row, iou_min=0.1)
    # greedy pairs one fragment, calls the other a false alarm
    assert (greedy["role"] == "false_alarm").sum() == 1
    # many-to-one merges both fragments into the single match
    assert (merged["role"] == "match").sum() == 1
    assert (merged["role"] == "false_alarm").sum() == 0
    gm = greedy[greedy["role"] == "match"].iloc[0]
    mm = merged[merged["role"] == "match"].iloc[0]
    assert mm["A_p"] > gm["A_p"]  # merged prediction area is larger


def test_summarize_by_scale_pod_far():
    h, w = 64, 128
    area_row = cell_area_km2(h, w)
    truth = _disk(h, w, 20, 30, 8, 1.0) + _disk(h, w, 40, 90, 8, 1.0)
    pred = _disk(h, w, 20, 30, 8, 1.0)  # one hit, one miss, no false alarm
    tbl = system_table(truth[None], pred[None], threshold=0.1, add_geo=False)
    summ = summarize_by_scale(tbl, default_scale_edges(25, 3000, 8))
    row = summ.by_scale.iloc[0]
    assert row["n_truth"] == 2 and row["n_match"] == 1 and row["n_miss"] == 1
    assert row["POD"] == pytest.approx(0.5)
    assert row["FAR"] == pytest.approx(0.0)


def test_fit_exp_decay_recovers_tau():
    from latentsky.metrics import fit_exp_decay
    t = np.array([0.25, 0.5, 1, 2, 3, 6])
    tau_true = 2.5
    y = np.exp(-(t - t[0]) / tau_true)
    out = fit_exp_decay(t - t[0], y)
    assert abs(out["tau"] - tau_true) < 1e-6


def test_fit_powerlaw_recovers_exponent():
    from latentsky.metrics import fit_powerlaw
    x = np.geomspace(30, 600, 6)
    y = 0.05 * x ** 1.3
    out = fit_powerlaw(x, y)
    assert abs(out["beta"] - 1.3) < 1e-6


def test_radial_spectrum_peaks_at_wave_scale():
    from latentsky.metrics import radial_spectrum
    h, w = 128, 256
    lam_km = 278.0  # 10-pixel wave
    x = np.arange(w) * 27.8
    field = np.sin(2 * np.pi * x / lam_km)[None, :].repeat(h, 0)
    k, p = radial_spectrum(field)
    assert abs(1.0 / k[np.nanargmax(p)] - lam_km) / lam_km < 0.15


def test_band_power_additive_scaling():
    from latentsky.metrics import band_power
    rng = np.random.default_rng(1)
    f = rng.random((64, 64))
    assert band_power(2 * f) == pytest.approx(4 * band_power(f), rel=1e-6)
