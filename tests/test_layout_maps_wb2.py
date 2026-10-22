"""Tests for the shared dataset layout, map helpers and WB2 helpers."""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from latentsky.comparison.wb2 import crop, to_mm  # noqa: E402
from latentsky.evaluation.maps import add_coastlines, grid_lines  # noqa: E402
from latentsky import layout  # noqa: E402


def test_input_channels_rain_last():
    assert len(layout.INPUT_VARS) == 10
    assert layout.INPUT_VARS[-1] == "rain_log1p"


def test_block_slice_and_leads():
    a = np.arange(layout.H * layout.W).reshape(layout.H, layout.W)
    blk = a[layout.block_slice((1, 4))]
    assert blk.shape == (layout.P, layout.P)
    assert blk[0, 0] == a[128, 512]
    assert layout.lead_frames(3) == 12 and layout.lead_frames(0.25) == 1
    with pytest.raises(ValueError):
        layout.lead_frames(0.1)


def test_load_frame_env_override(tmp_path, monkeypatch):
    np.save(tmp_path / "precipw_piece03.npy", np.ones((5, 4, 6), np.float64) * 2)
    monkeypatch.setenv("LATENTSKY_PIECES", str(tmp_path))
    lay = importlib.reload(layout)
    f = lay.load_frame("precipw", frame=2, piece=3)
    assert f.dtype == np.float32 and f.shape == (4, 6) and f[0, 0] == 2
    monkeypatch.delenv("LATENTSKY_PIECES")
    importlib.reload(layout)


def test_grid_lines_counts():
    fig, ax = plt.subplots()
    nr, nc = grid_lines(ax)
    assert (nr, nc) == (2, 11)
    assert len(ax.lines) == (nr + 1) + (nc + 1)
    plt.close(fig)


def test_add_coastlines_with_given_mask():
    fig, ax = plt.subplots()
    mask = np.zeros((layout.H, layout.W), bool)
    mask[100:200, 300:600] = True
    add_coastlines(ax, mask=mask)
    assert ax.collections or ax.get_children()
    plt.close(fig)


def test_to_mm_detects_metres():
    assert to_mm(np.array([0.0, 0.05]))[1] == pytest.approx(50.0)
    mm = np.array([0.0, 12.0])
    assert to_mm(mm) is mm


def test_crop_renames_sorts_and_cuts():
    lat = np.arange(90, -91, -10.0)
    da = xr.DataArray(np.arange(lat.size, dtype=float), coords={"latitude": lat},
                      dims="latitude")
    out = crop(da)
    assert "lat" in out.coords
    assert out.lat.values[0] == -40 and out.lat.values[-1] == 40
    assert np.all(np.diff(out.lat.values) > 0)


def test_rain_from_z_inverts_log_space():
    r = np.array([0.0, 0.5, 3.0, 40.0])
    z = np.log1p(r) / layout.LOG_RAIN_SCALE
    assert np.allclose(layout.rain_from_z(z), r)
    assert layout.rain_from_z(np.array([-1.0]))[0] == 0.0


def test_tiles_and_mosaic_invert_each_other():
    x = np.random.default_rng(0).random((3, layout.H, layout.W)).astype(np.float32)
    b = layout.tiles(x)
    assert b.shape == (22, 3, layout.P, layout.P)
    assert np.array_equal(b[4, 1], x[1, 0:128, 512:640])
    assert np.array_equal(layout.mosaic(b[:, :1], 2, 11), x[0, :256, :1408])
