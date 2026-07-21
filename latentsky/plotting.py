"""Plotting helpers used across the analysis notebooks.

Cherry-picked from tzhou/LatentSky/util_LS.py:
    plot_two_histograms, plot_pw_tsfc_diff, plot_regional_view,
    build_contingency_table, plot_contingency_heatmap.

All functions take numpy arrays — no torch, no model. Notebooks call
predict_piece(...) to get (X, Y, Y_pred), then pass those into these.
"""

from __future__ import annotations

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# SAM data is cropped to lat in [-40, 40] deg, lon in [0, 360] deg
# (320 x 1440 grid, 0.25 deg cells). Override per call if needed.
_DEFAULT_LAT_EXTENT = (-40.0, 40.0)
_DEFAULT_LON_EXTENT = (0.0, 360.0)


def _grid_latlon(ny: int, nx: int, lat_extent=_DEFAULT_LAT_EXTENT, lon_extent=_DEFAULT_LON_EXTENT):
    """Cell-center lat/lon for the full (ny, nx) grid."""
    lat = lat_extent[0] + (np.arange(ny) + 0.5) * (lat_extent[1] - lat_extent[0]) / ny
    lon = lon_extent[0] + (np.arange(nx) + 0.5) * (lon_extent[1] - lon_extent[0]) / nx
    return lat, lon


def plot_two_histograms(arr1, arr2, varname: str, bins: int = 80, labels=("Original", "Reconstructed")):
    data1 = arr1.ravel()
    data2 = arr2.ravel()
    gmin = min(data1.min(), data2.min())
    gmax = max(data1.max(), data2.max())
    bw = (gmax - gmin) / bins
    bin_edges = np.arange(np.floor(gmin / bw) * bw, np.ceil(gmax / bw) * bw + bw, bw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log in zip(axes, [False, True]):
        ax.hist(data1, bins=bin_edges, alpha=0.5, label=labels[0], edgecolor="black", log=log)
        ax.hist(data2, bins=bin_edges, alpha=0.5, label=labels[1], edgecolor="black", log=log)
        ax.set_xlabel("value")
        ax.set_ylabel("count" + (" (log)" if log else ""))
        ax.set_title(f"{varname}" + (" — log" if log else ""))
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_pw_tsfc_error(X, Y_true, Y_pred, pw_chan: int = 0, tsfc_chan: int = 1,
                       nbins=(50, 50), scale: float = 0.16):
    """Mean prediction error binned over (PW, T2m) phase space. Single panel,
    colorbar saturated at ±`scale`.
    """
    err = (Y_pred - Y_true).ravel()
    pw = X[:, pw_chan, :, :].ravel()
    tsfc = X[:, tsfc_chan, :, :].ravel()
    pw_edges = np.linspace(pw.min(), pw.max(), nbins[0] + 1)
    t_edges = np.linspace(tsfc.min(), tsfc.max(), nbins[1] + 1)

    sum_err, _, _ = np.histogram2d(tsfc, pw, bins=[t_edges, pw_edges], weights=err)
    count, _, _ = np.histogram2d(tsfc, pw, bins=[t_edges, pw_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = sum_err / count
        avg[count == 0] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        avg.T,
        origin="lower",
        extent=[t_edges[0], t_edges[-1], pw_edges[0], pw_edges[-1]],
        aspect="auto",
        cmap="RdBu",
        vmin=-scale,
        vmax=scale,
    )
    ax.set_xlabel("T2m (K)")
    ax.set_ylabel("PW (mm)")
    ax.set_title(f"mean(pred − true), saturated ±{scale:.2f}")
    plt.colorbar(im, ax=ax, extend="both")
    plt.tight_layout()
    plt.show()


def plot_regional_view(Y_true, Y_pred, lat_range, lon_range, figwidth: float = 10.0,
                       timesteps: int = 1, diff_sat: float = 0.5,
                       lat_extent=_DEFAULT_LAT_EXTENT, lon_extent=_DEFAULT_LON_EXTENT):
    """Vertical 3-row stack (truth / pred / diff) on a PlateCarree projection
    with coastlines, cropped to (lat_range, lon_range).

    Truth and pred share one precip colorbar; diff gets its own (RdBu_r:
    positive = red, negative = blue, saturated at ±`diff_sat`).
    """
    n = timesteps
    ny_full, nx_full = Y_true.shape[-2], Y_true.shape[-1]
    lat_full, lon_full = _grid_latlon(ny_full, nx_full, lat_extent, lon_extent)
    lat_s = lat_full[lat_range[0]:lat_range[1]]
    lon_s = lon_full[lon_range[0]:lon_range[1]]

    panel_aspect = abs(lat_s[-1] - lat_s[0]) / abs(lon_s[-1] - lon_s[0])
    panel_h = figwidth * panel_aspect
    fig_h = 3 * panel_h + 1.5

    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(3, n, figsize=(figwidth, fig_h),
                             subplot_kw={"projection": proj},
                             squeeze=False, constrained_layout=True)

    precip_cmap = cmocean.cm.rain
    vmin_log = 0.1
    diff_levels = np.linspace(-diff_sat, diff_sat, 21)
    for j in range(n):
        orig = Y_true[j, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
        rec = Y_pred[j, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
        diff = rec - orig
        vmax = float(max(orig.max(), rec.max(), vmin_log * 10))
        norm = LogNorm(vmin=vmin_log, vmax=vmax)
        levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 20)

        cf = axs[0, j].contourf(lon_s, lat_s, orig, levels=levels, cmap=precip_cmap,
                                 norm=norm, extend="both", transform=proj)
        axs[0, j].coastlines(resolution="110m", linewidth=0.5)
        axs[0, j].set_extent([lon_s[0], lon_s[-1], lat_s[0], lat_s[-1]], crs=proj)
        axs[0, j].set_title(f"sample {j+1}: truth")

        axs[1, j].contourf(lon_s, lat_s, rec, levels=levels, cmap=precip_cmap,
                            norm=norm, extend="both", transform=proj)
        axs[1, j].coastlines(resolution="110m", linewidth=0.5)
        axs[1, j].set_extent([lon_s[0], lon_s[-1], lat_s[0], lat_s[-1]], crs=proj)
        axs[1, j].set_title("prediction")
        fig.colorbar(cf, ax=[axs[0, j], axs[1, j]], orientation="vertical",
                     fraction=0.025, pad=0.02, label="precip (log, mm)")

        cf = axs[2, j].contourf(lon_s, lat_s, diff, levels=diff_levels, cmap="RdBu_r",
                                 extend="both", transform=proj)
        axs[2, j].coastlines(resolution="110m", linewidth=0.5)
        axs[2, j].set_extent([lon_s[0], lon_s[-1], lat_s[0], lat_s[-1]], crs=proj)
        axs[2, j].set_title("pred − truth")
        fig.colorbar(cf, ax=axs[2, j], orientation="vertical",
                     fraction=0.025, pad=0.02, label=f"pred − truth (±{diff_sat})")
    plt.show()


def plot_regional_with_inputs(X, Y_true, Y_pred, lat_range, lon_range,
                              sample_index: int = 0, figwidth: float = 14.0,
                              diff_sat: float = 0.5,
                              lat_extent=_DEFAULT_LAT_EXTENT,
                              lon_extent=_DEFAULT_LON_EXTENT):
    """Two-row regional figure on PlateCarree with coastlines.

    Row 1: input channels 0..2 of X (assumed precipw, t2m, olrtoa).
    Row 2: truth / pred / (pred - truth) for the precip target. Diff
    colorbar saturates at ±`diff_sat`.

    Suptitle reports the cropped sample size in pixels and the geographic
    extent inferred from `lat_extent` / `lon_extent`.
    """
    lat_slice = slice(lat_range[0], lat_range[1])
    lon_slice = slice(lon_range[0], lon_range[1])
    yt = Y_true[sample_index, 0, lat_slice, lon_slice]
    yp = Y_pred[sample_index, 0, lat_slice, lon_slice]
    diff = yp - yt
    pw = X[sample_index, 0, lat_slice, lon_slice]
    t2m = X[sample_index, 1, lat_slice, lon_slice]
    olr = X[sample_index, 2, lat_slice, lon_slice]
    ny, nx = yt.shape

    ny_full, nx_full = Y_true.shape[-2], Y_true.shape[-1]
    lat_full, lon_full = _grid_latlon(ny_full, nx_full, lat_extent, lon_extent)
    lat_s = lat_full[lat_slice]
    lon_s = lon_full[lon_slice]

    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(2, 3, figsize=(figwidth, 8),
                             subplot_kw={"projection": proj},
                             constrained_layout=True)

    def _decorate(ax):
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.set_extent([lon_s[0], lon_s[-1], lat_s[0], lat_s[-1]], crs=proj)

    im = axs[0, 0].contourf(lon_s, lat_s, pw, levels=20, cmap=cmocean.cm.rain, transform=proj)
    axs[0, 0].set_title("input: precipw (mm)")
    _decorate(axs[0, 0])
    fig.colorbar(im, ax=axs[0, 0], fraction=0.04, pad=0.02)

    im = axs[0, 1].contourf(lon_s, lat_s, t2m, levels=20, cmap=cmocean.cm.thermal, transform=proj)
    axs[0, 1].set_title("input: t2m (K)")
    _decorate(axs[0, 1])
    fig.colorbar(im, ax=axs[0, 1], fraction=0.04, pad=0.02)

    im = axs[0, 2].contourf(lon_s, lat_s, olr, levels=20, cmap="gray_r", transform=proj)
    axs[0, 2].set_title("input: olrtoa (W/m²)")
    _decorate(axs[0, 2])
    fig.colorbar(im, ax=axs[0, 2], fraction=0.04, pad=0.02)

    vmin_log = 0.1
    vmax = float(max(yt.max(), yp.max(), vmin_log * 10))
    norm = LogNorm(vmin=vmin_log, vmax=vmax)
    levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 20)

    cf = axs[1, 0].contourf(lon_s, lat_s, yt, levels=levels, cmap=cmocean.cm.rain,
                             norm=norm, extend="both", transform=proj)
    axs[1, 0].set_title("truth: precip")
    _decorate(axs[1, 0])
    axs[1, 1].contourf(lon_s, lat_s, yp, levels=levels, cmap=cmocean.cm.rain,
                        norm=norm, extend="both", transform=proj)
    axs[1, 1].set_title("prediction")
    _decorate(axs[1, 1])
    fig.colorbar(cf, ax=[axs[1, 0], axs[1, 1]], fraction=0.04, pad=0.02, label="precip (log)")

    diff_levels = np.linspace(-diff_sat, diff_sat, 21)
    cf = axs[1, 2].contourf(lon_s, lat_s, diff, levels=diff_levels, cmap="RdBu_r",
                             extend="both", transform=proj)
    axs[1, 2].set_title("pred − truth")
    _decorate(axs[1, 2])
    fig.colorbar(cf, ax=axs[1, 2], fraction=0.04, pad=0.02, label=f"diff (saturated ±{diff_sat})")

    fig.suptitle(
        f"sample {sample_index + 1} — region {ny}×{nx} px "
        f"(lat {lat_s[0]:.1f}°/{lat_s[-1]:.1f}°, lon {lon_s[0]:.1f}°/{lon_s[-1]:.1f}°)",
        fontsize=12,
    )
    plt.show()


def plot_regional_with_all_inputs(X, Y_true, Y_pred, lat_range, lon_range,
                                   sample_index: int = 0, figwidth: float = 14.0,
                                   diff_sat: float = 0.5,
                                   lat_extent=_DEFAULT_LAT_EXTENT,
                                   lon_extent=_DEFAULT_LON_EXTENT):
    """5-row × 3-col regional figure for the 9-channel mseuv setup.

    Row 1: precipw, t2m, olrtoa
    Row 2: mse_850hPa, mse_500hPa, (empty)
    Row 3: uzonal_850hPa, uzonal_500hPa, (empty)
    Row 4: umeridional_850hPa, umeridional_500hPa, (empty)
    Row 5: truth, prediction, diff (RdBu_r, saturated ±diff_sat)
    """
    lat_slice = slice(lat_range[0], lat_range[1])
    lon_slice = slice(lon_range[0], lon_range[1])
    s = sample_index

    yt = Y_true[s, 0, lat_slice, lon_slice]
    yp = Y_pred[s, 0, lat_slice, lon_slice]
    diff = yp - yt
    ny, nx = yt.shape

    ny_full, nx_full = Y_true.shape[-2], Y_true.shape[-1]
    lat_full, lon_full = _grid_latlon(ny_full, nx_full, lat_extent, lon_extent)
    lat_s = lat_full[lat_slice]
    lon_s = lon_full[lon_slice]

    panel_aspect = abs(lat_s[-1] - lat_s[0]) / abs(lon_s[-1] - lon_s[0])
    panel_h = (figwidth / 3) * panel_aspect
    fig_h = 5 * panel_h + 2.0

    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(5, 3, figsize=(figwidth, fig_h),
                             subplot_kw={"projection": proj},
                             constrained_layout=True)

    def _decorate(ax):
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.set_extent([lon_s[0], lon_s[-1], lat_s[0], lat_s[-1]], crs=proj)

    # Each entry: (row, col, channel_index, cmap, title, symmetric_about_zero)
    panels = [
        (0, 0, 0, cmocean.cm.rain, "precipw (mm)", False),
        (0, 1, 1, cmocean.cm.thermal, "t2m (K)", False),
        (0, 2, 2, "gray_r", "olrtoa (W/m²)", False),
        (1, 0, 4, cmocean.cm.thermal, "mse_850hPa", False),
        (1, 1, 3, cmocean.cm.thermal, "mse_500hPa", False),
        (2, 0, 6, "RdBu_r", "uzonal_850hPa (m/s)", True),
        (2, 1, 5, "RdBu_r", "uzonal_500hPa (m/s)", True),
        (3, 0, 8, "RdBu_r", "umeridional_850hPa (m/s)", True),
        (3, 1, 7, "RdBu_r", "umeridional_500hPa (m/s)", True),
    ]
    for r, c, ch, cmap, title, symmetric in panels:
        ax = axs[r, c]
        data = X[s, ch, lat_slice, lon_slice]
        if symmetric:
            m = float(max(abs(data.min()), abs(data.max()), 1e-6))
            levels = np.linspace(-m, m, 21)
            im = ax.contourf(lon_s, lat_s, data, levels=levels, cmap=cmap,
                             extend="both", transform=proj)
        else:
            im = ax.contourf(lon_s, lat_s, data, levels=20, cmap=cmap, transform=proj)
        ax.set_title(title)
        _decorate(ax)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Empty cells on the right of rows 2-4.
    for r in (1, 2, 3):
        axs[r, 2].set_visible(False)

    # Row 5: truth | pred | diff (precip).
    vmin_log = 0.1
    vmax = float(max(yt.max(), yp.max(), vmin_log * 10))
    norm = LogNorm(vmin=vmin_log, vmax=vmax)
    levels = np.logspace(np.log10(vmin_log), np.log10(vmax), 20)

    cf = axs[4, 0].contourf(lon_s, lat_s, yt, levels=levels, cmap=cmocean.cm.rain,
                             norm=norm, extend="both", transform=proj)
    axs[4, 0].set_title("truth: precip")
    _decorate(axs[4, 0])
    axs[4, 1].contourf(lon_s, lat_s, yp, levels=levels, cmap=cmocean.cm.rain,
                        norm=norm, extend="both", transform=proj)
    axs[4, 1].set_title("prediction")
    _decorate(axs[4, 1])
    fig.colorbar(cf, ax=[axs[4, 0], axs[4, 1]], fraction=0.04, pad=0.02,
                 label="precip (log, mm)")

    diff_levels = np.linspace(-diff_sat, diff_sat, 21)
    cf = axs[4, 2].contourf(lon_s, lat_s, diff, levels=diff_levels, cmap="RdBu_r",
                             extend="both", transform=proj)
    axs[4, 2].set_title("pred − truth")
    _decorate(axs[4, 2])
    fig.colorbar(cf, ax=axs[4, 2], fraction=0.04, pad=0.02,
                 label=f"diff (±{diff_sat})")

    fig.suptitle(
        f"sample {s + 1} — region {ny}×{nx} px "
        f"(lat {lat_s[0]:.1f}°/{lat_s[-1]:.1f}°, lon {lon_s[0]:.1f}°/{lon_s[-1]:.1f}°)",
        fontsize=12,
    )
    plt.show()


def build_contingency_table(y_true, y_pred, nbins: int = 10, bins=None):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if bins is None:
        vmin = min(yt.min(), yp.min())
        vmax = max(yt.max(), yp.max())
        bins = np.linspace(vmin, vmax, nbins + 1)
    else:
        bins = np.asarray(bins)
        nbins = len(bins) - 1
    confusion, _, _ = np.histogram2d(yt, yp, bins=[bins, bins])
    confusion = confusion.astype(np.int64)

    row_sums = confusion.sum(axis=1).reshape(-1, 1)
    col_sums = confusion.sum(axis=0).reshape(1, -1)
    total = np.array([[confusion.sum()]])
    aug = np.block([[confusion, row_sums], [col_sums, total]])
    row_labels = [f"O[{bins[i]:.3g},{bins[i+1]:.3g})" for i in range(nbins)] + ["Total"]
    col_labels = [f"P[{bins[i]:.3g},{bins[i+1]:.3g})" for i in range(nbins)] + ["Total"]
    df = pd.DataFrame(aug, index=row_labels, columns=col_labels).astype(np.int64)
    return df, confusion, bins


def plot_contingency_heatmap(confusion, bins, title: str = "Contingency: truth vs pred",
                              n_ticks: int = 7):
    """Plot a contingency heatmap on bin-index axes, with the actual bin edges
    shown as tick labels. Works for both uniform and non-uniform (e.g. log-spaced)
    bins — each cell is drawn at equal visual size regardless of bin width.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    n_bins = confusion.shape[0]
    with np.errstate(divide="ignore"):
        log_conf = np.log(np.where(confusion > 0, confusion, np.nan))
    im = ax.imshow(log_conf, origin="upper", aspect="auto")
    ax.plot(np.arange(n_bins), np.arange(n_bins), "o", mfc="none", mec="red", ms=4, zorder=3)
    tick_idx = np.unique(np.linspace(0, n_bins, n_ticks).astype(int))
    ax.set_xticks(tick_idx - 0.5)
    ax.set_yticks(tick_idx - 0.5)
    ax.set_xticklabels([f"{bins[i]:.2g}" for i in tick_idx], rotation=45)
    ax.set_yticklabels([f"{bins[i]:.2g}" for i in tick_idx])
    ax.set_xlabel("predicted (mm)")
    ax.set_ylabel("truth (mm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="log(count)")
    plt.tight_layout()
    plt.show()
