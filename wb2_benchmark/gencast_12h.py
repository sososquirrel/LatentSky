"""GenCast at ITS shortest lead (+12 h), 2020-08-15 — the diffusion model as
the closest operational cousin of our flow: ensemble mean vs single members
(per-draw), with GraphCast (6h+12h accumulations summed) and the 12-h
persistence null on the same day. Threshold 4.8 mm/12h (rate equivalent),
IoU >= 0.1. Outputs fig_gencast_12h.png + printed bands."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from latentsky.metrics import cell_area_km2, match_systems
from latentsky.comparison.wb2 import crop

CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "wb2_benchmark"
DAY = "2020-08-15"
MEMBERS = [0, 1, 2]
THR, IOU_MIN = 4.8, 0.1
BANDS = np.geomspace(30, 3000, 11)
MIDS = np.sqrt(BANDS[:-1] * BANDS[1:])
ERA5 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))




def fetch():
    fn = CACHE / f"wb2_12h_models_{DAY}.nc"
    if fn.exists():
        return xr.open_dataset(fn)
    so = {"token": "anon"}
    lead = np.timedelta64(12, "h")
    g = xr.open_zarr("gs://weatherbench2/datasets/gencast/2020-1440x721.zarr",
                     storage_options=so, decode_timedelta=True)
    inits = g.time.sel(time=DAY).values
    memb = crop(g["total_precipitation_12hr"].sel(
        time=inits, prediction_timedelta=lead).isel(sample=MEMBERS)).load()
    gm = xr.open_zarr("gs://weatherbench2/datasets/gencast/2020-1440x721_mean.zarr",
                      storage_options=so, decode_timedelta=True)
    mean = crop(gm["total_precipitation_12hr"].sel(
        time=inits, prediction_timedelta=lead)).load()
    gc = xr.open_zarr("gs://weatherbench2/datasets/graphcast/2020/"
                      "date_range_2019-11-16_2021-02-01_12_hours.zarr",
                      storage_options=so, decode_timedelta=True)
    gc12 = crop(gc["total_precipitation_6hr"].sel(
        time=inits, prediction_timedelta=[np.timedelta64(6, "h"),
                                          lead]).sum("prediction_timedelta")).load()
    fx = xr.open_zarr("gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
                      storage_options=so, decode_timedelta=True)
    fx12 = crop(fx["total_precipitation_6hr"].sel(
        time=inits, prediction_timedelta=[np.timedelta64(6, "h"),
                                          lead]).sum("prediction_timedelta")).load()
    era = xr.open_zarr(ERA5, storage_options=so, decode_timedelta=True)
    tp = crop(era["total_precipitation_6hr"])
    def acc12(t_end):
        return (tp.sel(time=t_end).values
                + tp.sel(time=t_end - np.timedelta64(6, "h")).values)
    truth = np.stack([acc12(t + lead) for t in inits])
    persist = np.stack([acc12(t) for t in inits])
    out = xr.Dataset({
        "member": (("init", "sample", "lat", "lon"),
                   memb.transpose("time", "sample", "lat", "lon")
                       .values.astype(np.float32)),
        "mean": (("init", "lat", "lon"), mean.values.astype(np.float32)),
        "graphcast": (("init", "lat", "lon"), gc12.values.astype(np.float32)),
        "fuxi": (("init", "lat", "lon"), fx12.values.astype(np.float32)),
        "truth": (("init", "lat", "lon"), truth.astype(np.float32)),
        "persistence": (("init", "lat", "lon"), persist.astype(np.float32)),
    }, coords={"init": inits, "sample": MEMBERS,
               "lat": memb.lat.values, "lon": memb.lon.values})
    out.to_netcdf(fn)
    print(f"fetched {len(inits)} inits, {len(MEMBERS)} members")
    return out


def pod_far_bands(dfs):
    df = pd.concat(dfs, ignore_index=True)
    b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
    pod = np.full(len(MIDS), np.nan)
    far = np.full(len(MIDS), np.nan)
    ns = np.zeros(len(MIDS), int)
    for k in range(len(MIDS)):
        g = df[(b == k) & df.role.isin(["match", "miss"])]
        gf = df[(b == k) & df.role.isin(["match", "false_alarm"])]
        if len(g):
            pod[k] = (g.role == "match").mean(); ns[k] = len(g)
        if len(gf):
            far[k] = (gf.role == "false_alarm").mean()
    return pod, far, ns


def main():
    ds = fetch()
    area_row = cell_area_km2(ds.lat.size, ds.lon.size,
                             lat_extent=(float(ds.lat[0]) - 0.125,
                                         float(ds.lat[-1]) + 0.125))

    def to_mm(a):
        return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a

    truths = [to_mm(ds.truth.values[i]) for i in range(ds.init.size)]

    def score(fields):
        return pod_far_bands([match_systems(yt, pf, THR, area_row,
                                            iou_min=IOU_MIN)
                              for yt, pf in zip(truths, fields)])

    series = {
        "graphcast (+12 h)": [score([to_mm(ds.graphcast.values[i])
                                     for i in range(ds.init.size)])],
        "fuxi (+12 h)": [score([to_mm(ds.fuxi.values[i])
                                for i in range(ds.init.size)])],
        "gencast mean": [score([to_mm(ds["mean"].values[i])
                                for i in range(ds.init.size)])],
        "gencast single member (per-draw ×3)": [
            score([to_mm(ds.member.values[i, s])
                   for i in range(ds.init.size)]) for s in range(len(MEMBERS))],
        "persistence": [score([to_mm(ds.persistence.values[i])
                               for i in range(ds.init.size)])],
    }
    NMIN = 20
    colors = {"graphcast (+12 h)": "#1b7837", "fuxi (+12 h)": "#762a83",
              "gencast mean": "#35978f",
              "gencast single member (per-draw ×3)": "#b2182b",
              "persistence": "0.5"}
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.7), sharex=True)
    for name, scores in series.items():
        pod = np.nanmean([s[0] for s in scores], axis=0)
        far = np.nanmean([s[1] for s in scores], axis=0)
        ok = scores[0][2] >= NMIN
        ls = ":" if name == "persistence" else "-"
        mk = "D" if "member" in name else "o"
        axs[0].plot(MIDS[ok], pod[ok], mk, ls=ls, ms=4, color=colors[name],
                    label=name)
        axs[1].plot(MIDS[ok], far[ok], mk, ls="--" if ls == "-" else ":",
                    ms=4, color=colors[name])
        print(name, "POD:", np.round(pod, 3))
        print(name, "FAR:", np.round(far, 3))
    for ax, t in [(axs[0], "POD"), (axs[1], "FAR")]:
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
        ax.set_axisbelow(True)
        ax.set_xlabel("system scale L = √A (km)")
        ax.set_title(t + " (+12 h, 2020-08-15, 4.8 mm/12 h)", fontsize=9)
    axs[0].set_ylabel("object-based rate")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("the diffusion model at its shortest lead: GenCast mean vs "
                 "members vs GraphCast", fontsize=9.5, y=1.02)
    fig.savefig(OUT / "fig_gencast_12h.png", dpi=150, bbox_inches="tight")
    print("saved fig_gencast_12h.png")


if __name__ == "__main__":
    main()
