"""I1 illustration: the object-based POD-by-scale lens applied to WB2 AI
forecasts at +6 h, tropics (40S-40N), one August day.

Models with a +6 h precipitation forecast in WB2: GraphCast (2020-08-15),
FuXi (2020-08-15), FGN (2022-08-15 - the model only covers 2022; labeled).
GenCast steps 12-hourly and has no +6 h forecast - excluded. Truth = ERA5
total_precipitation_6hr at the valid time; null = persistence (the ERA5 6 h
accumulation ending at init, under identical matching). Threshold 0.1 mm/6h,
IoU >= 0.1, greedy one-to-one - the project pipeline unchanged
(src/latentsky/metrics).

Fetch is cached to $SCRATCH/wb2_eval/wb2_6h_{model}_{day}.nc (a few MB).
Outputs: wb2_benchmark/systems_wb2_6h.csv.gz + fig_I1_pod6h.png.
"""
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

REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = REPO / "wb2_benchmark"
ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
ERA5_2022_URL = ("gs://weatherbench2/datasets/era5/"
                 "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
MODELS = {
    "graphcast": dict(
        url="gs://weatherbench2/datasets/graphcast/2020/"
            "date_range_2019-11-16_2021-02-01_12_hours.zarr",
        day="2020-08-15", color="#1b7837"),
    "fuxi": dict(
        url="gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr",
        day="2020-08-15", color="#762a83"),
    "fgn": dict(
        url="gs://weatherbench2/datasets/fgn/2022-1440x721_mean.zarr",
        day="2022-08-15", color="#e08214"),
}
THR, IOU_MIN = 2.4, 0.1  # 2.4 mm/6h = LatentSky 0.1 mm/15min rate equivalent
BANDS = np.geomspace(30, 3000, 11)

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))




def to_mm(a):
    """WB2 precip is meters (ERA5 convention) in most zarrs; some derived vars
    are already mm. Detect: a 6 h tropical-belt max below 2 means meters."""
    mx = float(np.nanmax(a))
    return (a * 1000.0, "m->mm") if mx < 2.0 else (a, "mm")


def fetch(model, cfg):
    fn = CACHE / f"wb2_6h_{model}_{cfg['day']}.nc"
    if fn.exists():
        return xr.open_dataset(fn)
    CACHE.mkdir(parents=True, exist_ok=True)
    so = {"token": "anon"}
    ds = xr.open_zarr(cfg["url"], storage_options=so, decode_timedelta=True)
    lead = np.timedelta64(6, "h")
    inits = ds.time.sel(time=cfg["day"]).values
    fc = crop(ds["total_precipitation_6hr"]).sel(time=inits,
                                                prediction_timedelta=lead).load()
    eurl = ERA5_2022_URL if cfg["day"].startswith("2022") else ERA5_URL
    era = xr.open_zarr(eurl, storage_options=so, decode_timedelta=True)
    tp = crop(era["total_precipitation_6hr"])
    truth = tp.sel(time=inits + lead).load()
    persist = tp.sel(time=inits).load()
    out = xr.Dataset({
        "forecast": (("init", "lat", "lon"), fc.values.astype(np.float32)),
        "truth": (("init", "lat", "lon"), truth.values.astype(np.float32)),
        "persistence": (("init", "lat", "lon"), persist.values.astype(np.float32)),
    }, coords={"init": inits, "lat": fc.lat.values, "lon": fc.lon.values})
    out.to_netcdf(fn)
    print(f"fetched {model}: inits {inits}, grid {fc.shape}")
    return out


def pod_by_band(df):
    b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
    pods, ns = np.full(len(BANDS) - 1, np.nan), np.zeros(len(BANDS) - 1, int)
    for k in range(len(BANDS) - 1):
        g = df[(b == k) & df["role"].isin(["match", "miss"])]
        if len(g):
            pods[k] = (g["role"] == "match").mean()
            ns[k] = len(g)
    return pods, ns


def main():
    rows = []
    for model, cfg in MODELS.items():
        ds = fetch(model, cfg)
        ny = ds.lat.size
        area_row = cell_area_km2(ny, ds.lon.size,
                                 lat_extent=(float(ds.lat[0]) - 0.125,
                                             float(ds.lat[-1]) + 0.125))
        for i in range(ds.init.size):
            yt, note_t = to_mm(ds.truth.values[i])
            for name, raw in [(model, ds.forecast.values[i]),
                              ("persistence-" + cfg["day"][:4],
                               ds.persistence.values[i])]:
                pf, note_p = to_mm(raw)
                dfr = match_systems(yt, pf, THR, area_row, iou_min=IOU_MIN)
                dfr.insert(0, "init", str(ds.init.values[i])[:16])
                dfr.insert(0, "model", name)
                rows.append(dfr)
        fmax = float(np.nanmax(to_mm(ds.forecast.values)[0]))
        print(f"{model} evaluated ({ds.init.size} inits; truth {note_t}; "
              f"max forecast {fmax:.1f} mm)", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "systems_wb2_6h.csv.gz", index=False, compression="gzip")

    mids = np.sqrt(BANDS[:-1] * BANDS[1:])
    NMIN = 20  # a band enters the plot only with >= NMIN truth systems
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.1))
    ax = axs[0]
    persisted = set()
    curves, nulls = {}, {}
    for model, cfg in MODELS.items():
        pods, ns = pod_by_band(df[df.model == model])
        ok = ns >= NMIN
        curves[model] = (pods, ok)
        lbl = model + (" (2022)" if cfg["day"].startswith("2022") else "")
        ax.plot(mids[ok], pods[ok], "o-", ms=4.5, color=cfg["color"], label=lbl)
        pkey = "persistence-" + cfg["day"][:4]
        pp, pn = pod_by_band(df[df.model == pkey])
        nulls[cfg["day"][:4]] = (pp, pn >= NMIN)
        if pkey not in persisted:
            persisted.add(pkey)
            style = dict(color="0.45", ls=":") if cfg["day"].startswith("2020") \
                else dict(color="0.65", ls="--")
            pok = pn >= NMIN
            ax.plot(mids[pok], pp[pok], "s", ms=3.5, lw=1.2, alpha=0.9,
                    label=f"persistence (ERA5, {cfg['day'][:4]})", **style)
        print(model, "POD:", np.round(pods, 3), "n:", ns)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
    ax.set_axisbelow(True)
    ax.set_xlabel("system scale L = √A (km)")
    ax.set_ylabel("POD at +6 h (object-based)")
    ax.set_title("detection by scale (each model on its own day)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # --- right: the fair comparison — skill margin over each model's OWN null
    ax = axs[1]
    for model, cfg in MODELS.items():
        pods, ok = curves[model]
        null, _ = nulls[cfg["day"][:4]]
        lbl = model + (" (2022)" if cfg["day"].startswith("2022") else "")
        ax.plot(mids[ok], (pods - null)[ok], "o-", ms=4.5, color=cfg["color"],
                label=lbl)
    fm = pd.read_csv(OUT / "fgn_member_bands.csv")
    memb = fm[fm.series.str.startswith("FGN single")].sort_values("L_mid")
    ok = memb.n.to_numpy() >= NMIN
    null22 = nulls["2022"][0]
    ax.plot(mids[ok], (memb.pod.to_numpy() - null22)[ok], "D--", ms=4.5,
            color="#b2182b", label="fgn single member (2022)")
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xscale("log")
    ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
    ax.set_axisbelow(True)
    ax.set_xlabel("system scale L = √A (km)")
    ax.set_ylabel("POD − POD of own persistence null")
    ax.set_title("the fair comparison: skill over each model's own null",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("WB2 AI forecasts through the same lens — tropics, one August "
                 "day, ERA5 truth, 2.4 mm/6 h", fontsize=9.5, y=1.0)
    fig.savefig(OUT / "fig_I1_pod6h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_I1_pod6h.png;", len(df), "systems")


if __name__ == "__main__":
    main()
