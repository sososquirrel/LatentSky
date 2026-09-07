"""The scoring-rule test on an operational model: FGN at +6 h, ensemble MEAN
vs individual MEMBERS, per-draw (FRAME's ensemble rule), one August day.

Prediction (registered in spirit with our R3): single members behave like our
flow — small-scale POD lifted above the mean's, at a FAR cost — while the
mean behaves like a regression. Members: samples 0/1/2 of the 64-member
zarr, each scored separately, band scores averaged (mean of scores, never
score of mean). Truth/persistence reuse the cached fgn fetch. Threshold
2.4 mm/6h, IoU >= 0.1. Outputs: fig_fgn_members.png + printed table."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from latentsky.metrics import cell_area_km2, match_systems

REPO = pathlib.Path("/glade/u/home/sabramian/LatentSky")
CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = REPO / "wb2_benchmark"
DAY = "2022-08-15"
MEMBERS = [0, 1, 2]
THR, IOU_MIN = 2.4, 0.1
BANDS = np.geomspace(30, 3000, 11)

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def fetch_members():
    fn = CACHE / f"wb2_6h_fgn_members_{DAY}.nc"
    if fn.exists():
        return xr.open_dataset(fn)
    ds = xr.open_zarr("gs://weatherbench2/datasets/fgn/2022-1440x721.zarr",
                      storage_options={"token": "anon"}, decode_timedelta=True)
    inits = ds.time.sel(time=DAY).values
    da = ds["total_precipitation_6hr"].sel(
        time=inits, prediction_timedelta=np.timedelta64(6, "h")
    ).isel(sample=MEMBERS)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    da = da.sel(lat=slice(-40, 40)).load()
    out = xr.Dataset({"member": (("init", "sample", "lat", "lon"),
                                 da.transpose("time", "sample", "lat", "lon")
                                   .values.astype(np.float32))},
                     coords={"init": inits, "sample": MEMBERS,
                             "lat": da.lat.values, "lon": da.lon.values})
    out.to_netcdf(fn)
    print(f"fetched {len(inits)} inits x {len(MEMBERS)} members")
    return out


def bands_scores(df):
    b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
    pod = np.full(len(BANDS) - 1, np.nan)
    far = np.full(len(BANDS) - 1, np.nan)
    ns = np.zeros(len(BANDS) - 1, int)
    for k in range(len(BANDS) - 1):
        g = df[(b == k) & df.role.isin(["match", "miss"])]
        gf = df[(b == k) & df.role.isin(["match", "false_alarm"])]
        if len(g):
            pod[k] = (g.role == "match").mean(); ns[k] = len(g)
        if len(gf):
            far[k] = (gf.role == "false_alarm").mean()
    return pod, far, ns


def main():
    mem = fetch_members()
    base = xr.open_dataset(CACHE / f"wb2_6h_fgn_{DAY}.nc")  # mean + truth + persistence
    ny = base.lat.size
    area_row = cell_area_km2(ny, base.lon.size,
                             lat_extent=(float(base.lat[0]) - 0.125,
                                         float(base.lat[-1]) + 0.125))

    def to_mm(a):
        return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a

    series = {}
    # mean and persistence: one field per init
    for name, get in [("FGN ensemble mean", lambda i: to_mm(base.forecast.values[i])),
                      ("persistence", lambda i: to_mm(base.persistence.values[i]))]:
        rows = []
        for i in range(base.init.size):
            yt = to_mm(base.truth.values[i])
            rows.append(match_systems(yt, get(i), THR, area_row, iou_min=IOU_MIN))
        series[name] = [bands_scores(pd.concat(rows, ignore_index=True))]
    # members: score EACH separately, then average the scores
    per_member = []
    for s in range(len(MEMBERS)):
        rows = []
        for i in range(base.init.size):
            yt = to_mm(base.truth.values[i])
            pf = to_mm(mem.member.values[i, s])
            rows.append(match_systems(yt, pf, THR, area_row, iou_min=IOU_MIN))
        per_member.append(bands_scores(pd.concat(rows, ignore_index=True)))
        print(f"member {s} scored", flush=True)
    series["FGN single member (per-draw ×3)"] = per_member

    mids = np.sqrt(BANDS[:-1] * BANDS[1:])
    NMIN = 20
    colors = {"FGN ensemble mean": "#e08214",
              "FGN single member (per-draw ×3)": "#b2182b",
              "persistence": "0.5"}
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.6))
    curves = {}
    for name, scores in series.items():
        pod = np.nanmean([s[0] for s in scores], axis=0)
        far = np.nanmean([s[1] for s in scores], axis=0)
        ns = scores[0][2]
        ok = ns >= NMIN
        curves[name] = (pod, far, ok)
        ls = ":" if name == "persistence" else "-"
        axs[0].plot(mids[ok], pod[ok], "o", ls=ls, ms=4, color=colors[name],
                    label=name)
        axs[1].plot(mids[ok], far[ok], "s", ls="--" if ls == "-" else ":",
                    ms=4, color=colors[name])
        print(name, "POD:", np.round(pod, 3))
        print(name, "FAR:", np.round(far, 3))
    for ax, t in [(axs[0], "POD"), (axs[1], "FAR")]:
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
        ax.set_axisbelow(True)
        ax.set_xlabel("system scale L = √A (km)")
        ax.set_title(t + " (+6 h, 2022-08-15)", fontsize=9)
    axs[0].set_ylabel("object-based rate")
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")

    # --- panel 3: the member's detection gain over the mean, explicit
    pod_m, _, ok = curves["FGN single member (per-draw ×3)"]
    pod_mean = curves["FGN ensemble mean"][0]
    d = pod_m - pod_mean
    ax = axs[2]
    bars = ax.bar(np.arange(ok.sum()), d[ok],
                  color=np.where(d[ok] >= 0, "#b2182b", "#2166ac"))
    for x, v in zip(np.arange(ok.sum()), d[ok]):
        if v >= 0:
            ax.text(x, v + 0.003, f"{v:+.2f}", ha="center", fontsize=6.5)
        else:
            ax.text(x, v - 0.003, f"{v:+.2f}", ha="center", va="top",
                    fontsize=6.5)
    ax.set_ylim(min(d[ok].min() - 0.015, -0.03), d[ok].max() + 0.015)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(np.arange(ok.sum()))
    ax.set_xticklabels([f"{m:.0f}" for m in mids[ok]], fontsize=7)
    ax.grid(True, axis="y", ls=":", lw=0.5, color="0.8")
    ax.set_axisbelow(True)
    ax.set_xlabel("system scale L (km)")
    ax.set_ylabel("ΔPOD")
    ax.set_title("member − mean: the detection\nthe mean field hides", fontsize=9)
    fig.suptitle("the ensemble-scoring rule on an operational model: "
                 "FGN mean vs FGN members", fontsize=9.5, y=1.04)
    fig.savefig(OUT / "fig_fgn_members.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_fgn_members.png")


if __name__ == "__main__":
    main()
