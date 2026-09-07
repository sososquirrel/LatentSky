"""fig_pod_scale_leads.png: POD by system scale, +6 h (left) and +24 h
(right), shared axes. 6 h: graphcast/fuxi (2020-08-15), fgn mean + single
member (2022-08-15), nulls. 24 h: graphcast/fuxi/gencast/fgn means
(month-pooled August), nulls. Thresholds rate-equivalent (2.4 mm/6h,
9.6 mm/24h). 24 h band curves computed here (cached to csv)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from latentsky.metrics import cell_area_km2, match_systems

REPO = Path("/glade/u/home/sabramian/LatentSky")
CACHE = Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = REPO / "wb2_benchmark"
BANDS = np.geomspace(30, 3000, 11)
MIDS = np.sqrt(BANDS[:-1] * BANDS[1:])
NMIN = 20
COLORS = {"graphcast": "#1b7837", "fuxi": "#762a83", "gencast": "#35978f",
          "fgn": "#e08214", "fgn-member": "#b2182b",
          "persistence-2020": "0.45", "persistence-2022": "0.65"}

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def pod_bands(df):
    b = np.clip(np.digitize(df["scale"], BANDS) - 1, 0, len(BANDS) - 2)
    pod = np.full(len(MIDS), np.nan)
    ns = np.zeros(len(MIDS), int)
    for k in range(len(MIDS)):
        g = df[(b == k) & df.role.isin(["match", "miss"])]
        if len(g):
            pod[k] = (g.role == "match").mean(); ns[k] = len(g)
    return pod, ns


def compute_24h():
    cachecsv = OUT / "pod_bands_24h.csv"
    if cachecsv.exists():
        return pd.read_csv(cachecsv)
    era = {y: xr.open_dataset(CACHE / f"era5_tp24hr_{y}-08_tropics.nc")
           for y in ("2020", "2022")}

    def to_mm(a):
        return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a

    rows = []
    jobs = [("graphcast", "2020"), ("fuxi", "2020"), ("gencast", "2020"),
            ("fgn", "2022"), ("persistence-2020", "2020"),
            ("persistence-2022", "2022")]
    for model, year in jobs:
        tr = era[year]
        tvals = list(tr.valid_time.values)
        if model.startswith("persistence"):
            ds, is_persist = tr, True
        else:
            ds = xr.open_dataset(CACHE / f"{model}_tp24hr_{year}-08_tropics.nc")
            is_persist = False
        area_row = cell_area_km2(321, 1440, lat_extent=(-40.125, 40.125))
        dfs = []
        for i, vt in enumerate(ds.valid_time.values):
            tprev = vt - np.timedelta64(24, "h")
            if vt not in tvals or tprev not in tvals:
                continue
            if is_persist:
                pf_raw = tr.total_precipitation_24hr.sel(valid_time=tprev).values
            else:
                pf_raw = ds.total_precipitation_24hr.values[i]
            if np.isnan(pf_raw).all():
                continue
            yt = to_mm(tr.total_precipitation_24hr.sel(valid_time=vt).values)
            dfs.append(match_systems(yt, to_mm(pf_raw), 9.6, area_row,
                                     iou_min=0.1))
        pod, ns = pod_bands(pd.concat(dfs, ignore_index=True))
        for k in range(len(MIDS)):
            rows.append(dict(model=model, L_mid=MIDS[k], pod=pod[k], n=ns[k]))
        print(model, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cachecsv, index=False)
    return df


def main():
    d24 = compute_24h()
    d6 = pd.read_csv(OUT / "systems_wb2_6h.csv.gz")
    fm = pd.read_csv(OUT / "fgn_member_bands.csv")

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
    # ---- +6 h
    ax = axs[0]
    for m, lbl in [("graphcast", "graphcast"), ("fuxi", "fuxi"),
                   ("fgn", "fgn mean (2022)"),
                   ("persistence-2020", "persistence 2020"),
                   ("persistence-2022", "persistence 2022")]:
        pod, ns = pod_bands(d6[d6.model == m])
        ok = ns >= NMIN
        ls = ":" if "persistence" in m else "-"
        ax.plot(MIDS[ok], pod[ok], "o", ls=ls, ms=4, color=COLORS[m], label=lbl)
    memb = fm[fm.series.str.startswith("FGN single")].sort_values("L_mid")
    ok = memb.n.to_numpy() >= NMIN
    ax.plot(memb.L_mid.to_numpy()[ok], memb.pod.to_numpy()[ok], "D--", ms=4,
            color=COLORS["fgn-member"], label="fgn member (2022)")
    ax.set_title("+6 h (one August day; 2.4 mm/6 h)", fontsize=9)
    # ---- +24 h
    ax = axs[1]
    for m, lbl in [("graphcast", "graphcast"), ("fuxi", "fuxi"),
                   ("gencast", "gencast mean"), ("fgn", "fgn mean (2022)"),
                   ("persistence-2020", "persistence 2020"),
                   ("persistence-2022", "persistence 2022")]:
        g = d24[d24.model == m].sort_values("L_mid")
        ok = g.n.to_numpy() >= NMIN
        ls = ":" if "persistence" in m else "-"
        ax.plot(g.L_mid.to_numpy()[ok], g.pod.to_numpy()[ok], "o", ls=ls, ms=4,
                color=COLORS[m], label=lbl)
    ax.set_title("+24 h (August month; 9.6 mm/24 h)", fontsize=9)
    for ax in axs:
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
        ax.set_axisbelow(True)
        ax.set_xlabel("system scale L = √A (km)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    axs[0].set_ylabel("POD (object-based)")
    fig.savefig(OUT / "fig_pod_scale_leads.png", dpi=150, bbox_inches="tight")
    print("saved fig_pod_scale_leads.png")


if __name__ == "__main__":
    main()
