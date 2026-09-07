"""Sophie's hypothesis: +6 h is simply a hard/noisy target for the global
models and WB2's own headline lead is 24 h. Test: the same object margins at
+24 h over each model's own persistence null, using the cached FULL-MONTH
24-h accumulations (May pipeline): graphcast/fuxi/gencast (2020-08) and
fgn ensemble mean (2022-08). Persistence = the ERA5 24-h accumulation ending
24 h before the valid time. Threshold 9.6 mm/24h (= 0.1 mm/15 min rate
equivalent), IoU >= 0.1. Prints month-pooled POD by band and the mid-scale
(100-400 km) margin per model. CPU ~15 min."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from latentsky.metrics import cell_area_km2, match_systems

CACHE = pathlib.Path("/glade/derecho/scratch/sabramian/wb2_eval")
OUT = pathlib.Path("/glade/u/home/sabramian/LatentSky/wb2_benchmark")
THR, IOU_MIN = 9.6, 0.1
BANDS = [(0, 100, "small<100"), (100, 400, "mid100-400"), (400, 1e9, "large>400")]
MODELS = {"graphcast": "2020-08", "fuxi": "2020-08", "gencast": "2020-08",
          "fgn": "2022-08"}


def to_mm(a):
    return a * 1000.0 if float(np.nanmax(a)) < 2.0 else a


def counts(df):
    out = {}
    for lo, hi, name in BANDS:
        g = df[(df.scale >= lo) & (df.scale < hi)
               & df.role.isin(["match", "miss"])]
        out[name] = (int((g.role == "match").sum()), len(g))
    return out


def main():
    rows = []
    era = {m: xr.open_dataset(CACHE / f"era5_tp24hr_{m}_tropics.nc")
           for m in ("2020-08", "2022-08")}
    for model, month in MODELS.items():
        ds = xr.open_dataset(CACHE / f"{model}_tp24hr_{month}_tropics.nc")
        tr = era[month]
        area_row = cell_area_km2(ds.lat.size, ds.lon.size,
                                 lat_extent=(float(ds.lat[0]) - 0.125,
                                             float(ds.lat[-1]) + 0.125))
        tvals = list(tr.valid_time.values)
        acc = {name: [0, 0] for _, _, name in BANDS}
        accp = {name: [0, 0] for _, _, name in BANDS}
        nused = 0
        for i, vt in enumerate(ds.valid_time.values):
            tprev = vt - np.timedelta64(24, "h")
            if vt not in tvals or tprev not in tvals:
                continue
            pf_raw = ds.total_precipitation_24hr.values[i]
            if np.isnan(pf_raw).all():
                continue  # partial fetches leave all-NaN frames (fgn 2022)
            yt = to_mm(tr.total_precipitation_24hr.sel(valid_time=vt).values)
            pf = to_mm(pf_raw)
            pp = to_mm(tr.total_precipitation_24hr.sel(valid_time=tprev).values)
            for df_, a in [(match_systems(yt, pf, THR, area_row,
                                          iou_min=IOU_MIN), acc),
                           (match_systems(yt, pp, THR, area_row,
                                          iou_min=IOU_MIN), accp)]:
                for name, (nm, ntot) in counts(df_).items():
                    a[name][0] += nm
                    a[name][1] += ntot
            nused += 1
        for _, _, name in BANDS:
            pod = acc[name][0] / max(acc[name][1], 1)
            podp = accp[name][0] / max(accp[name][1], 1)
            rows.append(dict(model=model, band=name, pod=pod, pod_null=podp,
                             margin=pod - podp, n=acc[name][1]))
        print(f"{model} ({month}, {nused} valid times) done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "margins_24h.csv", index=False)
    print("\n+24 h, month-pooled, margin over own persistence null:")
    for _, r in df.iterrows():
        print(f"  {r.model:10s} {r.band:11s}: POD {r.pod:.3f} "
              f"null {r.pod_null:.3f} margin {r.margin:+.3f} (n={r.n})")


if __name__ == "__main__":
    main()
