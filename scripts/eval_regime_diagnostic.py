"""Does LatentSky detect deep/organized (MCS-like) systems better than
shallow/unorganized ones? Tags every truth system (matched or missed) in the
9var(mse+uv) sweep with a physically-derived convective regime, using fields
already in the model's own input set -- no new classifier needed:

  olrtoa                        -- deep-convection proxy (cold cloud top = low OLR)
  mse_850hPa - mse_500hPa       -- instability proxy (low-level MSE excess)
  |shear(850->500)|             -- organization potential (deep-layer shear
                                    magnitude from uzonal/umeridional at 850/500hPa)

Regime fields are sampled at the SAME raw time index and centroid pixel as
the truth system (verification time = frame + delta_index, where delta_index
= 1 + 4*lead_h at 15-min cadence), from the same DYAMOND piece files the
rain fields come from.

Writes docs/scale_systems_regime.csv.gz (systems + regime columns) and
docs/figures/regime_vs_detection.png.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO
PIECES_DIR = pathlib.Path(
    "/glade/derecho/scratch/tzhou/splitted_1440x720_9vars_40pieces_v2"
)
TEST_PIECES = [2, 12, 22, 32]
AUX_VARS = ["olrtoa", "mse_500hPa", "mse_850hPa",
            "uzonal_500hPa", "uzonal_850hPa",
            "umeridional_500hPa", "umeridional_850hPa"]


def load_var(prefix: str, pid: int) -> np.ndarray:
    return np.load(PIECES_DIR / f"{prefix}_piece{pid:02d}.npy", mmap_mode="r").astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems-in", default="docs/scale_systems_v3.csv.gz")
    p.add_argument("--name-pattern", default=r"mseuv_lead[\d.]+h")
    p.add_argument("--out-csv", default="docs/scale_systems_regime.csv.gz")
    p.add_argument("--recompute", action="store_true",
                   help="Force resampling even if --out-csv already exists.")
    args = p.parse_args()

    out_csv = REPO / args.out_csv
    if out_csv.exists() and not args.recompute:
        print(f"loading cached {out_csv}", flush=True)
        df = pd.read_csv(out_csv, compression="gzip")
        _finish(df, out_csv)
        return

    print(f"loading {args.systems_in} ...", flush=True)
    df = pd.read_csv(REPO / args.systems_in, compression="gzip",
                      usecols=["name", "lead_h", "piece", "frame", "role",
                               "scale", "row", "col"])
    df = df[df["name"].str.fullmatch(args.name_pattern)].copy()
    df = df[df["role"].isin(["match", "miss"])].copy()
    df["delta"] = 1 + 4 * df["lead_h"].astype(float)
    df["raw_idx"] = (df["frame"] + df["delta"]).round().astype(int)
    print(f"{len(df)} match/miss systems across {df['piece'].nunique()} pieces", flush=True)

    for v in AUX_VARS:
        df[v] = np.nan

    ny, nx = 720, 1440
    for pid in sorted(df["piece"].unique()):
        m = df["piece"] == pid
        n = int(m.sum())
        print(f"piece {pid}: sampling {n} systems ...", flush=True)
        ridx = df.loc[m, "raw_idx"].to_numpy()
        rrow = np.clip(df.loc[m, "row"].round().to_numpy().astype(int), 0, ny - 1)
        rcol = np.clip(df.loc[m, "col"].round().to_numpy().astype(int), 0, nx - 1)
        ridx = np.clip(ridx, 0, 95)  # piece length is 96 frames
        for v in AUX_VARS:
            arr = load_var(v, pid)
            df.loc[m, v] = arr[ridx, rrow, rcol]

    df["instability_proxy"] = df["mse_850hPa"] - df["mse_500hPa"]
    df["shear_u"] = df["uzonal_850hPa"] - df["uzonal_500hPa"]
    df["shear_v"] = df["umeridional_850hPa"] - df["umeridional_500hPa"]
    df["shear_mag"] = np.sqrt(df["shear_u"] ** 2 + df["shear_v"] ** 2)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, compression="gzip")
    print(f"wrote {out_csv}", flush=True)

    _finish(df, out_csv)


def _finish(df: pd.DataFrame, out_csv: pathlib.Path) -> None:
    print("\n=== median regime value by role (match vs miss), pooled over all leads ===")
    print(df.groupby("role")[["olrtoa", "instability_proxy", "shear_mag", "scale"]].median())

    print("\n=== POD-relevant split: median olrtoa / shear_mag by role x lead bucket ===")
    df["lead_bucket"] = pd.cut(df["lead_h"], [-0.1, 0.5, 2.0, 10.0],
                                labels=["short (<=0.5h)", "mid (0.75-2h)", "long (3-6h)"])
    print(df.groupby(["lead_bucket", "role"], observed=True)[
        ["olrtoa", "instability_proxy", "shear_mag", "scale"]].median())

    # --- figure: regime vs scale (median + IQR band) with the marginal
    # y-distribution attached to the right of each row, match vs miss ---
    fig, axes = plt.subplots(3, 2, figsize=(10, 13), constrained_layout=True,
                              gridspec_kw={"width_ratios": [3.2, 1]})

    bins = np.geomspace(df["scale"].clip(lower=1).min(), df["scale"].max(), 12)
    mids = np.sqrt(bins[:-1] * bins[1:])

    panels = [
        ("olrtoa", "OLR (normalized input units)\nlower = colder top / deeper"),
        ("instability_proxy", "MSE(850)-MSE(500)\n[instability proxy, normalized units]"),
        ("shear_mag", "|shear(850→500)|\n[organization proxy, normalized units]"),
    ]
    for (ax, ax_m), (var, ylabel) in zip(axes, panels):
        for role, color in [("match", "#1f77b4"), ("miss", "#d62728")]:
            sub = df[df["role"] == role]
            b = np.digitize(sub["scale"], bins) - 1
            med = np.array([sub[var][b == k].median() for k in range(len(mids))])
            q25 = np.array([sub[var][b == k].quantile(0.25) for k in range(len(mids))])
            q75 = np.array([sub[var][b == k].quantile(0.75) for k in range(len(mids))])
            ax.fill_between(mids, q25, q75, color=color, alpha=0.15, lw=0)
            ax.plot(mids, med, marker="o", ms=4, lw=1.8, color=color,
                    label=f"{role}  (median, shaded=IQR)")
        ax.set_xscale("log")
        ax.set_xlabel("system scale L [km]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, which="both")

        # marginal: distribution of the variable itself, oriented along the
        # shared y-axis (rotated histogram), match vs miss overlaid.
        ylo, yhi = ax.get_ylim()
        yb = np.linspace(ylo, yhi, 60)
        for role, color in [("match", "#1f77b4"), ("miss", "#d62728")]:
            vals = df.loc[df["role"] == role, var].dropna()
            ax_m.hist(vals, bins=yb, orientation="horizontal", histtype="step",
                      density=True, lw=1.6, color=color)
        ax_m.set_ylim(ylo, yhi)
        ax_m.set_xlabel("density", fontsize=9)
        ax_m.tick_params(labelleft=False)
        ax_m.grid(alpha=0.25)

    fig.suptitle("Convective regime vs system scale, match vs miss\n"
                 "(left: median + IQR band vs scale; right: marginal distribution) — "
                 "9var(mse+uv) sweep, all leads pooled", fontsize=10)
    out_fig = REPO / "docs/figures/regime_vs_detection.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote {out_fig}", flush=True)


if __name__ == "__main__":
    main()
