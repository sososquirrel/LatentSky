"""R^2 (object-level total precip, pred vs truth) vs lead time: LatentSky vs
persistence, climatology, random-null baselines.

Same object-level scoring as plot_r2_vs_lead.py: per detected/missed/false-alarm
system, P_t (truth precip) and P_p (pred precip); misses -> P_p=0, false
alarms -> P_t=0. R^2 = 1 - sum((P_t-P_p)^2) / sum((P_t-mean(P_t))^2).

Baselines are recomputed from the same DYAMOND test pieces used by
eval_baselines.py (persistence/climatology/random null), since that script
only kept scale-binned aggregates, not per-system P_t/P_p needed for R^2.
Writes per-system tables to docs/scale_systems_baselines.csv.gz for reuse.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.scale_analysis import system_table

from latentsky.registry import REPO_ROOT as REPO
PIECES_DIR = pathlib.Path(
    "/glade/derecho/scratch/tzhou/splitted_1440x720_9vars_40pieces_v2"
)
TEST_PIECES = [2, 12, 22, 32]
LEADS = {0.0: 1, 1.0: 5, 2.0: 9, 3.0: 13, 6.0: 25}


def load_rain(pid: int) -> np.ndarray:
    return np.load(PIECES_DIR / f"rain_total_piece{pid:02d}.npy", mmap_mode="r").astype(np.float32)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_baseline_systems(threshold: float, iou_min: float) -> pd.DataFrame:
    print("computing climatological mean ...", flush=True)
    clim_sum, clim_n = None, 0
    for pid in TEST_PIECES:
        Y = load_rain(pid)
        clim_sum = Y.sum(axis=0) if clim_sum is None else clim_sum + Y.sum(axis=0)
        clim_n += Y.shape[0]
    clim_mean = clim_sum / clim_n

    all_frames = []
    for lead_h, delta in LEADS.items():
        print(f"\nlead {lead_h}h  (delta={delta}) ...", flush=True)
        rand_pieces = {
            pid: load_rain(TEST_PIECES[(i + 2) % len(TEST_PIECES)])
            for i, pid in enumerate(TEST_PIECES)
        }
        for pid in TEST_PIECES:
            Y = load_rain(pid)
            Y_rand = rand_pieces[pid]
            T = Y.shape[0] - delta - 1
            if T <= 0:
                continue
            truth = np.ascontiguousarray(Y[delta + 1: delta + 1 + T])
            pers = np.ascontiguousarray(Y[0:T])
            clim = np.broadcast_to(clim_mean[None], truth.shape).copy()
            rand = np.ascontiguousarray(Y_rand[0:T])

            for label, pred in [("persistence", pers), ("climatology", clim), ("random", rand)]:
                df = system_table(truth, pred, threshold=threshold, iou_min=iou_min)
                if df.empty:
                    continue
                df.insert(0, "lead_h", lead_h)
                df.insert(0, "predictor_set", label)
                all_frames.append(df[["predictor_set", "lead_h", "role", "P_t", "P_p",
                                       "scale", "lat", "lon", "lat_p", "lon_p"]])

            print(f"  piece {pid}: T={T}", flush=True)

    return pd.concat(all_frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--iou-min", type=float, default=0.1)
    p.add_argument("--out-systems", default=None)
    p.add_argument("--recompute", action="store_true",
                    help="Force recomputation even if cached systems file exists.")
    args = p.parse_args()

    out_systems = pathlib.Path(args.out_systems or REPO / "docs/scale_systems_baselines.csv.gz")

    if out_systems.exists() and not args.recompute:
        print(f"loading cached {out_systems}", flush=True)
        base_df = pd.read_csv(out_systems, compression="gzip")
    else:
        base_df = compute_baseline_systems(args.threshold, args.iou_min)
        out_systems.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_csv(out_systems, index=False, compression="gzip")
        print(f"wrote {out_systems}", flush=True)

    base_df["P_t"] = base_df["P_t"].fillna(0.0)
    base_df["P_p"] = base_df["P_p"].fillna(0.0)

    # --- LatentSky, from the fair-sampling per-system table (v5: full test-
    # piece frame coverage, matching the baselines above -- NOT the 8-frame
    # default used by v3/v4). "9var (mse+uv)" predictor_set collides with
    # rain-input/residual variants in v5 -- filter by `name` to keep only the
    # plain mseuv_lead* family. ---
    ls_full = pd.read_csv(REPO / "docs/scale_systems_v5_fullframes.csv.gz", compression="gzip",
                          usecols=["name", "lead_h", "role", "P_t", "P_p", "lat", "lon", "lat_p", "lon_p"])
    ls_full = ls_full[ls_full["name"].str.fullmatch(r"mseuv_lead[\d.]+h")].copy()

    ls_df = ls_full[["lead_h", "P_t", "P_p"]].copy()
    ls_df["P_t"] = ls_df["P_t"].fillna(0.0)
    ls_df["P_p"] = ls_df["P_p"].fillna(0.0)

    ls_cent = ls_full[["lead_h", "role", "lat", "lon", "lat_p", "lon_p"]].copy()

    series = {
        "LatentSky": (sorted(ls_df["lead_h"].unique()), ls_df),
        "persistence": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "persistence"]),
        "climatology": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "climatology"]),
        "random": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "random"]),
    }
    style = {
        "LatentSky":   dict(color="#1f77b4", marker="o", ls="-"),
        "persistence": dict(color="#ff7f0e", marker="^", ls="--"),
        "climatology": dict(color="0.2",     marker="s", ls="-."),
        "random":      dict(color="0.6",     marker="x", ls=":"),
    }

    fig, (ax_r2, ax_rmse, ax_disp) = plt.subplots(1, 3, figsize=(19.5, 5.2), constrained_layout=True)
    print("\n=== R^2 (object-level) vs lead ===")
    for name, (leads, df) in series.items():
        r2_vals = []
        for L in leads:
            sub = df[df["lead_h"] == L]
            r2_vals.append(r2(sub["P_t"].to_numpy(), sub["P_p"].to_numpy()))
        ax_r2.plot(leads, r2_vals, lw=1.8, ms=7, label=name, **style[name])
        print(f"{name:>12}: " + "  ".join(f"{L:g}h={v:.3f}" for L, v in zip(leads, r2_vals)))

    print("\n=== RMSE (object-level total precip P, same units as P_t/P_p) vs lead ===")
    for name, (leads, df) in series.items():
        rmse_vals = []
        for L in leads:
            sub = df[df["lead_h"] == L]
            rmse_vals.append(rmse(sub["P_t"].to_numpy(), sub["P_p"].to_numpy()))
        ax_rmse.plot(leads, rmse_vals, lw=1.8, ms=7, label=name, **style[name])
        print(f"{name:>12}: " + "  ".join(f"{L:g}h={v:.3g}" for L, v in zip(leads, rmse_vals)))

    ax_r2.axhline(0, color="0.4", lw=1.0, ls=":")
    ax_r2.text(6, 0.02, "R²=0 (no better than predicting the mean)", ha="right", fontsize=8, color="0.4")
    ax_r2.set_xlabel("forecast lead  [h]", fontsize=11)
    ax_r2.set_ylabel("R²  (object-level total precip, pred vs truth)", fontsize=11)
    ax_r2.set_title("(a) R² vs lead", fontsize=11, loc="left", fontweight="bold")
    ax_r2.legend(fontsize=9, frameon=False)
    ax_r2.grid(alpha=0.3)

    ax_rmse.set_xlabel("forecast lead  [h]", fontsize=11)
    ax_rmse.set_ylabel("RMSE  (object-level total precip P, mm·km²)", fontsize=11)
    ax_rmse.set_title("(b) RMSE vs lead", fontsize=11, loc="left", fontweight="bold")
    ax_rmse.legend(fontsize=9, frameon=False)
    ax_rmse.grid(alpha=0.3)

    # --- (c) centroid displacement error for matched systems: is the model
    # placing the system where it actually ended up, or where it started
    # (persistence's implicit assumption -- no advection)? ---
    disp_series = {
        "LatentSky":   (sorted(ls_cent["lead_h"].unique()), ls_cent),
        "persistence": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "persistence"]),
        "climatology": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "climatology"]),
        "random":      (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "random"]),
    }
    print("\n=== median centroid displacement (km, matched systems only) vs lead ===")
    for name, (leads, df) in disp_series.items():
        m = df[df["role"] == "match"]
        disp_vals = []
        for L in leads:
            sub = m[m["lead_h"] == L]
            if sub.empty:
                disp_vals.append(np.nan)
                continue
            d = haversine_km(sub["lat"].to_numpy(), sub["lon"].to_numpy(),
                              sub["lat_p"].to_numpy(), sub["lon_p"].to_numpy())
            disp_vals.append(float(np.median(d)))
        ax_disp.plot(leads, disp_vals, lw=1.8, ms=7, label=name, **style[name])
        print(f"{name:>12}: " + "  ".join(
            f"{L:g}h={v:.1f}" if np.isfinite(v) else f"{L:g}h=--"
            for L, v in zip(leads, disp_vals)))

    ax_disp.set_xlabel("forecast lead  [h]", fontsize=11)
    ax_disp.set_ylabel("median centroid displacement  [km]", fontsize=11)
    ax_disp.set_title("(c) Matched-system position error vs lead", fontsize=11, loc="left", fontweight="bold")
    ax_disp.legend(fontsize=9, frameon=False)
    ax_disp.grid(alpha=0.3)

    fig.suptitle("LatentSky vs persistence / climatology / random null\n"
                 "P per system, misses→P_p=0, false alarms→P_t=0", fontsize=10)

    out = REPO / "docs/figures/r2_rmse_displacement_vs_lead_baselines_v5fair.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
