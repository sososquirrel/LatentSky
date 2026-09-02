"""Plain pixel-level RMSE/R^2 (prediction vs truth, full rain field, mm
units) vs lead: LatentSky vs persistence/climatology/random-null.

No object decomposition here -- this is the "basic" comparison, scored over
every pixel of every frame (not just detected-system totals). LatentSky's
numbers come from docs/bulk_metrics_v1.csv (scripts/eval_bulk_metrics.py,
run on Casper GPU). The baselines are recomputed directly from the raw
DYAMOND rain fields, reusing the same piece-loading code as eval_baselines.py
but skipping connected-component labeling entirely, so this pass is much
cheaper than the object-level ones.
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
from latentsky.paths import FIGURES_DIR
PIECES_DIR = pathlib.Path(
    "/glade/derecho/scratch/tzhou/splitted_1440x720_9vars_40pieces_v2"
)
TEST_PIECES = [2, 12, 22, 32]
LEADS = {0.0: 1, 1.0: 5, 2.0: 9, 3.0: 13, 6.0: 25}


def load_rain(pid: int) -> np.ndarray:
    return np.load(PIECES_DIR / f"rain_total_piece{pid:02d}.npy", mmap_mode="r").astype(np.float32)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float64)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2, dtype=np.float64)
    return 1.0 - ss_res / ss_tot


def compute_baseline_bulk() -> pd.DataFrame:
    print("computing climatological mean ...", flush=True)
    clim_sum, clim_n = None, 0
    for pid in TEST_PIECES:
        Y = load_rain(pid)
        clim_sum = Y.sum(axis=0) if clim_sum is None else clim_sum + Y.sum(axis=0)
        clim_n += Y.shape[0]
    clim_mean = clim_sum / clim_n

    rows = []
    for lead_h, delta in LEADS.items():
        print(f"lead {lead_h}h  (delta={delta}) ...", flush=True)
        rand_pieces = {
            pid: load_rain(TEST_PIECES[(i + 2) % len(TEST_PIECES)])
            for i, pid in enumerate(TEST_PIECES)
        }
        accum = {"persistence": ([], []), "climatology": ([], []), "random": ([], [])}
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
                accum[label][0].append(truth.ravel())
                accum[label][1].append(pred.ravel())

        for label, (ts, ps) in accum.items():
            t = np.concatenate(ts)
            p = np.concatenate(ps)
            mse = float(np.mean((t - p) ** 2, dtype=np.float64))
            rows.append(dict(predictor_set=label, lead_h=lead_h,
                              MSE=mse, RMSE=np.sqrt(mse), R2_all=r2(t, p)))

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", default="docs/bulk_metrics_baselines.csv")
    args = p.parse_args()

    out_csv = REPO / args.out_csv
    if out_csv.exists():
        print(f"loading cached {out_csv}", flush=True)
        base_df = pd.read_csv(out_csv)
    else:
        base_df = compute_baseline_bulk()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}", flush=True)

    ls_df = pd.read_csv(REPO / "docs/bulk_metrics_v1.csv")
    ls_df["RMSE"] = np.sqrt(ls_df["MSE"])

    series = {
        "LatentSky":   (ls_df["lead_h"].to_numpy(), ls_df),
        "persistence": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "persistence"]),
        "climatology": (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "climatology"]),
        "random":      (sorted(LEADS.keys()), base_df[base_df["predictor_set"] == "random"]),
    }
    style = {
        "LatentSky":   dict(color="#1f77b4", marker="o", ls="-"),
        "persistence": dict(color="#ff7f0e", marker="^", ls="--"),
        "climatology": dict(color="0.2",     marker="s", ls="-."),
        "random":      dict(color="0.6",     marker="x", ls=":"),
    }

    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    print("\n=== R^2 (all pixels, full rain field) vs lead ===")
    for name, (_, df) in series.items():
        df = df.sort_values("lead_h")
        leads_sorted, vals = df["lead_h"].to_numpy(), df["R2_all"].to_numpy()
        ax_r2.plot(leads_sorted, vals, lw=1.8, ms=7, label=name, **style[name])
        print(f"{name:>12}: " + "  ".join(f"{L:g}h={v:.3f}" for L, v in zip(leads_sorted, vals)))

    print("\n=== RMSE (mm, full rain field) vs lead ===")
    for name, (_, df) in series.items():
        df = df.sort_values("lead_h")
        leads_sorted, vals = df["lead_h"].to_numpy(), df["RMSE"].to_numpy()
        ax_rmse.plot(leads_sorted, vals, lw=1.8, ms=7, label=name, **style[name])
        print(f"{name:>12}: " + "  ".join(f"{L:g}h={v:.3g}" for L, v in zip(leads_sorted, vals)))

    ax_r2.axhline(0, color="0.4", lw=1.0, ls=":")
    ax_r2.set_xlabel("forecast lead  [h]", fontsize=11)
    ax_r2.set_ylabel("R²  (all pixels, mm)", fontsize=11)
    ax_r2.set_title("(a) R² vs lead — plain pixel-wise", fontsize=11, loc="left", fontweight="bold")
    ax_r2.legend(fontsize=9, frameon=False)
    ax_r2.grid(alpha=0.3)

    ax_rmse.set_xlabel("forecast lead  [h]", fontsize=11)
    ax_rmse.set_ylabel("RMSE  [mm]", fontsize=11)
    ax_rmse.set_title("(b) RMSE vs lead — plain pixel-wise", fontsize=11, loc="left", fontweight="bold")
    ax_rmse.legend(fontsize=9, frameon=False)
    ax_rmse.grid(alpha=0.3)

    fig.suptitle("LatentSky vs persistence / climatology / random null — plain prediction vs truth",
                 fontsize=10)

    out = FIGURES_DIR / "bulk_r2_rmse_vs_lead_baselines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
