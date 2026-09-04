"""Draft scaling-law figure for docs/model_reference.html: POD(t; L) decay
across the six leads with exponential fits, and the resulting tau(L) with a
power-law fit — two-head vs the persistence null, one test block x one day
(draft grade). CPU login-safe (~3 min, cached per lead)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latentsky.metrics import cell_area_km2, fit_exp_decay, fit_powerlaw, match_systems

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"

INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, PIECE, BLOCK = 128, 8, (1, 4)
THR, IOU_MIN = 0.1, 0.1
LEADS = [0.25, 0.5, 1, 2, 3, 6]
BAND_EDGES = np.geomspace(25, 800, 6)   # 5 coarse bands (decay curves)
FINE_EDGES = np.geomspace(25, 800, 11)  # 10 fine bands (tau scatter)

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def systems_for_lead(lead_h: float) -> pd.DataFrame:
    cache = OUT / f"scaling_systems_lead{lead_h:g}h.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import torch
    torch.set_num_threads(4)
    from latentsky.models import UNet2DTwoHead
    delta = int(round(lead_h * 4)) + 1
    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / f"unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead{lead_h:g}h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    area_row = cell_area_km2(320, 1440)[r * P:(r + 1) * P]
    ytab = np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")
    vtabs = {v: np.load(PIECES / f"{v}_piece{PIECE:02d}.npy", mmap_mode="r")
             for v in INPUT_VARS}
    rows = []
    with torch.no_grad():
        for t in range(96 - delta):
            x = np.stack([np.asarray(vtabs[v][t], dtype=np.float32)[sl]
                          for v in INPUT_VARS], 0)
            pred, _ = twohead(torch.from_numpy(x[None]))
            pred = np.clip(pred.numpy()[0, 0], 0, None)
            yt = np.asarray(ytab[t + delta], dtype=np.float32)[sl]
            ypersist = np.asarray(ytab[t + 1], dtype=np.float32)[sl]  # aligned rain now
            for name, pfield in [("two-head", pred), ("persistence", ypersist)]:
                dfr = match_systems(yt, pfield, THR, area_row, iou_min=IOU_MIN)
                dfr.insert(0, "model", name)
                dfr.insert(0, "frame", t)
                rows.append(dfr)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def pod_by_band(df: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    pods = np.full(len(edges) - 1, np.nan)
    b = np.clip(np.digitize(df["scale"], edges) - 1, 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        g = df[(b == k) & df["role"].isin(["match", "miss"])]
        if len(g):
            pods[k] = (g["role"] == "match").mean()
    return pods


def main():
    leads = np.array(LEADS)
    pod = {m: np.zeros((len(BAND_EDGES) - 1, len(LEADS))) for m in ("two-head", "persistence")}
    podf = {m: np.zeros((len(FINE_EDGES) - 1, len(LEADS))) for m in ("two-head", "persistence")}
    for j, L in enumerate(LEADS):
        df = systems_for_lead(L)
        for m in ("two-head", "persistence"):
            sub = df[df["model"] == m]
            pod[m][:, j] = pod_by_band(sub, BAND_EDGES)
            podf[m][:, j] = pod_by_band(sub, FINE_EDGES)
        print(f"lead {L}h done", flush=True)

    mids = np.sqrt(BAND_EDGES[:-1] * BAND_EDGES[1:])
    midsf = np.sqrt(FINE_EDGES[:-1] * FINE_EDGES[1:])

    def efold(y):
        """Project fit convention (metrics.fits.fit_exp_decay): nonlinear
        least squares on POD(t)/POD(t0) = exp(-(t-t0)/tau), B=0 forced."""
        ok = y > 0
        if ok.sum() < 3 or y[0] <= 0:
            return np.nan
        out = fit_exp_decay(leads[ok] - leads[0], y[ok] / y[0])
        return out["tau"]

    tau = {m: np.array([efold(pod[m][k]) for k in range(len(mids))])
           for m in ("two-head", "persistence")}
    tauf = {m: np.array([efold(podf[m][k]) for k in range(len(midsf))])
            for m in ("two-head", "persistence")}

    def powerlaw(L, t):
        ok = np.isfinite(t) & (t > 0) & (t < 100)
        out = fit_powerlaw(L[ok], t[ok])
        a = out["beta"]
        c0 = out["tau_ref"] / (out["L_ref"] ** a)
        return a, c0, ok

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    vir = plt.get_cmap("viridis")
    from matplotlib.colors import LogNorm
    cn = LogNorm(vmin=mids[0], vmax=mids[-1])

    ax = axs[0]
    for k, Lmid in enumerate(mids):
        col = vir(cn(Lmid))
        y = pod["two-head"][k]
        ax.plot(leads, y, "o-", color=col, ms=4,
                label=f"{BAND_EDGES[k]:.0f}–{BAND_EDGES[k+1]:.0f} km")
        t = tau["two-head"][k]
        if np.isfinite(t) and t > 0 and t < 100:
            tt = np.linspace(leads[0], 6, 50)
            ax.plot(tt, y[0] * np.exp(-(tt - leads[0]) / t), ls="--", color=col,
                    lw=0.9, alpha=0.7)
        ax.plot(leads, pod["persistence"][k], ls=":", color=col, lw=1.0, alpha=0.65)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("POD (object-based)")
    ax.set_title("detection decays ~exponentially with lead\n(solid: two-head; dotted: persistence; dashed: fits)",
                 fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, title="scale band", title_fontsize=6.5,
              loc="lower left")

    ax = axs[1]
    for m, mk, colr in [("two-head", "o", "#2166ac"), ("persistence", "s", "0.4")]:
        t = tauf[m]
        a, c0, ok = powerlaw(midsf, t)
        pod0 = podf[m][:, 0]
        sc2 = ax.scatter(midsf[ok], t[ok], c=pod0[ok], cmap="plasma", vmin=0, vmax=1,
                         marker=mk, s=42, edgecolors=colr, linewidths=1.3,
                         label=f"{m}: α = {a:.2f}", zorder=4)
        Lg = np.geomspace(midsf[ok].min(), midsf[ok].max(), 40)
        ax.plot(Lg, c0 * Lg ** a, "-", color=colr, lw=1.1, alpha=0.8)
    cb2 = fig.colorbar(sc2, ax=ax, shrink=0.85, pad=0.02)
    cb2.set_label("POD at 0.25 h lead (POD₀)", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("system scale L (km)")
    ax.set_ylabel("τ(L) — POD e-folding time (h)")
    ax.set_title("the predictability horizon and its scaling\nT(L) = τ(L) ∝ L^α",
                 fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.savefig(OUT / "eval_method_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for m in ("two-head", "persistence"):
        a, c0, ok = powerlaw(midsf, tauf[m])
        print(m, "tau(fine):", np.round(tauf[m], 2), "alpha:", round(a, 3))


if __name__ == "__main__":
    main()
