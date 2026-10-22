"""R3 preview: the flow-matching draws through the same evaluation lens as the
two-head — per-lead systems tables (single draws, per-draw scoring), the
POD(t;L)/tau(L) scaling with two-head and persistence on the same five leads,
and the band-barycenter shift on the dlnA/dlnrho plane (pre-registered
predictions 1 and 2, journal/2026-09-03). Batched CPU; meant for a CPU PBS job.
Caches per lead; figures + numbers printed at the end."""
from __future__ import annotations

import sys
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
sys.path.insert(0, str(REPO / "docs"))
from make_scaling_assets import (  # noqa: E402
    BAND_EDGES, FINE_EDGES, INPUT_VARS, PIECE, BLOCK, P, THR, IOU_MIN,
    pod_by_band, systems_for_lead,
)

SCALE = 2.0
FLOW_LEADS = [0.25, 1, 2, 3, 6]
SIGMA = {0.25: 0.1018, 1: 0.1457, 2: 0.1713, 3: 0.1876, 6: 0.204}
NSTEPS, BATCH = 32, 8

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def flow_systems_for_lead(lead_h: float) -> pd.DataFrame:
    cache = OUT / f"scaling_systems_flow_lead{lead_h:g}h.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import torch
    torch.set_num_threads(16)
    from latentsky.models import VelocityUNet, UNet2DTwoHead
    from latentsky.models.flow import RectifiedFlow
    delta = int(round(lead_h * 4)) + 1
    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / f"unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead{lead_h:g}h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = VelocityUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / f"flow_models/flow_matching_lead{lead_h:g}h.checkpoint.pt",
        map_location="cpu", weights_only=True))
    net.eval()
    rf = RectifiedFlow(sigma=SIGMA[lead_h])

    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    area_row = cell_area_km2(320, 1440)[r * P:(r + 1) * P]
    ytab = np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")
    vtabs = {v: np.load(PIECES / f"{v}_piece{PIECE:02d}.npy", mmap_mode="r")
             for v in INPUT_VARS}
    frames = list(range(96 - delta))
    rows = []
    with torch.no_grad():
        for b0 in range(0, len(frames), BATCH):
            fb = frames[b0:b0 + BATCH]
            x = np.stack([[np.asarray(vtabs[v][t], dtype=np.float32)[sl]
                           for v in INPUT_VARS] for t in fb], 0)
            cond = torch.from_numpy(x)
            det, _ = twohead(cond)
            zf = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
            g = torch.Generator().manual_seed(9000 + b0)
            z = rf.sample(net, cond, zf, n_steps=NSTEPS, generator=g)
            mm = np.clip(np.expm1(SCALE * np.clip(z.numpy()[:, 0], 0, 2.5)), 0, None)
            for i, t in enumerate(fb):
                yt = np.asarray(ytab[t + delta], dtype=np.float32)[sl]
                dfr = match_systems(yt, mm[i], THR, area_row, iou_min=IOU_MIN)
                dfr.insert(0, "model", "flow")
                dfr.insert(0, "frame", t)
                rows.append(dfr)
            print(f"  lead {lead_h}h frames {fb[0]}-{fb[-1]} done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def main():
    leads = np.array(FLOW_LEADS)
    # POD(t;L) for flow, plus two-head & persistence restricted to the same leads
    pod = {m: np.zeros((len(BAND_EDGES) - 1, len(leads))) for m in
           ("two-head", "persistence", "flow")}
    podf = {m: np.zeros((len(FINE_EDGES) - 1, len(leads))) for m in
            ("two-head", "persistence", "flow")}
    m3h = {}
    for j, L in enumerate(leads):
        base = systems_for_lead(float(L))           # two-head + persistence (cached)
        fl = flow_systems_for_lead(float(L))        # flow (cached)
        for m, sub in [("two-head", base[base.model == "two-head"]),
                       ("persistence", base[base.model == "persistence"]),
                       ("flow", fl)]:
            pod[m][:, j] = pod_by_band(sub, BAND_EDGES)
            podf[m][:, j] = pod_by_band(sub, FINE_EDGES)
            if L == 3:
                m3h[m] = sub[sub.role == "match"].reset_index(drop=True)
        print(f"lead {L}h assembled", flush=True)

    midsf = np.sqrt(FINE_EDGES[:-1] * FINE_EDGES[1:])

    def efold(y):
        ok = y > 0
        if ok.sum() < 3 or y[0] <= 0:
            return np.nan
        return fit_exp_decay(leads[ok] - leads[0], y[ok] / y[0])["tau"]

    tauf = {m: np.array([efold(podf[m][k]) for k in range(len(midsf))])
            for m in pod}

    def powerlaw(Lv, t):
        ok = np.isfinite(t) & (t > 0) & (t < 100)
        out = fit_powerlaw(Lv[ok], t[ok])
        return out["beta"], out["tau_ref"] / (out["L_ref"] ** out["beta"]), ok

    # --- figure A: tau(L) with three series
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    styles = {"two-head": ("o", "#2166ac"), "persistence": ("s", "0.45"),
              "flow": ("D", "#b05e0a")}
    alphas = {}
    for m, (mk, colr) in styles.items():
        t = tauf[m]
        a, c0, ok = powerlaw(midsf, t)
        alphas[m] = a
        ax.plot(midsf[ok], t[ok], mk, color=colr, ms=5.5, ls="none",
                label=f"{m}: α = {a:.2f}")
        Lg = np.geomspace(midsf[ok].min(), midsf[ok].max(), 40)
        ax.plot(Lg, c0 * Lg ** a, "-", color=colr, lw=1.1, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("system scale L (km)")
    ax.set_ylabel("τ(L) — POD e-folding time (h)")
    ax.set_title("prediction 1: does the flow move the horizon?\n"
                 "(five leads, single flow draws, per-draw scoring)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(OUT / "flow_scaling_tau.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure B: barycenter shift at 3 h
    bands = [("small < 65 km", 0, 65, 40), ("65–200 km", 65, 200, 115),
             ("large > 200 km", 200, np.inf, 350)]
    from matplotlib.colors import LogNorm
    vir = plt.get_cmap("viridis")
    cn = LogNorm(vmin=30, vmax=600)
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    lim = 1.6
    ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
    xs = np.linspace(-lim, lim, 2)
    ax.plot(xs, -xs, ls=":", color="#b2182b", lw=1.1)
    report = []
    for name, lo, hi, lrep in bands:
        pts = {}
        for m, mdf in [("two-head", m3h["two-head"]), ("flow", m3h["flow"])]:
            g = mdf[(mdf.scale >= lo) & (mdf.scale < hi)]
            w = g["P_t"].to_numpy()
            pts[m] = (float((w * g.dlnA).sum() / w.sum()),
                      float((w * g.dlnrho).sum() / w.sum()), len(g))
        (x0, y0, n0), (x1, y1, n1) = pts["two-head"], pts["flow"]
        col = vir(cn(lrep))
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.plot(x0, y0, "o", color=col, ms=8, mec="0.2")
        ax.plot(x1, y1, "D", color=col, ms=8, mec="0.2")
        report.append(f"{name}: two-head ({x0:+.2f},{y0:+.2f}) n={n0} -> "
                      f"flow ({x1:+.2f},{y1:+.2f}) n={n1}")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="0.3", label="two-head barycenter"),
        Line2D([], [], marker="D", ls="none", color="0.3", label="flow barycenter"),
        Line2D([], [], color=vir(cn(40)), lw=2, label="small < 65 km"),
        Line2D([], [], color=vir(cn(115)), lw=2, label="65–200 km"),
        Line2D([], [], color=vir(cn(350)), lw=2, label="large > 200 km")],
        fontsize=7, frameon=False, loc="lower left")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("dlnA  (area error)")
    ax.set_ylabel(r"dln$\rho$  (intensity error)")
    ax.set_title("prediction 2: barycenters collapse toward the origin\n"
                 "(+3 h, per band, precip-weighted)", fontsize=9)
    fig.savefig(OUT / "flow_decomp_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure C: POD and FAR by scale at +3 h, three models
    from latentsky.metrics import default_scale_edges, summarize_by_scale
    edges = default_scale_edges(25, 600, 10)
    base3 = systems_for_lead(3.0)
    fl3 = flow_systems_for_lead(3.0)
    series = [("two-head", base3[base3.model == "two-head"], "#2166ac"),
              ("flow", fl3, "#b05e0a"),
              ("persistence", base3[base3.model == "persistence"], "0.45")]
    fig, axs2 = plt.subplots(1, 2, figsize=(9.4, 3.5), sharex=True)
    for name, sub, colr in series:
        summ = summarize_by_scale(sub, edges).by_scale
        axs2[0].plot(summ["scale_mid"], summ["POD"], "o-", color=colr, ms=4,
                     label=name)
        axs2[1].plot(summ["scale_mid"], summ["FAR"], "s--", color=colr, ms=4,
                     label=name)
    for ax2, ttl in [(axs2[0], "POD — probability of detection"),
                     (axs2[1], "FAR — false-alarm ratio")]:
        ax2.set_xscale("log"); ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("system scale L = √A (km)")
        ax2.set_title(ttl + " (+3 h)", fontsize=9)
    axs2[0].set_ylabel("object-based rate")
    axs2[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / "flow_podfar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("ALPHAS:", {k: round(v, 3) for k, v in alphas.items()})
    for r in report:
        print("BARY:", r)


if __name__ == "__main__":
    main()
