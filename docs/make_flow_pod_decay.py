"""pod_decay_flow.png: POD(t; L) for the FLOW at full record — five coarse
scale bands across the six leads, with the committed exponential fits
(dashed) and the two-head as thin dotted reference. Same bands, colors and
style as section 7's left panel so the two figures compare directly.
CPU pandas over the 12 alltest tables, ~3 min."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from latentsky.metrics import fit_exp_decay

REPO = Path("/glade/u/home/sabramian/LatentSky")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = REPO / "docs" / "assets_model_reference"
LEADS = np.array([0.25, 0.5, 1, 2, 3, 6])
BAND_EDGES = np.geomspace(25, 800, 6)

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def pod_by_band(df):
    b = np.clip(np.digitize(df["scale"], BAND_EDGES) - 1, 0, len(BAND_EDGES) - 2)
    pods = np.full(len(BAND_EDGES) - 1, np.nan)
    for k in range(len(BAND_EDGES) - 1):
        g = df[(b == k) & df["role"].isin(["match", "miss"])]
        if len(g):
            pods[k] = (g["role"] == "match").mean()
    return pods


def main():
    pod = {m: np.zeros((len(BAND_EDGES) - 1, len(LEADS)))
           for m in ("flow", "two-head")}
    for j, L in enumerate(LEADS):
        for m, fn in [("flow", f"systems_alltest_flow_lead{L:g}h.csv.gz"),
                      ("two-head", f"systems_alltest_lead{L:g}h.csv.gz")]:
            df = pd.read_csv(EVAL / fn, usecols=["role", "scale"])
            pod[m][:, j] = pod_by_band(df)
        print(f"lead {L:g}h done", flush=True)

    mids = np.sqrt(BAND_EDGES[:-1] * BAND_EDGES[1:])
    vir = plt.get_cmap("viridis")
    cn = LogNorm(vmin=mids[0], vmax=mids[-1])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for k, Lmid in enumerate(mids):
        col = vir(cn(Lmid))
        y = pod["flow"][k]
        ax.plot(LEADS, y, "D-", color=col, ms=4,
                label=f"{BAND_EDGES[k]:.0f}–{BAND_EDGES[k+1]:.0f} km")
        ok = y > 0
        if ok.sum() >= 3 and y[0] > 0:
            tau = fit_exp_decay(LEADS[ok] - LEADS[0], y[ok] / y[0])["tau"]
            if 0 < tau < 100:
                tt = np.linspace(LEADS[0], 6, 50)
                ax.plot(tt, y[0] * np.exp(-(tt - LEADS[0]) / tau), ls="--",
                        color=col, lw=0.9, alpha=0.7)
        ax.plot(LEADS, pod["two-head"][k], ls=":", color=col, lw=1.0, alpha=0.65)
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
    ax.set_axisbelow(True)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("POD (object-based)")
    ax.set_title("the flow's detection decay, POD(t; L) — full record\n"
                 "(solid: flow; dotted: two-head; dashed: exponential fits)",
                 fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, title="scale band",
              title_fontsize=6.5, loc="upper right")
    fig.savefig(OUT / "pod_decay_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved pod_decay_flow.png")


if __name__ == "__main__":
    main()
