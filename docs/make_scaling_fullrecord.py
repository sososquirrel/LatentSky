"""Registration-grade scaling law: POD(t; L) -> tau(L) -> alpha from the
FULL held-out test record (132 blocks x 40 days) at all six leads, for
two-head / flow / persistence. Committed fit conventions (fit_exp_decay
nonlinear B=0, fit_powerlaw). Writes tau_fullrecord.png + a csv of the
band PODs and taus. CPU pandas, ~4 min (18 tables)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latentsky.metrics import fit_exp_decay, fit_powerlaw

REPO = Path("/glade/u/home/sabramian/LatentSky")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = REPO / "docs" / "assets_model_reference"
LEADS = np.array([0.25, 0.5, 1, 2, 3, 6])
FINE_EDGES = np.geomspace(25, 800, 11)
STYLES = {"two-head": ("o", "#2166ac"), "persistence": ("s", "0.45"),
          "flow": ("D", "#b05e0a")}
FILES = {"two-head": "systems_alltest_lead{L:g}h.csv.gz",
         "flow": "systems_alltest_flow_lead{L:g}h.csv.gz",
         "persistence": "systems_alltest_persistence_lead{L:g}h.csv.gz"}

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def pod_by_band(df, edges):
    b = np.clip(np.digitize(df["scale"], edges) - 1, 0, len(edges) - 2)
    pods = np.full(len(edges) - 1, np.nan)
    ns = np.zeros(len(edges) - 1, int)
    for k in range(len(edges) - 1):
        g = df[(b == k) & df["role"].isin(["match", "miss"])]
        if len(g):
            pods[k] = (g["role"] == "match").mean()
            ns[k] = len(g)
    return pods, ns


def main():
    midsf = np.sqrt(FINE_EDGES[:-1] * FINE_EDGES[1:])
    pod = {m: np.zeros((len(midsf), len(LEADS))) for m in STYLES}
    nsys = {m: np.zeros((len(midsf), len(LEADS)), int) for m in STYLES}
    for j, L in enumerate(LEADS):
        for m in STYLES:
            df = pd.read_csv(EVAL / FILES[m].format(L=L),
                             usecols=["role", "scale"])
            pod[m][:, j], nsys[m][:, j] = pod_by_band(df, FINE_EDGES)
        print(f"lead {L:g}h loaded", flush=True)

    def efold(y):
        ok = y > 0
        if ok.sum() < 3 or y[0] <= 0:
            return np.nan
        return fit_exp_decay(LEADS[ok] - LEADS[0], y[ok] / y[0])["tau"]

    rows = []
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    alphas = {}
    for m, (mk, colr) in STYLES.items():
        tau = np.array([efold(pod[m][k]) for k in range(len(midsf))])
        ok = np.isfinite(tau) & (tau > 0) & (tau < 100)
        out = fit_powerlaw(midsf[ok], tau[ok])
        a = out["beta"]
        c0 = out["tau_ref"] / (out["L_ref"] ** a)
        alphas[m] = a
        ax.plot(midsf[ok], tau[ok], mk, color=colr, ms=5.5, ls="none",
                label=f"{m}: α = {a:.2f}")
        Lg = np.geomspace(28, 700, 40)
        ax.plot(Lg, c0 * Lg ** a, "-", color=colr, lw=1.1, alpha=0.8)
        for k in range(len(midsf)):
            rows.append(dict(model=m, L_mid=midsf[k], tau=tau[k],
                             n_min=int(nsys[m][k].min()),
                             **{f"pod_{L:g}h": pod[m][k, j]
                                for j, L in enumerate(LEADS)}))
        print(m, "tau:", np.round(tau, 2), "alpha:", round(a, 3))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(25, 800); ax.set_ylim(0.3, 110)
    ax.set_xlabel("system scale L (km)")
    ax.set_ylabel("τ(L) — POD e-folding time (h)")
    ax.set_title("the predictability horizon at full record\n"
                 "(132 test blocks × 40 days, six leads, per-draw)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(OUT / "tau_fullrecord.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "tau_fullrecord.csv", index=False)
    print("ALPHAS (full record):", {k: round(v, 3) for k, v in alphas.items()})
    print("saved tau_fullrecord.png + .csv")


if __name__ == "__main__":
    main()
