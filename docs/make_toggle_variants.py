"""Paired 'without / with the flow correction' variants of the two headline
skill figures, with IDENTICAL fixed axes so the page's interactive toggle
reads as an animation of what the correction adds:

  tau_noflow.png / tau_withflow.png        — τ(L) scaling, ±flow series
  podfar_noflow.png / podfar_withflow.png  — full-record POD/FAR at +3 h, ±flow

Reads the cached one-day scaling tables (τ) and the full-record alltest tables
(POD/FAR). CPU, seconds."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latentsky.metrics import default_scale_edges, fit_exp_decay, fit_powerlaw, summarize_by_scale

REPO = Path("/glade/u/home/sabramian/LatentSky")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
OUT = REPO / "docs" / "assets_model_reference"
sys.path.insert(0, str(REPO / "docs"))
from make_scaling_assets import FINE_EDGES, pod_by_band  # noqa: E402

FLOW_LEADS = np.array([0.25, 1, 2, 3, 6])
STYLES = {"two-head": ("o", "#2166ac"), "persistence": ("s", "0.45"),
          "flow": ("D", "#b05e0a")}

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def tau_series():
    midsf = np.sqrt(FINE_EDGES[:-1] * FINE_EDGES[1:])
    podf = {m: np.zeros((len(FINE_EDGES) - 1, len(FLOW_LEADS)))
            for m in STYLES}
    for j, L in enumerate(FLOW_LEADS):
        base = pd.read_csv(OUT / f"scaling_systems_lead{L:g}h.csv")
        fl = pd.read_csv(OUT / f"scaling_systems_flow_lead{L:g}h.csv")
        for m, sub in [("two-head", base[base.model == "two-head"]),
                       ("persistence", base[base.model == "persistence"]),
                       ("flow", fl)]:
            podf[m][:, j] = pod_by_band(sub, FINE_EDGES)

    def efold(y):
        ok = y > 0
        if ok.sum() < 3 or y[0] <= 0:
            return np.nan
        return fit_exp_decay(FLOW_LEADS[ok] - FLOW_LEADS[0], y[ok] / y[0])["tau"]

    tauf = {m: np.array([efold(podf[m][k]) for k in range(len(midsf))])
            for m in STYLES}
    return midsf, tauf


def fig_tau(midsf, tauf, with_flow, fn):
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    models = ["two-head", "persistence"] + (["flow"] if with_flow else [])
    for m in models:
        mk, colr = STYLES[m]
        t = tauf[m]
        ok = np.isfinite(t) & (t > 0) & (t < 100)
        out = fit_powerlaw(midsf[ok], t[ok])
        a = out["beta"]
        c0 = out["tau_ref"] / (out["L_ref"] ** a)
        ax.plot(midsf[ok], t[ok], mk, color=colr, ms=5.5, ls="none",
                label=f"{m}: α = {a:.2f}")
        Lg = np.geomspace(28, 700, 40)
        ax.plot(Lg, c0 * Lg ** a, "-", color=colr, lw=1.1, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(25, 800); ax.set_ylim(0.3, 110)
    ax.set_xlabel("system scale L (km)")
    ax.set_ylabel("τ(L) — POD e-folding time (h)")
    ax.set_title("the predictability horizon τ(L) ∝ L^α\n"
                 + ("with the flow correction" if with_flow
                    else "deterministic model vs the null"), fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(OUT / fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", fn)


def fig_podfar(with_flow, fn):
    tabs = {
        "two-head": pd.read_csv(EVAL / "systems_alltest_lead3h.csv.gz"),
        "persistence": pd.read_csv(EVAL / "systems_alltest_persistence_lead3h.csv.gz"),
    }
    if with_flow:
        tabs["flow"] = pd.read_csv(EVAL / "systems_alltest_flow_lead3h.csv.gz")
    edges = default_scale_edges(25, 600, 10)
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.5), sharex=True)
    for name in ("two-head", "flow", "persistence"):
        if name not in tabs:
            continue
        _, colr = STYLES[name]
        summ = summarize_by_scale(tabs[name], edges).by_scale
        axs[0].plot(summ["scale_mid"], summ["POD"], "o-", color=colr, ms=4,
                    label=name)
        axs[1].plot(summ["scale_mid"], summ["FAR"], "s--", color=colr, ms=4,
                    label=name)
    for ax, ttl in [(axs[0], "POD — probability of detection"),
                    (axs[1], "FAR — false-alarm ratio")]:
        ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlim(25, 620)
        ax.set_xlabel("system scale L = √A (km)")
        ax.set_title(ttl + " (+3 h, full test set)", fontsize=9)
    axs[0].set_ylabel("object-based rate")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", fn)


if __name__ == "__main__":
    midsf, tauf = tau_series()
    fig_tau(midsf, tauf, False, "tau_noflow.png")
    fig_tau(midsf, tauf, True, "tau_withflow.png")
    fig_podfar(False, "podfar_noflow.png")
    fig_podfar(True, "podfar_withflow.png")
