"""Paper Fig 5 — the predictability horizon and its scaling (full record).

(a) POD(L) at +3 h, three models      (b) FAR(L) at +3 h — the price
(c) POD(t; L) decay per coarse band   (d) tau(L) with the power-law fits

Caption sentence the figure must make obvious: the detection horizon grows as
a power law of system size; the flow sits above both the deterministic model
and the persistence null at every scale, at a false-alarm cost the same axes
price. CPU ~2 min (reads the 18 alltest tables once).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paperfig import COLOR, MARKER, W2, panel_label

from latentsky.metrics import default_scale_edges, fit_powerlaw, summarize_by_scale

REPO = Path("/glade/u/home/sabramian/LatentSky")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
ASSETS = REPO / "docs" / "assets_model_reference"
OUT = REPO / "paper" / "figures"

LEADS = [0.25, 0.5, 1, 2, 3, 6]
BANDS = [(25, 50), (50, 100), (100, 200), (200, 400), (400, 800)]
BAND_CMAP = plt.get_cmap("viridis")


def table_name(model, lead):
    tag = {"two-head": "", "persistence": "persistence_", "flow": "flow_"}[model]
    return EVAL / f"systems_alltest_{tag}lead{lead}h.csv.gz"


def pod_decay():
    """POD per coarse band per lead, flow + two-head."""
    out = {}
    for model in ("flow", "two-head"):
        pods = np.full((len(BANDS), len(LEADS)), np.nan)
        for j, lead in enumerate(LEADS):
            f = table_name(model, lead)
            if not f.exists():
                continue
            df = pd.read_csv(f, usecols=["role", "scale"])
            g = df[df.role.isin(["match", "miss"])]
            for i, (lo, hi) in enumerate(BANDS):
                b = g[(g.scale >= lo) & (g.scale < hi)]
                if len(b):
                    pods[i, j] = (b.role == "match").mean()
        out[model] = pods
    return out


def main():
    fig, axs = plt.subplots(2, 2, figsize=(W2, W2 * 0.98))
    ax_pod, ax_far, ax_dec, ax_tau = axs.ravel()

    # --- (a)/(b) POD and FAR by scale at +3 h -------------------------------
    edges = default_scale_edges(25, 600, 10)
    for model in ("two-head", "flow", "persistence"):
        df = pd.read_csv(table_name(model, 3))
        summ = summarize_by_scale(df, edges).by_scale
        kw = dict(color=COLOR[model], marker=MARKER[model], ms=3.5, lw=1.4)
        ax_pod.plot(summ["scale_mid"], summ["POD"], ls="-", label=model, **kw)
        ax_far.plot(summ["scale_mid"], summ["FAR"], ls="--", **kw)
    for ax, ylab in [(ax_pod, "Probability of Detection (POD)"), (ax_far, "False-Alarm Ratio (FAR)")]:
        ax.set_xscale("log")
        ax.set_xlim(25, 620)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("system scale L = √A (km)")
        ax.set_ylabel(ylab)
    ax_pod.legend(loc="center left", fontsize=8)
    panel_label(ax_pod, "A.")
    panel_label(ax_far, "B.")

    # --- (c) POD decay per band --------------------------------------------
    decay = pod_decay()
    for i, (lo, hi) in enumerate(BANDS):
        c = BAND_CMAP(0.88 * i / (len(BANDS) - 1))
        ax_dec.plot(LEADS, decay["flow"][i], "-", color=c, marker="D", ms=3,
                    lw=1.5)
        ax_dec.plot(LEADS, decay["two-head"][i], ":", color=c, lw=1.2)
    ax_dec.set_xlim(0, 6.2)
    ax_dec.set_ylim(0, 1.02)
    ax_dec.set_xlabel("lead time (h)")
    ax_dec.set_ylabel("Probability of Detection (POD)")
    ax_dec.plot([], [], "-", color="0.3", lw=1.5, label="flow")
    ax_dec.plot([], [], ":", color="0.3", lw=1.2, label="two-head")
    ax_dec.legend(loc="center", bbox_to_anchor=(0.72, 0.55), fontsize=7.5)
    panel_label(ax_dec, "C.")

    # --- (d) tau(L) scaling -------------------------------------------------
    df = pd.read_csv(ASSETS / "tau_fullrecord.csv")
    mids = np.sort(df.L_mid.unique())
    sc = None
    for model in ("two-head", "persistence", "flow"):
        d = df[df.model == model].sort_values("L_mid")
        t, pod0 = d.tau.to_numpy(), d["pod_0.25h"].to_numpy()
        ok = np.isfinite(t) & (t > 0) & (t < 100)
        out = fit_powerlaw(mids[ok], t[ok])
        a = out["beta"]
        c0 = out["tau_ref"] / (out["L_ref"] ** a)
        Lg = np.geomspace(28, 700, 40)
        ax_tau.plot(Lg, c0 * Lg ** a, "-", color=COLOR[model], lw=1.4,
                    label=f"{model}: α = {a:.2f}")
        sc = ax_tau.scatter(mids[ok], t[ok], c=pod0[ok], cmap="plasma",
                            vmin=0, vmax=1, marker=MARKER[model], s=26,
                            edgecolors=COLOR[model], linewidths=1.0, zorder=4)
    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_xlim(25, 800)
    ax_tau.set_ylim(0.3, 110)
    ax_tau.set_xlabel("system scale L (km)")
    ax_tau.set_ylabel("τ(L) (h)")
    ax_tau.legend(loc="upper left", fontsize=7.5)
    panel_label(ax_tau, "D.")
    # colorbar below panel D, without shrinking the panel itself: give it an
    # explicit axis so D stays the same size as the POD panel
    fig.subplots_adjust(left=0.075, right=0.985, top=0.99,
                        bottom=0.055, hspace=0.26, wspace=0.22)
    pos = ax_tau.get_position()
    cax = fig.add_axes([pos.x0 + 0.08 * pos.width, pos.y0 - 0.105,
                        0.84 * pos.width, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("POD at 0.25 h lead (POD₀)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)

    # discrete band colorbar below panel C, symmetric with D's
    from matplotlib import cm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    bounds = [25, 50, 100, 200, 400, 800]
    band_colors = [BAND_CMAP(0.88 * i / (len(BANDS) - 1))
                   for i in range(len(BANDS))]
    dmap = ListedColormap(band_colors)
    dnorm = BoundaryNorm(range(len(BANDS) + 1), len(BANDS))
    posc = ax_dec.get_position()
    caxc = fig.add_axes([posc.x0 + 0.08 * posc.width, posc.y0 - 0.105,
                         0.84 * posc.width, 0.016])
    cbc = fig.colorbar(cm.ScalarMappable(norm=dnorm, cmap=dmap), cax=caxc,
                       orientation="horizontal", spacing="uniform")
    cbc.set_ticks(range(len(BANDS) + 1))
    cbc.set_ticklabels([str(b) for b in bounds])
    cbc.set_label("scale band L (km)", fontsize=8)
    cbc.ax.tick_params(labelsize=7, length=2, width=0.6)
    cbc.outline.set_linewidth(0.5)
    fig.savefig(OUT / "fig5_horizon.pdf")
    print("saved fig5_horizon.pdf")


if __name__ == "__main__":
    main()
