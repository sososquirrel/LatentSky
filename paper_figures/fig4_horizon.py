"""Paper Fig 5 — the predictability horizon and its scaling (full record).

(a) POD(L) at +3 h, three models      (b) FAR(L) at +3 h — the price
(c) POD(t; L) decay per coarse band   (d) tau(L), power law fitted on the
                                          59-336 km bands for all three forecasts

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
from paperfig import COLOR, MARKER, W2

from latentsky.metrics import default_scale_edges, fit_powerlaw, summarize_by_scale

REPO = Path("/glade/u/home/sabramian/LatentSky")
EVAL = Path("/glade/derecho/scratch/sabramian/latentsky/eval_method")
ASSETS = REPO / "docs" / "assets_model_reference"
OUT = REPO / "paper" / "figures"

LEADS = [0.25, 0.5, 1, 2, 3, 6]
BANDS = [(25, 50), (50, 100), (100, 200), (200, 400), (400, 800)]
BAND_CMAP = plt.get_cmap("viridis")
LABEL = {"two-head": "U-Net", "flow": "flow", "persistence": "persistence"}
FIT_WINDOW = (50, 400)   # km; band mids 59 ... 336


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


def log_ticks(axis, ticks):
    """Numbered ticks, and hence grid lines, at the given values of a log axis."""
    axis.set_major_locator(plt.FixedLocator(ticks))
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(plt.NullLocator())


def title(ax, letter, text):
    """Panel letter and a short title above the axes, as in Figs. 2 and 5."""
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontfamily="cmr10")
    ax.text(0.10, 1.02, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5)


def main():
    # the paper's text font
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})
    fig, axs = plt.subplots(2, 2, figsize=(W2, W2 * 1.0))
    ax_pod, ax_far, ax_dec, ax_tau = axs.ravel()
    for ax in axs.ravel():
        ax.grid(True, color="0.88", lw=0.4)
        ax.set_axisbelow(True)

    # --- (a)/(b) POD and FAR by scale at +3 h -------------------------------
    edges = default_scale_edges(25, 600, 10)
    for model in ("two-head", "flow", "persistence"):
        df = pd.read_csv(table_name(model, 3))
        summ = summarize_by_scale(df, edges).by_scale
        kw = dict(color=COLOR[model], marker=MARKER[model], ms=3.5, lw=1.4)
        ax_pod.plot(summ["scale_mid"], summ["POD"], ls="-", label=LABEL[model], **kw)
        ax_far.plot(summ["scale_mid"], summ["FAR"], ls="--", **kw)
    for ax, ylab in [(ax_pod, "Probability of Detection (POD)"), (ax_far, "False-Alarm Ratio (FAR)")]:
        ax.set_xscale("log")
        ax.set_xlim(25, 620)
        log_ticks(ax.xaxis, [30, 50, 100, 200, 400])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"system size $L = \sqrt{A}$ (km)")
        ax.set_ylabel(ylab)
    ax_pod.legend(loc="center left", fontsize=8)
    title(ax_pod, "(a)", "Detection by system size, +3 h")
    title(ax_far, "(b)", "False alarms by system size, +3 h")

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
    ax_dec.plot([], [], ":", color="0.3", lw=1.2, label="U-Net")
    ax_dec.legend(loc="center", bbox_to_anchor=(0.72, 0.55), fontsize=7.5)
    title(ax_dec, "(c)", "Detection against lead time, per size band")

    # --- (d) tau(L) scaling -------------------------------------------------
    df = pd.read_csv(ASSETS / "tau_fullrecord.csv")
    mids = np.sort(df.L_mid.unique())
    # one fit window for the three forecasts: the bands a 6 h record constrains
    # for persistence, and the same bands for the two models
    ax_tau.axvspan(*FIT_WINDOW, color="0.93", lw=0, zorder=0)
    ax_tau.axhline(6, color="0.55", lw=0.7, ls=":", zorder=1)
    ax_tau.text(27, 6.5, "longest lead, 6 h", fontsize=6.5, color="0.4")
    inw = (mids > FIT_WINDOW[0]) & (mids < FIT_WINDOW[1])
    for model in ("persistence", "two-head", "flow"):
        d = df[df.model == model].sort_values("L_mid")
        t = d.tau.to_numpy()
        out = fit_powerlaw(mids[inw], t[inw])
        a = out["beta"]
        c0 = out["tau_ref"] / (out["L_ref"] ** a)
        Lg = np.array(FIT_WINDOW, dtype=float)
        ax_tau.plot(Lg, c0 * Lg ** a, "-", color=COLOR[model], lw=1.4,
                    label=rf"{LABEL[model]}: $\alpha$ = {a:.2f}")
        # bands in the fit: filled with the model's colour
        ax_tau.scatter(mids[inw], t[inw], marker=MARKER[model], s=34,
                       facecolors=COLOR[model], edgecolors="white",
                       linewidths=0.6, zorder=4)
        # bands left out of the fit: hollow
        left = ~inw & np.isfinite(t) & (t > 0) & (t < 110)
        ax_tau.scatter(mids[left], t[left], marker=MARKER[model], s=26,
                       facecolors="white", edgecolors=COLOR[model],
                       linewidths=1.0, zorder=3)
    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    log_ticks(ax_tau.xaxis, [30, 50, 100, 200, 400, 800])
    log_ticks(ax_tau.yaxis, [0.5, 1, 2, 5, 10, 20, 50, 100])
    ax_tau.set_xlim(25, 800)
    ax_tau.set_ylim(0.3, 110)
    ax_tau.set_xlabel(r"system size $L = \sqrt{A}$ (km)")
    ax_tau.set_ylabel(r"detection horizon $\tau(L)$ (h)")
    ax_tau.legend(loc="upper left", fontsize=7.5)
    title(ax_tau, "(d)", "Horizon against system size")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965,
                        bottom=0.055, hspace=0.32, wspace=0.22)
    # below panel D: what filled and hollow markers mean, level with C's colorbar
    pos = ax_tau.get_position()
    from matplotlib.lines import Line2D
    hk = [Line2D([], [], ls="", marker="o", ms=5, mfc="0.35", mec="white", label="in the fit"),
          Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec="0.35", label="not fitted")]
    fig.legend(handles=hk, loc="center", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(pos.x0 + pos.width / 2, pos.y0 - 0.097))
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
    fig.savefig(OUT / "fig4_horizon.pdf")
    print("saved fig4_horizon.pdf")


if __name__ == "__main__":
    main()
