"""Paper Fig 5 candidate (WB2) — the size distribution of rain systems at +12 h.

ERA5 truth (black line, light fill) and, on top, GraphCast, FuXi, the GenCast ensemble
mean and one GenCast member, pooled over August 2020 (62 initializations, 00
and 12 UTC). Systems are segmented as everywhere in the paper: 8-connected,
above 4.8 mm per 12 h (the rate of 0.1 mm per 15 min). (a) number of systems
per field and per unit ln L; (b) mean intensity of the systems of
each size (rho = P/A, mm per 12 h).

Two stages: `compute()` segments the cached month fields
($SCRATCH/wb2_eval/wb2m12_*.nc, built by wb2_benchmark/wb2_decomp_12h_month.py)
and writes data/paper_figs/fig5_sizedist.npz (CPU, a few minutes); `draw()`
only reads that cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "paper_figures"))
from fig3_decomp import PAPER_FONT  # noqa: E402
from fig4_horizon import log_ticks, title  # noqa: E402
from wb2_common import COLW, THR  # noqa: E402
from paperfig import W2  # noqa: E402

CACHE = Path("/glade/derecho/scratch/sabramian/wb2_eval")
DATA = REPO / "data" / "paper_figs" / "fig5_sizedist.npz"
OUT = REPO / "paper_figures" / "extra"
EDGES = np.geomspace(20, 3000, 17)
MIDS = np.sqrt(EDGES[:-1] * EDGES[1:])
FIELDS = {"truth": "truth", "graphcast": "graphcast", "fuxi": "fuxi",
          "gencast mean": "gencast_mean", "gencast member": "gencast_member0"}
NAMES = {"graphcast": "GraphCast", "fuxi": "FuXi", "gencast mean": "GenCast mean",
         "gencast member": "GenCast member"}


def compute():
    import xarray as xr

    from latentsky.comparison.wb2 import to_mm
    from latentsky.metrics import cell_area_km2
    from latentsky.metrics.scale import label_systems

    area_row = cell_area_km2(321, 1440, lat_extent=(-40.125, 40.125))
    dlnL = np.diff(np.log(EDGES))
    out = {"edges": EDGES}
    for key, fname in FIELDS.items():
        f = xr.open_dataset(CACHE / f"wb2m12_{fname}.nc")["field"].values
        L, P, RHO, nfield, ptot = [], [], [], 0, 0.0
        for a in f:
            if np.isnan(a).all():
                continue
            a = to_mm(a)
            _, t = label_systems(np.nan_to_num(a), THR, area_row)
            L.append(t.L.to_numpy())
            P.append(t.P.to_numpy())
            RHO.append(t.rho.to_numpy())
            ptot += float(t.P.sum())
            nfield += 1
        L, P = np.concatenate(L), np.concatenate(P)
        n, _ = np.histogram(L, EDGES)
        p, _ = np.histogram(L, EDGES, weights=P)
        out[f"{key}/number"] = n / nfield / dlnL          # systems per field per unit ln L
        out[f"{key}/rain"] = p / ptot / dlnL              # share of the rain per unit ln L
        out[f"{key}/count"] = n
        r, _ = np.histogram(L, EDGES, weights=np.concatenate(RHO))
        out[f"{key}/rho"] = r / np.maximum(n, 1)        # mean intensity per size bin (mm / 12 h)
        print(f"{key:15s} {nfield} fields, {len(L)} systems, "
              f"{len(L) / nfield:.0f} per field")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, **out)


def plot_sizedist(ax_n, ax_p, letters=("(a)", "(b)"),
                  titles=("Number of systems, +12 h", "Mean intensity of the systems, +12 h"),
                  legend=True):
    """The two distribution panels, on given axes: (a) number of systems per
    field and unit ln L, (b) mean intensity rho of the systems of each size."""
    d = np.load(DATA)
    for ax, kind, base in ((ax_n, "number", 0.5), (ax_p, "rho", 0.0)):
        y = d[f"truth/{kind}"].astype(float)
        y[d["truth/count"] < 5] = np.nan
        # the truth: a light transparent fill under a black line
        ax.fill_between(MIDS, y, base, color="#5a6b7d", alpha=0.18, lw=0, zorder=0)
        ax.plot(MIDS, y, color="black", lw=2.0, zorder=1, label="ERA5 (truth)")
        for key, name in NAMES.items():
            y = d[f"{key}/{kind}"].astype(float)
            y[d[f"{key}/count"] < 5] = np.nan
            ax.plot(MIDS, y, color=COLW[key], marker="o", ms=3, lw=1.4, label=name)
        ax.set_xscale("log")
        ax.set_xlim(20, 3000)
        log_ticks(ax.xaxis, [30, 100, 300, 1000, 3000])
        ax.set_xlabel(r"system size $L = \sqrt{A}$ (km)")
        ax.grid(True, color="0.88", lw=0.4)
        ax.set_axisbelow(True)
    ax_n.set_yscale("log")
    ax_n.set_ylim(0.6, 1.2e3)
    ax_n.set_ylabel(r"systems per field, $\mathrm{d}N/\mathrm{d}\ln L$")
    ax_p.set_ylim(4, 16)
    ax_p.set_ylabel(r"mean intensity $\rho$ (mm per 12 h)")
    # the detection threshold: no system can be weaker than this
    ax_p.axhline(THR, color="0.4", lw=0.8, ls=(0, (4, 3)), zorder=0.5)
    ax_p.text(2600, THR + 0.2, "threshold", ha="right", va="bottom", fontsize=7, color="0.35")
    if legend:
        ax_n.legend(fontsize=7.5, loc="lower left", frameon=False)
    if titles:
        title(ax_n, letters[0], titles[0])
        title(ax_p, letters[1], titles[1])


def draw():
    plt.rcParams.update(PAPER_FONT)
    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(W2, W2 * 0.42))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.17, wspace=0.25)
    plot_sizedist(ax_n, ax_p)
    fig.savefig(OUT / "fig5_sizedist.pdf")
    print("saved fig5_sizedist.pdf")


if __name__ == "__main__":
    if "--compute" in sys.argv or not DATA.exists():
        compute()
    draw()
