"""dlnA(t) -- the SIGNED log-area error (ln(A_pred/A_truth)) for matched
systems, plotted directly vs lead time, one curve per scale bin.

Deliberately shows the signed quantity, not the |dlnA| used everywhere else
in the tau(L) decomposition analysis (plot_dln_fit_curves.py,
fit_dln_tau_scaling.py) -- the absolute value there was needed to build a
bounded agreement index for the exponential-decay fit, but it throws away
the direction of the bias (over- vs under-estimating system area), which is
exactly what this plot is for.

Reads the raw matched-system table directly (not the pre-aggregated
summary CSV) so bin edges/count are independent of any other figure.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from latentsky.paths import FIGURES_DIR
from latentsky.scale_analysis import default_scale_edges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems_diffusion.csv.gz")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--tag", default="diffusion")
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--min-km", type=float, default=25.0)
    p.add_argument("--max-km", type=float, default=3000.0)
    p.add_argument("--quantity", choices=["dlnA", "dlnrho"], default="dlnA")
    p.add_argument("--signed", action="store_true",
                    help="plot the signed median (over/under-estimate visible) instead of "
                         "|median| -- default is absolute value so every scale's curve sits "
                         "on the same side of zero and magnitudes are directly comparable.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    edges = default_scale_edges(args.min_km, args.max_km, args.n_bins)
    scale_mid = np.sqrt(edges[:-1] * edges[1:])

    sysdf = pd.read_csv(repo / args.systems)
    sysdf = sysdf[sysdf["name"].isin(args.runs) & (sysdf["role"] == "match")].copy()
    if sysdf.empty:
        raise SystemExit(f"no matched systems for runs {args.runs!r} in {args.systems!r}")
    b = np.clip(np.digitize(sysdf["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    sysdf["scale_mid"] = scale_mid[b]

    g = (sysdf.groupby(["scale_mid", "lead_h"])[args.quantity]
         .agg(med="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
              n="count")
         .reset_index())

    if not args.signed:
        # |median|, not median(|.|) -- same convention as the info_X(t) =
        # exp(-|dlnX_med(t)|) index used everywhere else in this analysis.
        # The IQR band is re-centered on |med| with its original half-width
        # (a rough visual spread, not an exact quantile of |dlnX|), clipped
        # at 0 -- exact would require re-deriving quantiles of |dlnX| from
        # the raw per-system values, overkill for a comparison plot.
        half_width = (g["q75"] - g["q25"]).abs() / 2
        g["med"] = g["med"].abs()
        g["q25"] = (g["med"] - half_width).clip(lower=0)
        g["q75"] = g["med"] + half_width

    bins_present = sorted(g["scale_mid"].unique())
    cmap = plt.get_cmap("viridis")
    norm = LogNorm(vmin=min(bins_present), vmax=max(bins_present))

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for smid in bins_present:
        sub = g[g["scale_mid"] == smid].sort_values("lead_h")
        color = cmap(norm(smid))
        ax.fill_between(sub["lead_h"], sub["q25"], sub["q75"], color=color, alpha=0.12, zorder=1)
        ax.plot(sub["lead_h"], sub["med"], "o-", color=color, lw=1.8, ms=5, zorder=3)

    if args.signed:
        ax.axhline(0.0, color="k", lw=1.0, ls=":", zorder=2)
    ax.set_xlabel("lead (h)")
    base_label = "dlnA = ln(A_pred / A_truth)" if args.quantity == "dlnA" else "dlnrho = ln(rho_pred / rho_truth)"
    qlabel = base_label if args.signed else f"|{base_label}|"
    sign_note = "0 = perfect; negative = underestimate, positive = overestimate" if args.signed \
        else "0 = perfect; larger = more wrong, sign of the error not shown"
    ax.set_ylabel(f"{qlabel}\n(median, matched systems; shaded ~ IQR)")
    ax.set_title(f"{qlabel} vs lead time, by system scale -- {args.tag}\n{sign_note}")
    ax.grid(alpha=0.25)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("system scale  L = √A  (km)")

    out = (repo / args.out if args.out
           else FIGURES_DIR / f"{args.quantity}_vs_lead_by_scale_{args.tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
