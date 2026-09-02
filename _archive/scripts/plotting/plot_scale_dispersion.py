"""Dispersion ratio vs scale — does the model under/over-disperse, and where?

For each scale bin and lead, the ratio of predicted to truth spread
(std in log space) of A, rho and P:

    disp = std(ln x_pred) / std(ln x_truth)

  disp < 1  -> predicted distribution too narrow (under-dispersed)
  disp = 1  -> spread matches truth
  disp > 1  -> too wide (over-dispersed)

Spreads are unpaired: truth systems = role in {match, miss};
pred systems = role in {match, false_alarm}. One line per lead
(viridis: dark = nowcast, bright = long horizon).

Reveals whether the MSE-per-patch under-dispersion of intensity is
uniform or concentrated in the small systems.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

_MIN_N = 20  # need this many systems in a bin for a stable std


def _logstd(series: pd.Series) -> float:
    v = series.to_numpy()
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.std(np.log(v))) if v.size >= _MIN_N else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems_v2.csv.gz")
    p.add_argument("--summary", default="docs/scale_summary_v2.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--leads", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    sp = repo / args.systems
    sysdf = (pd.read_parquet(sp) if str(sp).endswith(".parquet")
             else pd.read_csv(sp))
    summ = pd.read_csv(repo / args.summary)

    ps = args.predictor_set
    sysdf = sysdf[sysdf["predictor_set"] == ps]
    summ = summ[summ["predictor_set"] == ps]
    if sysdf.empty or summ.empty:
        raise SystemExit(f"no rows for predictor set {ps!r}")

    # scale-bin edges from the summary (consistent with every other figure)
    g0 = summ.drop_duplicates("scale_mid").sort_values("scale_mid")
    edges = np.r_[g0["scale_lo"].to_numpy(), g0["scale_hi"].to_numpy()[-1]]
    mids = g0["scale_mid"].to_numpy()

    avail = set(summ["lead_h"].unique())
    leads = sorted(ld for ld in args.leads
                   if np.any(np.isclose(list(avail), ld)))
    skipped = sorted(set(args.leads) - set(leads))
    if skipped:
        print(f"note: no trained run at lead(s) {skipped} h — skipped")
    norm = Normalize(vmin=min(leads), vmax=max(leads))
    cmap = cm.viridis

    fields = [("A", "(a) area A dispersion"),
              ("rho", "(b) intensity ρ dispersion"),
              ("P", "(c) total P dispersion")]

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    for axis, (f, title) in zip(ax, fields):
        for ld in leads:
            s = sysdf[sysdf["lead_h"] == ld].copy()
            s["_b"] = np.clip(np.digitize(s["scale"], edges) - 1,
                              0, len(edges) - 2)
            ratio = []
            for k in range(len(edges) - 1):
                gk = s[s["_b"] == k]
                t = _logstd(gk.loc[gk[f"{f}_t"].notna(), f"{f}_t"])
                pr = _logstd(gk.loc[gk[f"{f}_p"].notna(), f"{f}_p"])
                ratio.append(pr / t if (t and np.isfinite(t)
                                        and np.isfinite(pr)) else np.nan)
            ratio = np.array(ratio)
            m = np.isfinite(ratio)
            axis.plot(mids[m], ratio[m], "o-", ms=4,
                      color=cmap(norm(ld)), label=f"{ld:g} h")
        axis.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
        axis.set_xscale("log")
        axis.set_xlabel("system scale L = √A  (km)")
        axis.set_ylabel("pred spread / truth spread")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("lead time (h)")
    fig.suptitle(
        f"Distribution spread vs scale — {ps}   "
        f"(<1 = under-dispersed / too smooth,  >1 = over-dispersed)",
        fontsize=13)

    safe = ps.split()[0]
    out = (repo / (args.out or f"docs/figures/scale_dispersion_{safe}.png")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
