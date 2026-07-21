"""Same-axes comparison of the scale-aware decomposition across leads.

Default: 0 h vs 3 h for one predictor set, every panel on shared axes so the
horizon-induced degradation is read directly. Generalises to any lead list
via --leads.

  (a) system-size distribution   truth (black) vs pred at each lead
  (b) precip by scale            truth (black) vs pred at each lead
  (c) POD vs scale               one line per lead
  (d) FAR vs scale               one line per lead
  (e) area bias  dlnA vs scale   one line per lead  (+ = too big)
  (f) intensity  dlnρ vs scale   one line per lead  (− = too weak)

Panels (a,b) come from the per-system table; (c–f) from the summary CSV.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems_v2.csv.gz")
    p.add_argument("--summary", default="docs/scale_summary_v2.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--leads", type=float, nargs="+", default=[0.0, 3.0],
                   help="Lead times (h) to overlay. Default: 0 3.")
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
        opts = sorted(summ["predictor_set"].unique())
        raise SystemExit(f"no rows for predictor set {ps!r}; available: {opts}")

    leads = sorted(args.leads)
    avail = set(summ["lead_h"].unique())
    missing = [ld for ld in leads if not np.any(np.isclose(list(avail), ld))]
    if missing:
        print(f"note: no trained run at lead(s) {missing} h for {ps} — skipped")
        leads = [ld for ld in leads if ld not in missing]
    norm = Normalize(vmin=min(leads), vmax=max(leads))
    cmap = cm.viridis
    colors = {ld: cmap(norm(ld)) for ld in leads}

    # shared log-x edges for the size distribution, across all shown leads
    Lt_all = sysdf.loc[sysdf["L_t"].notna(), "L_t"]
    Lp_all = sysdf.loc[sysdf["L_p"].notna(), "L_p"]
    le = np.geomspace(min(Lt_all.min(), Lp_all.min()),
                      max(Lt_all.max(), Lp_all.max()), 30)

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))

    # ---- (a) system-size distribution ------------------------------------
    a = ax[0, 0]
    ld0 = leads[0]
    s0 = sysdf[sysdf["lead_h"] == ld0]
    a.hist(s0.loc[s0["L_t"].notna(), "L_t"], bins=le, histtype="step",
           color="k", lw=2.0, label="truth")
    for ld in leads:
        s = sysdf[sysdf["lead_h"] == ld]
        a.hist(s.loc[s["L_p"].notna(), "L_p"], bins=le, histtype="step",
               color=colors[ld], lw=1.5)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("system scale L = √A  (km)"); a.set_ylabel("# systems")
    a.set_title("(a) system-size distribution  (black = truth)")
    a.legend(fontsize=8, loc="upper right"); a.grid(True, which="both", alpha=0.25)

    # ---- total precipitating-area conservation, per lead -----------------
    # Integral of area = sum of A over every system (truth: match+miss,
    # pred: match+false_alarm). ratio ~1 => total raining area conserved.
    lines = ["∫A  pred/truth:"]
    print(f"total precipitating area  Σ A  (km²)  —  {ps}")
    for ld in leads:
        s = sysdf[sysdf["lead_h"] == ld]
        At = s.loc[s["L_t"].notna(), "A_t"].sum()
        Ap = s.loc[s["L_p"].notna(), "A_p"].sum()
        r = Ap / At if At else float("nan")
        verdict = "conserved" if 0.9 <= r <= 1.1 else (
            "pred LOW" if r < 0.9 else "pred HIGH")
        print(f"  lead {ld:>4g} h :  truth {At:.3e}   pred {Ap:.3e}   "
              f"pred/truth = {r:.2f}  ({verdict})")
        lines.append(f"{ld:>4g} h: {r:4.2f}")
    a.text(0.03, 0.03, "\n".join(lines), transform=a.transAxes,
           fontsize=8, va="bottom", ha="left", family="monospace",
           bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    # ---- (b) precip carried by each scale bin ----------------------------
    b = ax[0, 1]
    g0 = summ[summ["lead_h"] == ld0].sort_values("scale_mid")
    edges = np.r_[g0["scale_lo"].to_numpy(), g0["scale_hi"].to_numpy()[-1]]
    x = g0["scale_mid"].to_numpy()
    cats0 = pd.cut(s0["scale"], bins=edges)
    Pt = s0.groupby(cats0, observed=False)["P_t"].sum().to_numpy()
    b.plot(x, Pt, "o-", color="k", lw=2.0, label="truth")
    for ld in leads:
        s = sysdf[sysdf["lead_h"] == ld]
        cats = pd.cut(s["scale"], bins=edges)
        Pp = s.groupby(cats, observed=False)["P_p"].sum().to_numpy()
        b.plot(x, Pp, "s--", color=colors[ld])
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("system scale (km)"); b.set_ylabel("Σ P  (mm·km²)")
    b.set_title("(b) precip by scale  (black = truth)")
    b.legend(fontsize=8); b.grid(True, which="both", alpha=0.25)

    # ---- (c–f) summary-derived skill / bias, one line per lead -----------
    def overlay(axis, col, title, ylab, ylim=None, zeroline=False):
        for ld in leads:
            sub = summ[summ["lead_h"] == ld].sort_values("scale_mid")
            m = sub[col].notna()
            axis.plot(sub.loc[m, "scale_mid"], sub.loc[m, col], "o-", ms=4,
                      color=colors[ld], label=f"{ld:g} h")
        if zeroline:
            axis.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        axis.set_xscale("log")
        if ylim:
            axis.set_ylim(*ylim)
        axis.set_xlabel("system scale (km)"); axis.set_ylabel(ylab)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)

    overlay(ax[0, 2], "POD", "(c) detection: POD vs scale", "POD", (0, 1))
    overlay(ax[1, 0], "FAR", "(d) false alarms: FAR vs scale", "FAR", (0, 1))
    overlay(ax[1, 1], "dlnA_med", "(e) area bias  (+ = too big)",
            "median dln A", zeroline=True)
    overlay(ax[1, 2], "dlnrho_med", "(f) intensity bias  (− = too weak)",
            "median dln ρ", zeroline=True)

    fig.suptitle(f"Scale-aware decomposition — {ps}   "
                 f"(P = ρ·A,  dlnP = dlnA + dlnρ)", fontsize=14)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("lead time (h)")

    safe = ps.split()[0] + "_" + "_".join(f"{ld:g}h" for ld in leads)
    out = (repo / (args.out or f"docs/figures/scale_compare_{safe}.png")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
