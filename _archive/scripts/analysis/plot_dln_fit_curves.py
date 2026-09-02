"""tau(L) scaling law for the dlnA/dlnrho decomposition, with the area (A)
quantity split into 3 data-driven regimes rather than one global power law.

Why regimes: plotting the raw per-scale dlnA(t) curves showed area behaves
completely differently across scale -- small systems show a real, clean,
growing error with lead; mid-size systems show almost no lead-dependence at
all (flat); the largest bin is too data-poor to say. A single global power
law forced through all of that (tried first, see git history) either gets
dragged wildly off by the flat middle bins (when all bins are weighted by
n_match) or has to silently drop most of the domain (when reliability-
filtered) -- neither honestly represents "one scaling law," because there
isn't one. This script instead fits (up to) 3 independent regimes for A:

  Regime 1: MIN_SCALE_KM to L*_1 (small)
  Regime 2: L*_1 to L*_2         (mid)
  Regime 3: L*_2 to max_km       (large)

Regime boundaries are NOT hand-picked round numbers -- both come from the
data itself:
  L*_1 = the scale where diffusion's POD stops beating two-head's POD
         (pooled match/miss counts from the raw systems tables, both
         tracks, same bin edges) -- the scale below which the diffusion
         refinement step demonstrably helps detection, above which it
         doesn't. This lines up closely with where area's own decay signal
         disappears, which is why it's a sensible regime boundary rather
         than a coincidence.
  L*_2 = the scale where total matched systems (summed over leads) drops
         below MIN_TOTAL_N_MATCH -- i.e. where any fit becomes unreliable
         on sample-size grounds alone, independent of what the fit says.

Each regime is fit independently (own reliable-bin selection, own joint
fit, own bootstrap CI) -- a regime with fewer than 2 reliable bins reports
"no fit" rather than forcing one. dlnrho (intensity) does not get this
treatment: its own per-bin R2 is good almost everywhere, so it keeps a
single global fit.

Reads the raw systems tables directly (not pre-aggregated summary CSVs) so
the bootstrap has access to per-system values, not just per-bin medians.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.fits import fit_joint_tau_powerlaw, fit_tau_from_neg_abs_dln
from latentsky.paths import FIGURES_DIR

MIN_SCALE_KM = 56.0  # ERA5 Nyquist -- below this, matched systems are only 1-3 grid
              # pixels, so dlnA is quantized to a few discrete values, not real signal.
R2_MIN = 0.3  # a scale bin's own -|dln(t)| vs t fit has to clear this to count as
              # "a real decay was detected here", not just noise around a flat/non-monotonic
              # bias that happens to have a lot of matched systems behind it.
MIN_TOTAL_N_MATCH = 1000  # a scale bin also needs this many matched systems (summed over
              # leads) to be trusted for a tau(L) claim, independent of its R2 -- a
              # good-looking R2 from only ~150 systems is easy to get by chance and
              # shouldn't out-leverage bins with 10-100x the data.


def _standalone_fits(agg: pd.DataFrame) -> pd.DataFrame:
    """Per-bin standalone tau/R2/total-n (fit_tau_from_neg_abs_dln), one row
    per scale_mid -- used both to select reliable bins and for diagnostics."""
    rows = []
    for smid, sub in agg.groupby("scale_mid"):
        sub = sub.sort_values("lead_h")
        f = fit_tau_from_neg_abs_dln(sub["lead_h"].to_numpy(), sub["dln_med"].to_numpy(),
                                      sub["n_match"].to_numpy())
        rows.append(dict(scale_mid=smid, tau=f["tau"], tau_se=f["tau_se"], R2=f["R2"],
                         total_n=int(sub["n_match"].sum())))
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(["lead_h", "scale_mid"])[col].agg(["median", "count"]).reset_index()
    return g.rename(columns={"median": "dln_med", "count": "n_match"})


def _fit_regime(sysdf: pd.DataFrame, col: str, lo: float, hi: float, n_boot: int, rng) -> dict:
    """Reliable-bin joint fit + bootstrap CI, restricted to scale_mid in [lo, hi).
    Returns a dict with standalone fits (all bins in range, for markers),
    reliable_scales, and either a joint fit + bootstrap curves, or fit=None
    if fewer than 2 bins in this range pass the reliability filter."""
    sub = sysdf[(sysdf["scale_mid"] >= lo) & (sysdf["scale_mid"] < hi)]
    out = dict(lo=lo, hi=hi, standalone=pd.DataFrame(), reliable_scales=set(), fit=None)
    if sub.empty:
        return out
    g = _aggregate(sub, col)
    sf = _standalone_fits(g)
    out["standalone"] = sf
    rel = sf[(sf["R2"] >= R2_MIN) & (sf["total_n"] >= MIN_TOTAL_N_MATCH)]
    rel_scales = set(rel["scale_mid"])
    out["reliable_scales"] = rel_scales
    if len(rel_scales) < 2:
        return out

    gk = g[g["scale_mid"].isin(rel_scales)]
    pl = fit_joint_tau_powerlaw(gk["lead_h"].to_numpy(), gk["scale_mid"].to_numpy(),
                                 gk["dln_med"].to_numpy(), gk["n_match"].to_numpy())
    out["fit"] = pl

    Lgrid = np.geomspace(min(rel_scales), max(rel_scales), 40)
    groups = {}
    for (lead, smid), s in sub[sub["scale_mid"].isin(rel_scales)].groupby(["lead_h", "scale_mid"]):
        groups[(lead, smid)] = s[col].to_numpy()
    keys = list(groups.keys())
    boot_curves = np.full((n_boot, len(Lgrid)), np.nan)
    boot_beta = np.full(n_boot, np.nan)
    for i in range(n_boot):
        leads_b, scales_b, med_b, n_b = [], [], [], []
        for (lead, smid) in keys:
            arr = groups[(lead, smid)]
            resample = arr[rng.integers(0, len(arr), size=len(arr))]
            leads_b.append(lead); scales_b.append(smid)
            med_b.append(np.median(resample)); n_b.append(len(arr))
        pl_b = fit_joint_tau_powerlaw(np.array(leads_b), np.array(scales_b),
                                      np.array(med_b), np.array(n_b))
        boot_beta[i] = pl_b["beta"]
        if np.isfinite(pl_b["beta"]):
            boot_curves[i] = pl_b["tau_ref"] * (Lgrid / pl_b["L_ref"]) ** pl_b["beta"]
    out["Lgrid"] = Lgrid
    out["boot_curves"] = boot_curves
    out["boot_beta"] = boot_beta
    return out


def _pod_crossover(diffusion_systems: pd.DataFrame, twohead_path: str, twohead_runs: list,
                    edges: np.ndarray, repo) -> float | None:
    """Scale L* where diffusion's pooled POD stops exceeding two-head's pooled
    POD (log-interpolated between the two bracketing bins). Returns None if
    no crossing is found (e.g. one track dominates at every scale)."""
    def pooled_pod(df):
        d = df[df["role"].isin(["match", "miss"])].copy()
        b = np.clip(np.digitize(d["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
        scale_mid = np.sqrt(edges[:-1] * edges[1:])
        d["scale_mid"] = scale_mid[b]
        n_match = d[d["role"] == "match"].groupby("scale_mid").size()
        n_miss = d[d["role"] == "miss"].groupby("scale_mid").size()
        idx = sorted(set(n_match.index) | set(n_miss.index))
        nm = n_match.reindex(idx, fill_value=0)
        nmi = n_miss.reindex(idx, fill_value=0)
        return (nm / (nm + nmi)).replace([np.inf, -np.inf], np.nan)

    pod_diff = pooled_pod(diffusion_systems)
    th = pd.read_csv(repo / twohead_path)
    th = th[th["name"].isin(twohead_runs)]
    pod_th = pooled_pod(th)

    scales = sorted(set(pod_diff.index) & set(pod_th.index))
    d = np.array([pod_diff[s] - pod_th[s] for s in scales])
    for i in range(len(scales) - 1):
        if d[i] > 0 and d[i + 1] < 0:
            l0, l1 = scales[i], scales[i + 1]
            frac = d[i] / (d[i] - d[i + 1])
            return float(np.exp(np.log(l0) + frac * (np.log(l1) - np.log(l0))))
    return None


def _n_match_threshold(sf_all: pd.DataFrame, min_n: float) -> float | None:
    """Scale where total matched systems crosses below min_n (log-interpolated)."""
    sf = sf_all.sort_values("scale_mid")
    scales = sf["scale_mid"].to_numpy()
    n = sf["total_n"].to_numpy()
    for i in range(len(scales) - 1):
        if n[i] >= min_n and n[i + 1] < min_n:
            frac = (np.log(n[i]) - np.log(min_n)) / (np.log(n[i]) - np.log(n[i + 1]))
            return float(np.exp(np.log(scales[i]) + frac * (np.log(scales[i + 1]) - np.log(scales[i]))))
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems_diffusion.csv.gz")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--twohead-systems", default="docs/scale_systems_modelcompare.csv.gz")
    p.add_argument("--twohead-runs", nargs="+",
                    default=[f"patchsplit_twohead_olr_fss_lead{L}"
                             for L in ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]])
    p.add_argument("--tag", default="diffusion")
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--min-km", type=float, default=25.0)
    p.add_argument("--max-km", type=float, default=3000.0)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from latentsky.registry import REPO_ROOT as repo
    from latentsky.scale_analysis import default_scale_edges
    edges = default_scale_edges(args.min_km, args.max_km, args.n_bins)
    scale_mid_all = np.sqrt(edges[:-1] * edges[1:])

    raw = pd.read_csv(repo / args.systems)
    raw = raw[raw["name"].isin(args.runs)]
    if raw.empty:
        raise SystemExit(f"no rows for runs {args.runs!r} in {args.systems!r}")

    sysdf = raw[raw["role"] == "match"].copy()
    b = np.clip(np.digitize(sysdf["scale"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    sysdf["scale_mid"] = scale_mid_all[b]
    sysdf = sysdf[sysdf["scale_mid"] >= MIN_SCALE_KM]

    # ---- data-driven regime boundaries for area ----
    lstar1 = _pod_crossover(raw, args.twohead_systems, args.twohead_runs, edges, repo)
    sf_all = _standalone_fits(_aggregate(sysdf, "dlnA"))
    lstar2 = _n_match_threshold(sf_all, MIN_TOTAL_N_MATCH)
    if lstar1 is None or lstar2 is None or not (lstar1 < lstar2):
        raise SystemExit(f"could not determine regime boundaries (L*_1={lstar1}, L*_2={lstar2})")
    print(f"regime boundaries for area: L*_1={lstar1:.1f}km (POD crossover, diffusion vs "
          f"two-head), L*_2={lstar2:.1f}km (n_match={MIN_TOTAL_N_MATCH:g} threshold)")

    rng = np.random.default_rng(args.seed)
    area_regimes = [
        ("small (diffusion > two-head POD)", MIN_SCALE_KM, lstar1),
        ("mid (two-head >= diffusion POD)", lstar1, lstar2),
        ("large (data-limited)", lstar2, args.max_km),
    ]
    area_fits = [(label, lo, hi, _fit_regime(sysdf, "dlnA", lo, hi, args.n_boot, rng))
                 for label, lo, hi in area_regimes]
    for label, lo, hi, r in area_fits:
        if r["fit"] is not None:
            pl = r["fit"]
            blo, bhi = np.nanpercentile(r["boot_beta"], [2.5, 97.5])
            print(f"area [{label}] {lo:.0f}-{hi:.0f}km: tau_ref={pl['tau_ref']:.2f}h @ "
                  f"L_ref={pl['L_ref']:.0f}km beta={pl['beta']:.2f} [{blo:.2f},{bhi:.2f}] "
                  f"R2={pl['R2']:.2f} ({len(r['reliable_scales'])} reliable bins)")
        else:
            print(f"area [{label}] {lo:.0f}-{hi:.0f}km: no fit -- "
                  f"{len(r['reliable_scales'])} reliable bin(s), need >=2")

    rho_fit = _fit_regime(sysdf, "dlnrho", MIN_SCALE_KM, args.max_km, args.n_boot, rng)
    pl = rho_fit["fit"]
    blo, bhi = np.nanpercentile(rho_fit["boot_beta"], [2.5, 97.5])
    print(f"rho [global]: tau_ref={pl['tau_ref']:.2f}h @ L_ref={pl['L_ref']:.0f}km "
          f"beta={pl['beta']:.2f} [{blo:.2f},{bhi:.2f}] R2={pl['R2']:.2f} "
          f"({len(rho_fit['reliable_scales'])} reliable bins)")

    # ==== Figure: tau(L) with 3 area regimes + 1 global rho fit ====
    fig, ax = plt.subplots(figsize=(8, 6.2))
    area_colors = ["#1f4e8c", "#4a90d9", "#a8c8ec"]  # dark -> light blue, small -> large regime
    for (label, lo, hi, r), color in zip(area_fits, area_colors):
        sf = r["standalone"]
        if sf.empty:
            continue
        finite = np.isfinite(sf["tau"].to_numpy())
        used = sf["scale_mid"].isin(r["reliable_scales"]).to_numpy() & finite
        excluded = finite & ~used
        L_pts, tau_pts = sf["scale_mid"].to_numpy(), sf["tau"].to_numpy()
        ax.scatter(L_pts[used], tau_pts[used], color=color, marker="o", s=70,
                   edgecolors="k", linewidths=0.6, zorder=4,
                   label=f"area [{label}]")
        if excluded.any():
            ax.scatter(L_pts[excluded], tau_pts[excluded], color=color, marker="o", s=70,
                       edgecolors="k", linewidths=0.6, alpha=0.3, zorder=4)
        if r["fit"] is not None and np.isfinite(r["fit"]["beta"]):
            pl = r["fit"]
            lo_ci = np.nanpercentile(r["boot_curves"], 2.5, axis=0)
            hi_ci = np.nanpercentile(r["boot_curves"], 97.5, axis=0)
            ax.fill_between(r["Lgrid"], lo_ci, hi_ci, color=color, alpha=0.2, zorder=1)
            blo, bhi = np.nanpercentile(r["boot_beta"], [2.5, 97.5])
            ax.plot(r["Lgrid"], pl["tau_ref"] * (r["Lgrid"] / pl["L_ref"]) ** pl["beta"],
                    "--", color=color, lw=1.8, zorder=3,
                    label=f"  beta={pl['beta']:.2f} [{blo:.2f},{bhi:.2f}] R2={pl['R2']:.2f}")

    sf = rho_fit["standalone"]
    finite = np.isfinite(sf["tau"].to_numpy())
    used = sf["scale_mid"].isin(rho_fit["reliable_scales"]).to_numpy() & finite
    excluded = finite & ~used
    L_pts, tau_pts = sf["scale_mid"].to_numpy(), sf["tau"].to_numpy()
    ax.scatter(L_pts[used], tau_pts[used], color="C3", marker="s", s=70,
              edgecolors="k", linewidths=0.6, zorder=4, label="rho [global]")
    if excluded.any():
        ax.scatter(L_pts[excluded], tau_pts[excluded], color="C3", marker="s", s=70,
                  edgecolors="k", linewidths=0.6, alpha=0.3, zorder=4)
    if rho_fit["fit"] is not None and np.isfinite(rho_fit["fit"]["beta"]):
        pl = rho_fit["fit"]
        lo_ci = np.nanpercentile(rho_fit["boot_curves"], 2.5, axis=0)
        hi_ci = np.nanpercentile(rho_fit["boot_curves"], 97.5, axis=0)
        ax.fill_between(rho_fit["Lgrid"], lo_ci, hi_ci, color="C3", alpha=0.15, zorder=1)
        blo, bhi = np.nanpercentile(rho_fit["boot_beta"], [2.5, 97.5])
        ax.plot(rho_fit["Lgrid"], pl["tau_ref"] * (rho_fit["Lgrid"] / pl["L_ref"]) ** pl["beta"],
                "--", color="C3", lw=1.8, zorder=3,
                label=f"  beta={pl['beta']:.2f} [{blo:.2f},{bhi:.2f}] R2={pl['R2']:.2f}")

    for lstar in (lstar1, lstar2):
        ax.axvline(lstar, color="0.5", lw=1.0, ls=":", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("system scale L = √A  (km)")
    ax.set_ylabel("tau  (h)   [info_X(t) = I0 exp(-t/tau)]")
    ax.set_title(f"tau(L): area split into 3 data-driven regimes, intensity global -- {args.tag}\n"
                 f"L*_1={lstar1:.0f}km (POD crossover), L*_2={lstar2:.0f}km (n_match floor); "
                 f"bands = {args.n_boot}-resample bootstrap 95% CI")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    out = FIGURES_DIR / f"tau_vs_scale_regimes_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
