"""τ vs scale: LatentSky vs GraphCast vs FuXi vs persistence — do AI weather
models share the same scale-dependent predictability transition, or is it a
LatentSky artifact? Persistence pins down the "do nothing" floor.

Caveat: GraphCast/FuXi/persistence have no lead=0 output in this multi-lead
setup (earliest available lead is 6h), so their POD(t) curves are fit with
a free-amplitude exponential A*exp(-t/tau), t in {6,12,18,24}h — unlike
LatentSky's fit, which is normalized to POD(0h)=1 (B=0 forced). Absolute tau
magnitudes are therefore not on strictly identical footing between LatentSky
and the rest; the log-log slope (tau ~ L^b) is the fair cross-model
comparison.

Domain of applicability: GraphCast and FuXi are archived on the 0.25° WB2
grid (~28 km spacing), so nothing below the Nyquist scale (~56 km) is a
meaningful independent sample — bins below that are dropped before
fitting/plotting for GraphCast and FuXi.

Persistence: NOT the ERA5-grid persistence (dropped — its 18h/24h lags are
diurnally contaminated, see git history for that analysis). Instead using
docs/scale_summary_baselines.csv's "persistence" predictor_set — persistence
computed on LatentSky's own DYAMOND-domain held-out test pieces, same grid
LatentSky itself is scored on. It has a real lead=0h anchor, so it's fit the
same POD(0h)-normalized way as LatentSky (not a free-amplitude fit), making
it directly, cleanly comparable to LatentSky's own curve.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = pathlib.Path(__file__).resolve().parents[1]
LEADS = [6, 12, 18, 24]


def load_multilead(model, leads=LEADS):
    frames = []
    for lead in leads:
        f = REPO / f"docs/scale_summary_{model}_2020-08_lead{lead}h_thr2p4.csv"
        df = pd.read_csv(f)
        df["lead_h"] = lead  # override buggy constant column with true forecast lead
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def free_exp(t, A, tau):
    return A * np.exp(-t / tau)


ERA5_NYQUIST_KM = 56.0  # 2x the 0.25 deg (~28 km) WB2 grid spacing


def restrict_to_resolved(sc, tau, cutoff):
    keep = sc >= cutoff
    return sc[keep], tau[keep]


def fit_ai_model(df):
    scales = sorted(df["scale_mid"].unique())
    sc_out, tau_out, pod6_out = [], [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        pod = sub["POD"].values
        if (pod <= 0).all():
            continue
        try:
            popt, _ = curve_fit(free_exp, t, pod, p0=[pod[0], 10.0],
                                bounds=([1e-4, 0.5], [1.0, 500.0]))
            sc_out.append(s)
            tau_out.append(popt[1])
            pod6_out.append(pod[0])  # POD at earliest available lead (6h)
        except Exception:
            pass
    span = float(df["lead_h"].max() - df["lead_h"].min())
    return np.array(sc_out), np.array(tau_out), np.array(pod6_out), span


def fit_latentsky_rain():
    """LatentSky + rain_log1p input sweep. No lead=0h in this sweep (starts
    at 0.25h) — normalized fit anchored at 0.25h instead, same method as
    LatentSky-no-rain and persistence (see fit_pod0_normalized)."""
    df = pd.read_csv(REPO / "docs/scale_summary_rain.csv")
    df = df[df["name"].str.startswith("mseuv_rain_lead")]
    return fit_pod0_normalized(df, anchor=0.25)


def fit_pod0_normalized(df, anchor=None):
    """Shared fitter for anything with a real t=0 (or earliest-available)
    anchor: normalize POD(t)/POD(anchor)=1 and fit exp(-(t-anchor)/tau)
    (B=0 forced, no free amplitude) — as opposed to the free-amplitude fit
    used for GraphCast/FuXi, which have no such anchor. Putting LatentSky,
    LatentSky+rain, and persistence all through this SAME normalized method
    (anchor=0h for the first two, 0.25h for the rain sweep, its earliest
    lead) is what makes their tau values comparable to each other — mixing
    a free-amplitude fit in for just one of them was an apples-to-oranges
    bug (rain's tau came out artificially low, looking "worse than
    persistence" only because of the fit method, not real skill)."""
    if anchor is None:
        anchor = df["lead_h"].min()
    scales = sorted(df["scale_mid"].unique())
    pod0 = df[df["lead_h"] == anchor].set_index("scale_mid")["POD"]
    sc_out, tau_out, pod0_out = [], [], []
    for s in scales:
        sub = df[df["scale_mid"] == s].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float) - anchor
        p0 = pod0.get(s, np.nan)
        if np.isnan(p0) or p0 <= 0:
            continue
        norm_pod = (sub["POD"] / p0).values
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), t, norm_pod,
                                p0=[2.0], bounds=([0.05], [500.0]))
            sc_out.append(s)
            tau_out.append(popt[0])
            pod0_out.append(p0)
        except Exception:
            pass
    span = float(df["lead_h"].max() - anchor)
    return np.array(sc_out), np.array(tau_out), np.array(pod0_out), span


def fit_latentsky():
    df = pd.read_csv(REPO / "docs/scale_summary_v3.csv")
    df = df[df["predictor_set"] == "9var (mse+uv)"]
    return fit_pod0_normalized(df)


def fit_persistence_dyamond():
    df = pd.read_csv(REPO / "docs/scale_summary_baselines.csv")
    df = df[df["predictor_set"] == "persistence"]
    return fit_pod0_normalized(df)


def powerlaw_fit(scales, taus):
    valid = np.isfinite(taus) & (taus > 0)
    if valid.sum() < 2:
        return None, None
    b, lna = np.polyfit(np.log(scales[valid]), np.log(taus[valid]), 1)
    return np.exp(lna), b


ls_sc, tau_ls, pod_ls, ls_span = fit_latentsky()
gc_sc, tau_gc, pod_gc, gc_span = fit_ai_model(load_multilead("graphcast"))
fx_sc, tau_fx, pod_fx, fx_span = fit_ai_model(load_multilead("fuxi"))
rn_sc, tau_rn, pod_rn, rn_span = fit_latentsky_rain()
pr_sc, tau_pr, pod_pr, pr_span = fit_persistence_dyamond()

gc_sc, tau_gc = restrict_to_resolved(gc_sc, tau_gc, ERA5_NYQUIST_KM)
fx_sc, tau_fx = restrict_to_resolved(fx_sc, tau_fx, ERA5_NYQUIST_KM)

fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

# One solid color per series, shared by markers and its fit line — no
# second (POD) color channel here, that just fought with series identity.
# Per-model POD-vs-lead detail lives in the standalone pod_vs_lead_*.png
# plots instead.
series = [
    ("LatentSky",              ls_sc, tau_ls, "o", "#1f77b4", ls_span),
    ("LatentSky+rain",         rn_sc, tau_rn, "P", "#e377c2", rn_span),
    ("GraphCast",              gc_sc, tau_gc, "s", "#2ca02c", gc_span),
    ("FuXi",                   fx_sc, tau_fx, "D", "#ff7f0e", fx_span),
    ("Persistence (DYAMOND)",  pr_sc, tau_pr, "^", "#7f7f7f", pr_span),
]

for label, sc, tau, marker, color, span in series:
    # tau > the lead-time span actually sampled means the fit is
    # EXTRAPOLATING past its own data, not interpolating — flag those
    # points as hollow/open so the reader can see which numbers are
    # well-constrained vs a guess. See module docstring.
    interp = tau <= span
    extrap = ~interp
    label_used = False
    if interp.any():
        ax.scatter(sc[interp], tau[interp], color=color, marker=marker, s=70,
                   alpha=0.85, edgecolors="k", linewidths=0.5, zorder=5, label=label)
        label_used = True
    if extrap.any():
        ax.scatter(sc[extrap], tau[extrap], facecolors="none", edgecolors=color,
                   marker=marker, s=70, linewidths=1.4, zorder=5,
                   label=None if label_used else label)
    a, b = powerlaw_fit(sc, tau)
    if a:
        # Fit line spans only this series' own resolved domain — no
        # extrapolating GraphCast/FuXi fits down into unresolved scales.
        L_fit = np.geomspace(sc.min(), sc.max(), 200)
        ax.plot(L_fit, a * L_fit ** b, color=color, lw=1.5, ls="--",
                alpha=0.8, label=f"  τ = {a:.3f}·L^{b:.2f}")

ax.axhline(6, color="0.4", lw=1.0, ls=":", zorder=1)
ax.text(28, 6.35, "6 h lead", fontsize=8, color="0.4")
ax.axvline(ERA5_NYQUIST_KM, color="0.4", lw=1.2, ls=":", zorder=1)
ax.text(ERA5_NYQUIST_KM * 1.05, 0.42,
        f"ERA5 grid Nyquist\n~{ERA5_NYQUIST_KM:.0f} km\n"
        "(GraphCast/FuXi domain cutoff)", fontsize=7, color="0.35")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("system scale  L = √area  [km]", fontsize=11)
ax.set_ylabel("POD decay timescale  τ  [h]", fontsize=11)
ax.set_title("Scale-dependent predictability — LatentSky (±rain input) vs GraphCast vs FuXi vs persistence\n"
             "LatentSky/persistence: 40-day DYAMOND-domain held-out; GraphCast/FuXi: August 2020 "
             "tropics ±40°, trimmed below ERA5 grid Nyquist (56 km)\n"
             "hollow markers: τ exceeds the fit's own lead-time span — extrapolated, not interpolated",
             fontsize=8.5)

handles, labels = ax.get_legend_handles_labels()
shape_h = [h for h, l in zip(handles, labels) if not l.startswith("  τ")]
shape_l = [l for l in labels if not l.startswith("  τ")]
ax.legend(shape_h, shape_l, frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, which="both")

out = REPO / "docs/figures/tau_vs_scale_ai_models.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}", flush=True)

print("\n=== power-law fits (tau = a * L^b) ===")
for label, sc, tau, marker, color, span in series:
    a, b = powerlaw_fit(sc, tau)
    n_extrap = int((tau > span).sum())
    if a:
        print(f"{label:22s}  a={a:.4f}  b={b:.3f}  n_scales={len(sc)}  "
              f"fit_span={span:.2f}h  extrapolated={n_extrap}/{len(sc)}")
