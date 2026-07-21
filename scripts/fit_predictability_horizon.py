"""Fit the conditional-predictability decay model to the object-based skill
curves, separately for the TRIGGER (detection) and the AMPLITUDE (intensity),
and by system scale. Tests two hypotheses:

  H1  trigger decorrelates faster than amplitude:  Te(detection) < Te(intensity)
  H2  small systems decorrelate faster than large:  Te(small) < Te(large)

Skill model (Eq. 2 in the paper):
    S(tau) = (S0 - Sinf) * exp(-tau / Te) + Sinf

Detection skill  = POD (probability of detection) per lead.
Amplitude skill  = Pearson correlation of ln(rho) over matched systems
                   (intensity skill conditional on detection).

Reads docs/scale_systems_v3.csv.gz (all leads, 9var mse+uv).
Writes docs/predictability_horizon.csv and docs/figures/predictability_horizon.png
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
SYS = REPO / "docs" / "scale_systems_v3.csv.gz"
OUT_CSV = REPO / "docs" / "predictability_horizon.csv"
OUT_FIG = REPO / "docs" / "figures" / "predictability_horizon.png"

SMALL_HI = 65.0      # km, "small" systems
LARGE_LO = 200.0     # km, "large" systems


def decay(tau, S0, Sinf, Te):
    return (S0 - Sinf) * np.exp(-tau / Te) + Sinf


def fit_decay(tau, S):
    tau = np.asarray(tau, float); S = np.asarray(S, float)
    ok = np.isfinite(S)
    tau, S = tau[ok], S[ok]
    if len(tau) < 4:
        return dict(S0=np.nan, Sinf=np.nan, Te=np.nan)
    p0 = [S[0], S[-1], 1.0]
    try:
        popt, _ = curve_fit(decay, tau, S, p0=p0,
                            bounds=([-1, -1, 0.05], [2, 2, 50]), maxfev=20000)
        return dict(S0=popt[0], Sinf=popt[1], Te=popt[2])
    except Exception:
        return dict(S0=np.nan, Sinf=np.nan, Te=np.nan)


def detection_skill(df):
    nm = (df.role == "match").sum(); nmiss = (df.role == "miss").sum()
    n = nm + nmiss
    return nm / n if n else np.nan


def amplitude_skill(df):
    m = df[df.role == "match"]
    m = m[(m.rho_t > 0) & (m.rho_p > 0)]
    if len(m) < 20:
        return np.nan
    return np.corrcoef(np.log(m.rho_t), np.log(m.rho_p))[0, 1]


def main():
    df = pd.read_csv(SYS)
    df = df[df.predictor_set == "9var (mse+uv)"].copy()
    df["scale_use"] = np.where(df.role == "false_alarm", df.L_p, df.L_t)
    leads = sorted(df.lead_h.unique())

    bands = {"small (<65km)": (0, SMALL_HI),
             "large (>200km)": (LARGE_LO, 1e9),
             "all": (0, 1e9)}

    rows = []
    curves = {}
    for band, (lo, hi) in bands.items():
        sub = df[(df.scale_use >= lo) & (df.scale_use < hi)]
        det = [detection_skill(sub[sub.lead_h == t]) for t in leads]
        amp = [amplitude_skill(sub[sub.lead_h == t]) for t in leads]
        curves[band] = dict(det=np.array(det), amp=np.array(amp))
        for metric, S in [("detection", det), ("intensity", amp)]:
            f = fit_decay(leads, S)
            f.update(band=band, metric=metric)
            rows.append(f)

    res = pd.DataFrame(rows)[["band", "metric", "S0", "Te", "Sinf"]]
    res.to_csv(OUT_CSV, index=False)
    print(res.round(3).to_string(index=False))

    # ---- figure ----
    tt = np.linspace(0, 6, 100)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    cols = {"small (<65km)": "#d62728", "large (>200km)": "#1f77b4", "all": "0.4"}
    for j, metric in enumerate(["det", "amp"]):
        mname = {"det": "detection", "amp": "intensity"}[metric]
        for band in bands:
            S = curves[band][metric]
            c = cols[band]
            ax[j].plot(leads, S, "o", color=c, ms=6)
            f = res[(res.band == band) & (res.metric == mname)].iloc[0]
            if np.isfinite(f.Te):
                ax[j].plot(tt, decay(tt, f.S0, f.Sinf, f.Te), "-", color=c,
                           label=f"{band}: $T_e$={f.Te:.2f} h")
        ax[j].set_xlabel("lead time $\\tau$ [h]")
        ax[j].set_title(f"({'ab'[j]}) {'TRIGGER (detection POD)' if j==0 else 'AMPLITUDE (intensity corr.)'}",
                        loc="left", fontsize=11)
        ax[j].grid(alpha=0.3); ax[j].legend(frameon=False, fontsize=9)
        ax[j].set_xlim(0, 6)
    ax[0].set_ylabel("detection skill (POD)")
    ax[1].set_ylabel("amplitude skill (corr. of $\\ln\\rho$)")
    fig.suptitle("Conditional-predictability decay: trigger vs amplitude, by scale "
                 "($S(\\tau)=(S_0-S_\\infty)e^{-\\tau/T_e}+S_\\infty$)", y=1.05, fontsize=11)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT_FIG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
