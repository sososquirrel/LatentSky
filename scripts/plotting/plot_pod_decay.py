"""POD vs lead time for each scale bin -- exponential decay fit per scale,
per model/run family (--tag). Two panels: raw POD(t), and POD(t)/POD(0h)
normalized + fit.

Genericized (2026-07) from a script hardcoded to docs/scale_summary_v3.csv
+ "9var (mse+uv)" into a --summary/--predictor-set/--tag CLI tool, using
the shared latentsky/fits.py exp-decay helper instead of a local copy.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from latentsky.fits import fit_exp_decay
from latentsky.registry import REPO_ROOT as REPO
from latentsky.paths import FIGURES_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)",
                    help="Ignored if --runs is given -- predictor_set (input-var label) "
                         "can't distinguish different architectures that share the same "
                         "input variables (e.g. old-LatentSky vs two-head both show up as "
                         "the same predictor_set in a unified multi-track summary CSV).")
    p.add_argument("--runs", nargs="+", default=None,
                    help="Filter to these exact `name` values instead of --predictor-set.")
    p.add_argument("--tag", default="v3", help="short model-family label, used in output filename")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df_all = pd.read_csv(REPO / args.summary)
    if args.runs is not None:
        df = df_all[df_all["name"].isin(args.runs)]
        if df.empty:
            opts = sorted(df_all["name"].unique())
            raise SystemExit(f"no rows for runs {args.runs!r}; available: {opts}")
    else:
        df = df_all[df_all["predictor_set"] == args.predictor_set]
        if df.empty:
            opts = sorted(df_all["predictor_set"].unique())
            raise SystemExit(f"no rows for predictor set {args.predictor_set!r}; available: {opts}")

    scales = sorted(df["scale_mid"].unique())
    lead_max = df["lead_h"].max()
    t_fine = np.linspace(0, lead_max, 200)

    cmap = cm.plasma
    n = len(scales)
    colors = [cmap(i / (n - 1)) for i in range(n)] if n > 1 else [cmap(0.5)]

    pod0 = df[df["lead_h"] == 0.0].set_index("scale_mid")["POD"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_raw, ax_norm = axes

    taus = []
    for scale, color in zip(scales, colors):
        sub = df[df["scale_mid"] == scale].sort_values("lead_h")
        t = sub["lead_h"].values.astype(float)
        p0 = pod0.get(scale, 1.0)
        if p0 <= 0:
            continue
        norm = (sub["POD"] / p0).values

        tau_fit = fit_exp_decay(t, norm)["tau"]
        taus.append(tau_fit)

        ax_raw.plot(t, sub["POD"].values, color=color, lw=1.8, marker="o", ms=4)

        ax_norm.plot(t, norm, color=color, lw=0, marker="o", ms=5)
        if np.isfinite(tau_fit):
            ax_norm.plot(t_fine, np.exp(-t_fine / tau_fit),
                         color=color, lw=1.8, alpha=0.85,
                         label=f"{scale:.0f} km  tau={tau_fit:.1f}h")

    ax_raw.set_xlabel("lead time [h]", fontsize=11)
    ax_raw.set_ylabel("POD", fontsize=11)
    ax_raw.set_title("(a) POD vs lead time", fontsize=11, loc="left", fontweight="bold")
    ax_raw.set_ylim(0, 1)
    ax_raw.grid(alpha=0.25)

    ax_norm.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.4)
    ax_norm.set_xlabel("lead time [h]", fontsize=11)
    ax_norm.set_ylabel("POD / POD(0h)", fontsize=11)
    ax_norm.set_title("(b) Normalised skill  --  fit: exp(-t/tau)  [B=0 forced]",
                      fontsize=11, loc="left", fontweight="bold")
    ax_norm.set_ylim(0, 1.05)
    ax_norm.grid(alpha=0.25)
    ax_norm.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    bounds = np.array(scales + [scales[-1] * 1.4])
    d_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    d_cmap = matplotlib.colors.ListedColormap(colors)
    sm = plt.cm.ScalarMappable(cmap=d_cmap,
         norm=matplotlib.colors.BoundaryNorm(np.arange(n + 1), n))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_raw, shrink=0.9, pad=0.02, ticks=np.arange(n) + 0.5)
    cb.set_ticklabels([f"{int(s)}" for s in scales])
    cb.set_label("system scale L [km]", fontsize=10)

    fig.suptitle(
        f"{args.tag} -- skill decay: POD/POD(0h) = exp(-t/tau)   [B=0 forced]\n"
        "Assumes skill goes to zero at long lead.",
        fontsize=9,
    )

    out = REPO / args.out if args.out else FIGURES_DIR / f"pod_decay_by_scale_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print("\nScale (km)  |  tau (h)")
    for s, t in zip(scales[:len(taus)], taus):
        print(f"  {s:6.0f}    |  {t:5.2f}")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
