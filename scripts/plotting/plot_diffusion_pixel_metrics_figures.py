"""Figs 01 (confusion matrix), 02 (distribution), 03 (R2/RMSE vs lead) for
final_12_diffusion, built entirely from the persisted
docs/pixel_metrics_diffusion_lead{L}h.npz files -- no live diffusion
inference here, so this can be re-run freely (different bin subsets,
color scales, etc.) without ever touching the GPU again.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR
from latentsky.registry import REPO_ROOT as REPO

ALL_LEADS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]


def load(lead: str) -> dict:
    return dict(np.load(REPO / f"docs/pixel_metrics_diffusion_lead{lead}.npz"))


def fig01_confusion(leads: list[str], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, len(leads), figsize=(5.5 * len(leads), 5),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, lead in zip(axes, leads):
        d = load(lead)
        conf, bins = d["conf"], d["conf_bins"]
        n_bins = len(bins) - 1
        with np.errstate(divide="ignore"):
            log_conf = np.log(np.where(conf > 0, conf, np.nan))
        im = ax.imshow(log_conf, origin="upper", aspect="auto")
        ax.plot(np.arange(n_bins), np.arange(n_bins), "o", mfc="none", mec="red", ms=4)
        tick_idx = np.unique(np.linspace(0, n_bins, 7).astype(int))
        ax.set_xticks(tick_idx - 0.5)
        ax.set_yticks(tick_idx - 0.5)
        ax.set_xticklabels([f"{bins[i]:.2g}" for i in tick_idx], rotation=45, fontsize=7)
        ax.set_yticklabels([f"{bins[i]:.2g}" for i in tick_idx], fontsize=7)
        ax.set_title(f"diffusion, lead={lead}", fontsize=10)
        ax.set_xlabel("predicted (mm/15min)", fontsize=8)
    axes[0].set_ylabel("truth (mm/15min)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="log(count)")
    fig.suptitle("Diffusion -- contingency (truth vs predicted rain rate), full test-set scan")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig02_distribution(leads: list[str], out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(leads) - 1, 1)) for i in range(len(leads))]
    truth_plotted = False
    for lead, color in zip(leads, colors):
        d = load(lead)
        bins = d["hist_bins"]
        mids = np.sqrt(bins[:-1] * bins[1:])
        truth_dens = d["truth_hist"] / (d["truth_hist"].sum() * np.diff(bins))
        pred_dens = d["pred_hist"] / (d["pred_hist"].sum() * np.diff(bins))
        if not truth_plotted:
            ax.plot(mids, truth_dens, color="black", lw=2.2,
                    label=f"truth (n={int(d['truth_hist'].sum())})")
            truth_plotted = True
        ax.plot(mids, pred_dens, color=color, lw=1.8,
                label=f"pred, lead={lead} (n={int(d['pred_hist'].sum())})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rain rate (mm/15min), wet pixels only")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Diffusion -- pixel-level rain-rate distribution, full test-set scan")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig03_r2_rmse(leads: list[str], out: pathlib.Path) -> None:
    lead_h = []
    r2 = []
    rmse = []
    for lead in leads:
        d = load(lead)
        lead_h.append(float(d["lead_h"]))
        r2.append(float(d["r2"]))
        rmse.append(float(d["rmse"]))
    order = np.argsort(lead_h)
    lead_h = np.array(lead_h)[order]
    r2 = np.array(r2)[order]
    rmse = np.array(rmse)[order]

    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    ax_r2.plot(lead_h, r2, "o-", color="tab:blue")
    ax_r2.set_xlabel("lead time (h)")
    ax_r2.set_ylabel("R2")
    ax_r2.set_title("(a) R2 vs lead", loc="left", fontweight="bold")
    ax_r2.grid(alpha=0.3)

    ax_rmse.plot(lead_h, rmse, "o-", color="tab:red")
    ax_rmse.set_xlabel("lead time (h)")
    ax_rmse.set_ylabel("RMSE (mm/15min)")
    ax_rmse.set_title("(b) RMSE vs lead", loc="left", fontweight="bold")
    ax_rmse.grid(alpha=0.3)

    fig.suptitle("Diffusion -- bulk pixel accuracy vs lead, full test-set scan")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fig01-leads", nargs="+", default=["0.5h", "1h"])
    p.add_argument("--fig02-leads", nargs="+", default=["0.5h", "1h", "3h"])
    p.add_argument("--fig03-leads", nargs="+", default=ALL_LEADS)
    p.add_argument("--outdir", default=str(FIGURES_DIR / "final_12_diffusion"))
    args = p.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig01_confusion(args.fig01_leads, outdir / "01_confusion_matrix.png")
    fig02_distribution(args.fig02_leads, outdir / "02_distribution.png")
    fig03_r2_rmse(args.fig03_leads, outdir / "03_r2_rmse_vs_lead.png")


if __name__ == "__main__":
    main()
