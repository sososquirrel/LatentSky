"""Re-bin the diffusion track's already-computed per-system table
(docs/scale_systems_diffusion.csv.gz -- no new inference needed) at
specific requested scale values, then plot dlnA/dlnrho vs lead per bin.

Bin edges are placed at the geometric mean between consecutive requested
L values (guaranteed monotonic for any increasing L list -- solving for
edges that make scale_mid *exactly* equal every requested L can produce
invalid non-monotonic edges when the L list isn't evenly log-spaced, as
is the case here). Only the first and last bin centers land exactly on
the requested value; interior ones are close (see printed comparison).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from latentsky.paths import FIGURES_DIR
from latentsky.scale_analysis import summarize_by_scale


def custom_edges(L: np.ndarray) -> np.ndarray:
    edges = np.empty(len(L) + 1)
    edges[1:-1] = np.sqrt(L[:-1] * L[1:])
    edges[0] = L[0] / np.sqrt(L[1] / L[0])
    edges[-1] = L[-1] * np.sqrt(L[-1] / L[-2])
    assert np.all(np.diff(edges) > 0), "requested L values produced non-monotonic edges"
    return edges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, nargs="+", default=[30, 50, 80, 100, 200, 300, 500, 800])
    p.add_argument("--systems", default="docs/scale_systems_diffusion.csv.gz")
    p.add_argument("--name-prefix", default="diffusion_lead")
    p.add_argument("--tag", default="diffusion")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    L = np.array(sorted(args.L), dtype=float)
    edges = custom_edges(L)
    print("requested L: ", L)
    print("actual bin scale_mid:", np.round(np.sqrt(edges[:-1] * edges[1:]), 2))

    df = pd.read_csv(args.systems)
    df = df[df["name"].str.startswith(args.name_prefix)]
    if df.empty:
        raise SystemExit(f"no rows with name starting {args.name_prefix!r} in {args.systems}")

    rows = []
    for name, g in df.groupby("name"):
        summ = summarize_by_scale(g, edges).by_scale
        summ.insert(0, "name", name)
        summ.insert(0, "lead_h", g["lead_h"].iloc[0])
        rows.append(summ)
    summary = pd.concat(rows, ignore_index=True)

    scales = sorted(summary["scale_mid"].unique())
    cmap = cm.viridis
    colors = [cmap(i / (len(scales) - 1)) for i in range(len(scales))]

    fig, (ax_a, ax_rho) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for L_actual, color in zip(scales, colors):
        sub = summary[np.isclose(summary["scale_mid"], L_actual)].sort_values("lead_h")
        label = f"L~{L_actual:.0f} km"
        ax_a.plot(sub["lead_h"], sub["dlnA_med"], "o-", color=color, label=label)
        ax_rho.plot(sub["lead_h"], sub["dlnrho_med"], "o-", color=color, label=label)

    for ax, title in [(ax_a, "dln A (area error)"), (ax_rho, "dln rho (intensity error)")]:
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel("median log error (pred/truth)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.3)
    ax_rho.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"Precip decomposition vs lead, custom scale bins -- {args.tag}\n"
                 f"requested L={[int(x) for x in L]} km")

    out = args.out or str(FIGURES_DIR / f"final_12_diffusion/decomposition_custom_L_{args.tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
