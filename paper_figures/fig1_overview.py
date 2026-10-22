"""Paper Fig 1 (GRL) — the domain map (a) above the forecast pipeline (b-d).

One file for one publication unit: fig1a_domain.py and fig2_pipeline.py draw
into regions of the same canvas. The map keeps its latitude-label strip on
the left; the pipeline sits under the map axes, both 17.5 cm wide.
Reads the pipeline cache (run fig2_pipeline.py --compute first if missing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/glade/u/home/sabramian/LatentSky")
sys.path.insert(0, str(REPO / "paper_figures"))
from fig1_domain import CM, FH, FONT, FW, LABW, draw_domain  # noqa: E402
from fig1_pipeline import W2, draw_pipeline  # noqa: E402

OUT = REPO / "paper" / "figures"
PH = W2 * 0.45          # pipeline height, as in fig2_pipeline.pdf
SEP = 0.25 * CM         # gap between the map's colorbar label and the pipeline
LET = 0.5 * CM          # strip above the map for the (a) tag


def main():
    plt.rcParams.update(FONT)
    fig = plt.figure(figsize=(FW, FH + SEP + PH + LET))
    fw, fh = fig.get_size_inches()
    draw_domain(fig, 0.0, PH + SEP, letter="(a)")
    ax = fig.add_axes([LABW / fw, 0.005 * PH / fh, W2 / fw, 0.99 * PH / fh])
    draw_pipeline(ax, letters=("(b)", "(c)", "(d)"))
    fig.savefig(OUT / "fig1_overview.pdf", dpi=300)
    print("saved fig1_overview.pdf")


if __name__ == "__main__":
    main()
