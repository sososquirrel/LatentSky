"""Shared path constants for figures/docs output locations.

Originally introduced as a zero-behavior-change prep step for the docs/
reorganization (every script used to hardcode its own literal
"docs/figures/..." string, with no shared source of truth). The actual
figures move has now happened: docs/figures/ -> figures/main/,
docs/paper_figures/ -> figures/paper/. `docs/figures` and
`docs/paper_figures` are left as symlinks to the new locations, so any
script still using a hardcoded "docs/figures/..." literal (not yet
repointed at these constants) keeps working unchanged -- same safety net
used for the large CSV files moved to scratch earlier tonight.

The ~50-60 small final-summary CSVs -> docs/results/ move is still
deferred (too many independently-hardcoded path duplicates to do safely at
the same time as this pass).
"""
from __future__ import annotations

from pathlib import Path

from latentsky.registry import REPO_ROOT

DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = REPO_ROOT / "figures" / "main"
PAPER_FIGURES_DIR = REPO_ROOT / "figures" / "paper"
DOCS_RESULTS_DIR = DOCS_DIR                    # future: DOCS_DIR / "results"


def figure_path(fig_type: str, tag: str, lead_h: float | None = None) -> Path:
    """Standard output-naming convention for the modulable figure suite:
    FIGURES_DIR / "{fig_type}_{tag}[_lead{lead_h}h].png".

    `tag` should be a short MODEL-FAMILY label (e.g. "old_latentsky",
    "two_head", "diffusion") -- not the input-variable "predictor_set"
    label used elsewhere in this codebase, since architecture (not input
    variables) is the axis that was actually colliding on output filename
    across the three model tracks. One place defining this convention so
    multi-model/multi-lead figures never silently overwrite each other
    again.
    """
    name = f"{fig_type}_{tag}"
    if lead_h is not None:
        lead_str = f"{lead_h:g}"
        name += f"_lead{lead_str}h"
    return FIGURES_DIR / f"{name}.png"
