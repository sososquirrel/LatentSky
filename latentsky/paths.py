"""Shared path constants for docs/figures output locations.

Introduced as a zero-behavior-change prep step for the docs/ reorganization:
today every script hardcodes its own literal "docs/figures/..." or
"docs/<name>.csv" string, with no shared source of truth (the same string is
independently duplicated across writer and reader scripts). This module
gives them one place to point at.

This pass only introduces the constants and repoints existing scripts at
them -- **nothing physically moves yet**, so every constant below resolves
to exactly where its content lives on disk today. The actual reorg (moving
docs/figures/ -> figures/main/, docs/paper_figures/ -> figures/paper/, and
the ~50-60 small final-summary CSVs -> docs/results/) is a deferred future
session: when that happens, only these four definitions need to change, not
the ~100+ call sites that import them.
"""
from __future__ import annotations

from pathlib import Path

from latentsky.registry import REPO_ROOT

DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"            # future: REPO_ROOT / "figures" / "main"
PAPER_FIGURES_DIR = DOCS_DIR / "paper_figures"  # future: REPO_ROOT / "figures" / "paper"
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
