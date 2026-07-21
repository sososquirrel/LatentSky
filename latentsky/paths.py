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

from latentsky.registry import REPO_ROOT

DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"            # future: REPO_ROOT / "figures" / "main"
PAPER_FIGURES_DIR = DOCS_DIR / "paper_figures"  # future: REPO_ROOT / "figures" / "paper"
DOCS_RESULTS_DIR = DOCS_DIR                    # future: DOCS_DIR / "results"
