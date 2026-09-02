---
name: triage
description: Audit the repository for orphaned figures, unreferenced scripts and duplicated functions, and propose moves to _archive/. Use when cleaning up or when asked which files are still needed.
---

# Triage

Goal: everything in the repo either serves a result in `docs/FRAME.md` or lives in
`exploration/` or `_archive/`.

1. Read `docs/FRAME.md` and `results/REGISTRY.md`.
2. List every figure file (`*.pdf *.png *.svg`) outside `_archive/`. For each, grep the repo
   for its filename or stem. No script produces it → orphan.
3. List every `*.py` and `*.ipynb` outside `src/`, `tests/`, `_archive/`. For each: is it
   imported or called by an experiment, a sensitivity test, a `fig_R*.py`, or `Makefile`?
   No → unreferenced.
4. In `src/` and scripts, find functions with the same name or near-identical bodies
   defined more than once. Report the duplicates and which one is used by a result.
5. Output a table: path | category (orphan figure / unreferenced script / duplicate) |
   proposed action (archive / move to exploration / merge into src) | reason.
6. Do not move or delete anything. Wait for approval, then execute with `git mv`.
