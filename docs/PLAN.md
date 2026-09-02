# LatentSky — reset plan

## Step 1 — Archive and restart the core
1. `git checkout -b reset` (keep `main` untouched until the end).
2. `bash setup_structure.sh` — creates the new tree and moves everything not in the keep-list into `_archive/`.
3. Rebuild `src/latentsky/` from `_archive/` one module at a time, in this order, each with a test:
   `models` → `training` → `metrics` → `evaluation` → `comparison`.
   Rule: a function comes back from `_archive/` only when a key result or an experiment needs it.
4. `_archive/` is git-tracked and deleted on the day of submission.

## Step 2 — Folders
```
latentsky/
├── src/latentsky/
│   ├── models/         # architectures, latent space, checkpoints loading
│   ├── training/       # data pipeline (DYAMOND), losses, train loop, SLURM launchers
│   ├── metrics/        # cluster identification, POD/FAR, area & intensity log-ratio, tau(L) fits
│   ├── evaluation/     # runs metrics on a forecast set → tidy tables (parquet/netCDF)
│   └── comparison/     # LatentSky vs persistence vs WB2 models; scaling laws
├── experiments/        # one folder per hypothesis: HYPOTHESIS.md + config.yaml + run.py + job.sh
├── sensitivity_tests/  # same structure; each names the experiment/result it stresses
├── exploration/        # notebooks, dead ends; nothing here is cited by a result
├── wb2_benchmark/      # WeatherBench2 models evaluated with the same metrics (the reference set)
├── results/            # only promoted results: fig_R<n>.py → fig_R<n>.pdf, REGISTRY.md
├── docs/FRAME.md       # thesis, key results, out of scope
├── journal/            # lab notebook, one entry per day, written by Claude (/journal)
├── data/ → $SCRATCH/latentsky (symlink, never in git)
├── _archive/
├── CLAUDE.md
└── .claude/
```

## Step 3 — Frame and CLAUDE.md
- Fill the `[to confirm]` fields in `docs/FRAME.md`.
- Lifecycle of a finding, enforced by CLAUDE.md and the skills:
  `exploration/` → `experiments/<name>/` (hypothesis stated) → `sensitivity_tests/<name>/` (survives) → `results/` (registered).
- Open VS Code (Remote-SSH to Derecho), open the Claude Code panel: it reads CLAUDE.md, then FRAME.md, and works inside the rules.
