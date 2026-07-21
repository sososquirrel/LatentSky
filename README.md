# LatentSky (clean)

Rebuild of [Tim Zhou's LatentSky](https://github.com/TimZzm/LatentSky) for the
precipitation-prediction-horizon study (Shamekh / Abramian).

The goal of this layout: every experiment is **one YAML in `configs/`**, every
trained model is **one `runs/<name>/` directory** with a manifest, and every
analysis notebook reads runs through `latentsky.registry` — no filename parsing.

## Layout

```
configs/      One YAML per experiment. Filename = experiment name = run name.
              Currently 26 entries: full lead × predictor-set sweep.
latentsky/    Python package: config, data, models, train, registry,
              metrics, inference, plotting.
runs/         One directory per trained experiment. Created by the trainer
              (or by scripts/import_legacy_runs.py for tzhou's checkpoints).
              Each contains: manifest.json, checkpoint.pt, train.log, loss.png.
notebooks/    Analysis notebooks, organized by question (not by date):
                01_horizon_vs_predictors.ipynb  — skill vs lead, per predictor set
                02_qualitative_maps.ipynb       — truth/pred/diff maps + histograms
                03_contingency.ipynb            — joint truth/prediction histograms
                04_scale_decomposition.ipynb    — object-based P = ρ·A skill by system scale
              All read runs via `from latentsky import list_runs, load_run`.
scripts/      One-off utilities. `import_legacy_runs.py` wraps tzhou's existing
              .pt files into runs/ so notebooks can read them without retraining.
submit/       Generic PBS script + helper to submit every config.
archive/      Old notebooks/scripts kept for reference. Not imported by anything.
```

## One-time setup: build the conda env

```bash
bash scripts/setup_env.sh
```

This creates `/glade/work/$USER/conda-envs/latentsky-env` from
[environment.yaml](environment.yaml) and registers it as a Jupyter kernel
named `Python (latentsky-env)` so it shows up in JupyterHub.

To activate manually:

```bash
module load conda/latest
conda activate /glade/work/$USER/conda-envs/latentsky-env
```

## Train one experiment

```bash
qsub -v CONFIG=configs/mseuv_lead3h.yaml -N mseuv_lead3h submit/submit.pbs
```

Local debug:

```bash
python -m latentsky.train --config configs/mseuv_lead3h.yaml
```

## Train every experiment

```bash
./submit/submit_all.sh
```

## Read results in a notebook

```python
from latentsky import list_runs, load_run

for name in list_runs():
    run = load_run(name)
    print(name, run.lead_hours, "h", run.predictors, run.best_val_loss)
```

## Adding a new experiment

1. Drop a new YAML in `configs/` (copy an existing one as template).
2. `qsub -v CONFIG=configs/new.yaml -N new submit/submit.pbs`.
3. Notebooks pick it up automatically next time `list_runs()` runs.

## Importing tzhou's existing checkpoints (no retraining)

```bash
python scripts/import_legacy_runs.py
```

Copies the matching `.pt` files from `/glade/u/home/tzhou/LatentSky/models/`
into `runs/<name>/checkpoint.pt` and writes a manifest. The notebooks then see
both legacy and freshly trained runs via the same `list_runs()` interface.

## What's in `archive/`

Tim's per-week notebooks and one-off scripts. Use these as reference for what
analyses have been done; do not import them. The intent is that everything
worth keeping is reborn as a config + a registry-driven notebook.
