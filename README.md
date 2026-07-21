# LatentSky (clean)

Rebuild of [Tim Zhou's LatentSky](https://github.com/TimZzm/LatentSky) for the
precipitation-prediction-horizon study (Shamekh / Abramian).

The goal of this layout: every experiment is **one YAML in `configs/`**, every
trained model is **one `runs/<name>/` directory** with a manifest, and every
analysis notebook reads runs through `latentsky.registry` — no filename parsing.

## Layout

```
configs/      One YAML per experiment. Filename = experiment name = run name.
latentsky/    Python package: config, data, models, train, train_diffusion,
              registry, paths, metrics, inference, plotting, diffusion, losses.
runs/         Trained experiments (symlink to scratch), organized by
              architecture and input-variable count -- never read by path
              directly, always via `latentsky.registry` by NAME:
                unet_models/one_head/{5,9,10}var/<name>/
                unet_models/two_head/10var/<name>/
                gen_models/10var/<name>/               (the diffusion model)
              Each run dir contains: manifest.json, checkpoint.pt, train.log,
              loss.png. New runs are auto-filed into this tree by
              `latentsky.registry.classify_new_run()` -- you never need to
              pick the subfolder by hand.
notebooks/    Analysis notebooks, organized by question (not by date):
                01_horizon_vs_predictors.ipynb  — skill vs lead, per predictor set
                02_qualitative_maps.ipynb       — truth/pred/diff maps + histograms
                03_contingency.ipynb            — joint truth/prediction histograms
                04_scale_decomposition.ipynb    — object-based P = ρ·A skill by system scale
              All read runs via `from latentsky import list_runs, load_run`.
scripts/      analysis/    scaling-law / tau-fitting / decomposition compute scripts
              wb2/         WeatherBench2 model (GraphCast/FuXi/GenCast) fetch+eval+plot
              baselines/   climatology / persistence / random comparison scripts
              plotting/    everything else that produces a figure
              paper_figures/  scripts behind the paper's numbered figures
              (root)       one-off utilities: import_legacy_runs.py, make_rain_input.py,
                           make_split.py, setup_env.sh
docs/         CSV/figure outputs of the scripts above. `docs/figures/` and
              `docs/paper_figures/` hold PNGs; docs/*.csv[.gz] hold summary
              tables. Large regenerable `scale_systems_*.csv.gz` intermediate
              tables live on scratch (`/glade/derecho/scratch/$USER/
              LatentSky_docs_data/`) with symlinks left in place at their
              usual docs/ path. Scripts read/write these paths via the
              shared constants in `latentsky/paths.py`, not ad hoc strings.
submit/       Generic PBS script + helper to submit every config, plus one
              PBS script per one-off analysis/eval/plot job.
logs/         Archived PBS stdout/stderr (*.o<jobid>) log files.
archive/      Not imported by anything. Currently: stale_stubs/, a handful of
              incomplete/superseded training runs kept aside rather than deleted.
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
into `runs/` (auto-filed by architecture/var-count, same as any other run) and
writes a manifest. The notebooks then see both legacy and freshly trained runs
via the same `list_runs()` interface.

## What's in `archive/`

`stale_stubs/`: a handful of `runs/` entries that only ever got a partial
checkpoint (no manifest.json, so already invisible to `list_runs()`) --
kept aside rather than deleted, not meant to be read by anything.
