# LatentSky

Km-scale precipitation forecasting model trained on DYAMOND, evaluated with a
scale-decomposed precipitation metric. Goal: a finished, shareable repository and a paper.
Read `docs/FRAME.md` before doing anything else. It defines the thesis, the target
results, and what is out of scope. If a task does not serve a result in FRAME.md, say so.

## How we do science here

- Every experiment starts from a written hypothesis (`experiments/<name>/HYPOTHESIS.md`):
  what we expect, what observation would falsify it, and which result in FRAME.md it serves.
- A conclusion becomes a result only after it survives at least one sensitivity test
  (`sensitivity_tests/<name>/` names the experiment it stresses and what it varies:
  fit form, threshold, lead-time window, subset of days, resolution).
- A result is registered in `results/REGISTRY.md` with: R-number, claim (one sentence with
  a number in it), figure script, data path, sensitivity tests passed.
- A negative or modest result that answers a well-posed question is a result. We do not
  need to revolutionize the field; a clearly defined problem with a clear, robust answer
  is the standard. Do not inflate, do not soften.
- Exploration is welcome and lives in `exploration/`. Nothing there is cited by a result.
- Part 2 (WB2) is an outlook: one illustrative figure, no sensitivity requirement, no
  comparison with LatentSky. Do not expand it.

## Scientific style

Write in the tradition of Sophie Abramian, Caroline Muller, Sandrine Bony, Bjorn Stevens,
Pierre Gentine, Andreas Prein, Robert Pincus, Benjamin Fildier, Nadir Jeevanjee, Martin Singh.
In practice this means:

- Physically motivated. Start from a mechanism or a scaling argument, then test it.
  Metrics serve a physical question; they are never the point.
- Declarative sentences. State the finding, then the evidence. No "may suggest",
  "potentially", "it is interesting to note". Uncertainty is quantified, not hedged.
- Every claim is falsifiable and carries a number, a scale, or a lead time.
- Simple models and idealized limits before complex ones. Persistence is the null;
  beat the null before comparing to anything else.
- Prefer one clean figure that carries the argument over four that decorate it.
  Figures are typography-forward: one message per panel, axis labels with units,
  log axes where the physics is a power law, no chartjunk.
- Short paragraphs, one idea each. Equations when they clarify, not to impress.

## Repository rules

- Reusable code goes in `src/latentsky/` (models, training, metrics, evaluation, comparison),
  is importable, and has a test in `tests/`. Never copy functions into scripts or notebooks.
- `experiments/` and `sensitivity_tests/`: one folder per run, containing
  `HYPOTHESIS.md`, `config.yaml`, `run.py`, `job.sh`. Runs are reproducible from the config.
- `results/`: one script per figure (`fig_R<n>.py` → `fig_R<n>.pdf`). A file enters
  `results/` only with a `REGISTRY.md` entry. `make results` regenerates all of them.
- `wb2_benchmark/`: WeatherBench2 models evaluated with the same `src/latentsky/metrics`;
  this is the reference set, not a separate project.
- Data lives in `data/` → `$SCRATCH/latentsky` (symlink). Nothing large is committed.
- `_archive/`: legacy code. Read it to recover a function; never import from it.
  Recovered code is rewritten cleanly into `src/` with a test.
- Do not write to `exploration/` unless explicitly asked.
- Notebooks: outputs stripped (`nbstripout`), numbered `NN_topic.ipynb`.
- `journal/`: lab notebook, one file per day (`YYYY-MM-DD.md`). At session start, read
  `journal/INDEX.md` and the latest entry. At session end, or whenever an experiment
  finishes, run `/journal`. Every experiment — including failed ones — is documented with
  motivation, setup, numerical result and conclusion. Past entries are never edited.

## Environment

- Cluster: NCAR Derecho/Casper (user `sabramian`). GPU jobs via PBS (`qsub`), see
  `experiments/_template/job.sh`. Run heavy work on compute nodes, never on login nodes.
- Python env: conda `latentsky-env` at `/glade/work/sabramian/conda-envs/latentsky-env`.
  In PBS jobs call its `bin/python` directly (see `experiments/_template/job.sh`);
  do not create a `.venv` or pip-install into base.
- Tests: `pytest tests/`. Lint/format: `ruff`. Both run in pre-commit.
- Language: code and docs in English.

## Skills

- `/triage` — list orphaned figures and unreferenced scripts, propose moves to `_archive/`.
- `/new-experiment <name>` — scaffold an experiment folder with a hypothesis template.
- `/promote-result <experiment>` — check sensitivity coverage and write the REGISTRY entry.
- `/figure` — apply the figure conventions and register the figure.
- `/journal` — write today's lab-notebook entry and regenerate `journal/INDEX.md`.
