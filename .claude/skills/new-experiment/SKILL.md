---
name: new-experiment
description: Scaffold a new experiment or sensitivity test folder with a hypothesis file, config, run script and PBS job. Use when starting any new run.
---

# New experiment

Usage: `/new-experiment <name> [--sensitivity <experiment it stresses>]`

1. Ask which R-number in `docs/FRAME.md` this serves. Refuse to scaffold if the answer
   is none; suggest `exploration/` instead.
2. Copy `experiments/_template/` to `experiments/<YYYY-MM>_<name>/`
   (or `sensitivity_tests/<name>/` with `--sensitivity`; then `HYPOTHESIS.md` must name the
   experiment stressed and the single factor varied).
3. Fill `HYPOTHESIS.md` with the user: hypothesis, falsification criterion, method.
   Hypothesis must state an expected sign or magnitude.
4. Fill `config.yaml`; every path under `data/`. Set `#PBS -N ls_<name>` in `job.sh`.
5. `run.py` imports only from `latentsky.*`. If a needed function is missing from `src/`,
   add it there with a test first.
