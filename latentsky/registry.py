"""Run registry: discover and load past experiments by name.

Notebooks should never parse checkpoint filenames. Instead:

    from latentsky import list_runs, load_run
    runs = list_runs()                 # list of run names
    run = load_run("mseuv_lead3h")     # dict with config, manifest, ckpt path
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from latentsky.config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class Run:
    name: str
    path: Path
    config: ExperimentConfig
    manifest: dict

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoint.pt"

    @property
    def loss_history(self) -> dict:
        return self.manifest.get("loss_history", {})

    @property
    def best_val_loss(self) -> Optional[float]:
        return self.manifest.get("best_val_loss")

    @property
    def lead_hours(self) -> float:
        return self.config.lead_hours

    @property
    def predictors(self) -> list[str]:
        return self.config.input_vars


def _load_run(run_dir: Path) -> Run:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    cfg = ExperimentConfig(**manifest["config"])
    return Run(name=run_dir.name, path=run_dir, config=cfg, manifest=manifest)


def list_runs(runs_dir: Path | str = RUNS_DIR) -> list[str]:
    """Return all run names that have a manifest.json."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted(
        d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()
    )


def load_run(name: str, runs_dir: Path | str = RUNS_DIR) -> Run:
    runs_dir = Path(runs_dir)
    return _load_run(runs_dir / name)


def iter_runs(runs_dir: Path | str = RUNS_DIR) -> Iterator[Run]:
    for name in list_runs(runs_dir):
        yield load_run(name, runs_dir)
