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


def _manifest_paths(runs_dir: Path) -> list[Path]:
    """All manifest.json files under runs_dir, at any nesting depth (runs/
    is organized as unet_models/{one_head,two_head}/{n}var/<name>/ and
    gen_models/{n}var/<name>/, not flat) -- os.walk (not Path.rglob) because
    rglob doesn't descend into symlinked directories on Python < 3.13, and
    individual run dirs here are sometimes symlinks onto scratch."""
    import os
    found = []
    for dirpath, _dirnames, filenames in os.walk(runs_dir, followlinks=True):
        if "manifest.json" in filenames:
            found.append(Path(dirpath) / "manifest.json")
    return found


def list_runs(runs_dir: Path | str = RUNS_DIR) -> list[str]:
    """Return all run names that have a manifest.json, searched recursively
    (run directories may be nested under unet_models/.../ or gen_models/...).
    Raises if the same run name is found in more than one location."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    names = [p.parent.name for p in _manifest_paths(runs_dir)]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise RuntimeError(f"Ambiguous run name(s) found in multiple locations under {runs_dir}: {dupes}")
    return sorted(names)


def load_run(name: str, runs_dir: Path | str = RUNS_DIR) -> Run:
    runs_dir = Path(runs_dir)
    matches = [p.parent for p in _manifest_paths(runs_dir) if p.parent.name == name]
    if not matches:
        raise FileNotFoundError(f"No run named {name!r} found under {runs_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous run name {name!r}: found at {matches}")
    return _load_run(matches[0])


def iter_runs(runs_dir: Path | str = RUNS_DIR) -> Iterator[Run]:
    for name in list_runs(runs_dir):
        yield load_run(name, runs_dir)


def classify_new_run(cfg: ExperimentConfig, kind: str) -> Path:
    """Where a run belongs under runs/, derived from its config -- used both
    for the one-time historical reorg and to file all new runs going
    forward, so the two can never drift apart.

    kind: "unet" (latentsky.train) or "gen" (latentsky.train_diffusion).
    """
    n = len(cfg.input_vars)
    if kind == "gen":
        return Path("gen_models") / f"{n}var"
    if kind == "unet":
        arch = "two_head" if getattr(cfg, "two_head", False) else "one_head"
        return Path("unet_models") / arch / f"{n}var"
    raise ValueError(f"Unknown run kind: {kind!r} (expected 'unet' or 'gen')")
