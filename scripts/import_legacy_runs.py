"""Wrap tzhou's existing checkpoints into the new runs/ layout.

For each (legacy filename, config) pair this script:
  - copies the .pt to runs/<config.name>/checkpoint.pt (no re-training)
  - writes runs/<config.name>/manifest.json with the config
    and an "imported_from" field pointing to the original .pt path

After running, list_runs() surfaces the legacy models alongside any new
ones; notebooks treat them uniformly.

Usage:
    python scripts/import_legacy_runs.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Make `latentsky` importable when run as `python scripts/import_legacy_runs.py`
# from the repo root — no install required.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latentsky.config import ExperimentConfig
from latentsky.registry import REPO_ROOT

LEGACY_MODELS_DIR = Path("/glade/u/home/tzhou/LatentSky/models")


def _step_to_lead_str(steps: int) -> str:
    h = steps * 0.25
    return f"{h:g}h"


# Build the mapping: family -> {steps: legacy_filename}.
# Filenames follow tzhou's pattern:
#   unet_rain_1472_<depth>_64_<steps>step_<family-token>_lead<hh>h_ep<ep>_Sreg_<ep>ep.pt
# epoch suffix differs per family (mseuv→155, humid→150, mse→155 for steps>0 else 150).
LEGACY_TABLE = {
    # mseuv: depth=5, all 155 epochs
    "mseuv": {
        s: f"unet_rain_1472_5_64_{s}step_mseuv_lead{_step_to_lead_str(s)}_ep155_Sreg_155ep.pt"
        for s in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24)
    },
    # humid: depth=4, all 150 epochs
    "humid": {
        s: f"unet_rain_1472_4_64_{s}step_humid_114514_lead{_step_to_lead_str(s)}_ep150_Sreg_150ep.pt"
        for s in (0, 1, 2, 3, 4, 6, 8, 12)
    },
    # mse: depth=4, 155ep except step=0 which is 150ep
    "mse": {
        s: (
            f"unet_rain_1472_4_64_{s}step_mse_1919810_lead{_step_to_lead_str(s)}_"
            f"ep{'150' if s == 0 else '155'}_Sreg_{'150' if s == 0 else '155'}ep.pt"
        )
        for s in (0, 1, 2, 3, 4, 6, 8, 12)
    },
}


def import_one(pt_name: str, cfg_name: str) -> bool:
    pt_src = LEGACY_MODELS_DIR / pt_name
    cfg_path = REPO_ROOT / "configs" / f"{cfg_name}.yaml"
    if not cfg_path.exists():
        print(f"  SKIP (no config): {cfg_name}")
        return False
    if not pt_src.exists():
        print(f"  SKIP (no .pt):    {pt_name}")
        return False

    cfg = ExperimentConfig.from_yaml(cfg_path)
    dst = REPO_ROOT / "runs" / cfg.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pt_src, dst / "checkpoint.pt")

    manifest = {
        "name": cfg.name,
        "git_sha": "imported",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "loss_history": {"train": [], "val": []},
        "best_val_loss": None,
        "epochs_completed": None,
        "checkpoint_relpath": "checkpoint.pt",
        "imported_from": str(pt_src),
    }
    with open(dst / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  imported {cfg.name}")
    return True


def main() -> None:
    print(f"Importing legacy runs into {REPO_ROOT / 'runs'}/")
    n_ok = 0
    for family, by_step in LEGACY_TABLE.items():
        for steps, pt_name in by_step.items():
            cfg_name = f"{family}_lead{_step_to_lead_str(steps)}"
            if import_one(pt_name, cfg_name):
                n_ok += 1
    print(f"Done. {n_ok} runs imported.")


if __name__ == "__main__":
    main()
