"""Experiment config schema. One YAML in configs/ -> one ExperimentConfig."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str
    input_vars: List[str]
    lead_steps_15: int
    target_prefix: str = "rain_total"
    residual: bool = False  # predict target - persisted(t=0) instead of target

    # model
    base_ch: int = 64
    depth: int = 4
    dropout_p: float = 0.05
    out_activation: str = "softplus"

    # training
    patch_size: int = 64
    patch_stride: Optional[int] = None  # None -> non-overlapping (== patch_size)
    batch_size: int = 256
    num_epochs: int = 155
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_step: int = 8
    lr_gamma: float = 0.5
    seed: int = 1919810
    train_pieces_per_group: int = 4

    # two-head (wet/dry classification + intensity regression) + OLR-FiLM
    # conditioning + multi-scale/FSS loss. See latentsky/losses.py.
    two_head: bool = False
    olr_var_name: str = "olrtoa"
    bce_weight: float = 1.0
    fss_weight: float = 0.0
    fss_threshold: float = 0.1
    fss_sharpness: float = 20.0
    fss_max_kernel_px: int = 33

    # Pretrain-then-fine-tune (see latentsky/masking.py). Pretraining: same
    # architecture/loss, lead=0, but `pretrain_mask_channel` (e.g.
    # "rain_log1p") is randomly block-masked per batch so the model can't
    # shortcut same-timestep rain from a persisted-rain input -- forces it
    # to learn rain from the other physical variables instead. None -> no
    # masking (default, i.e. every existing config is unaffected).
    pretrain_mask_channel: Optional[str] = None
    pretrain_mask_frac: float = 0.4
    pretrain_mask_block: int = 16
    # Fine-tuning: path to a checkpoint.pt whose weights initialize the
    # model before training starts (instead of random init). None -> no
    # change from today's from-scratch training.
    pretrain_checkpoint: Optional[str] = None

    # data
    pieces_dir: str = "/glade/derecho/scratch/tzhou/splitted_1440x720_9vars_40pieces_v2"
    n_total_pieces: int = 40
    test_piece_ids: List[int] = field(default_factory=lambda: [2, 12, 22, 32])
    val_piece_ids: List[int] = field(default_factory=lambda: [6, 16, 26, 36])
    # Pieces excluded from train but not used for val/test either -- closes
    # the temporal-adjacency leak between train and held-out pieces (see
    # scripts/make_split.py). Empty by default so existing configs are
    # unaffected.
    buffer_piece_ids: List[int] = field(default_factory=list)

    # Conditional DDPM (see latentsky/diffusion.py, models.ConditionalDiffusionUNet,
    # train_diffusion.py). Unrelated to two_head -- a separate training script.
    diffusion_timesteps: int = 1000
    diffusion_schedule: str = "cosine"
    diffusion_time_emb_dim: int = 128
    diffusion_target_scale: float = 2.0  # x0 = log1p(rain) / scale, roughly unit range

    # Patch-level (not day-level) split -- see latentsky/patch_split.py.
    # "piece" (default) is today's whole-day split above. "patch" instead
    # randomly assigns individual non-overlapping spatial patches to
    # train/val/test across ALL pieces, so every day contributes to every
    # split (just at different spatial locations) -- requires patch_stride
    # to be None/==patch_size (non-overlapping), enforced in train.py.
    split_mode: str = "piece"
    patch_split_seed: int = 1919810
    patch_split_train_frac: float = 0.7
    patch_split_val_frac: float = 0.15

    # bookkeeping
    notes: str = ""

    @property
    def lead_hours(self) -> float:
        return self.lead_steps_15 * 15 / 60.0

    @property
    def delta_index(self) -> int:
        # Off-by-one between X[i] and Y[i+1] in raw data, plus the requested lead.
        return self.lead_steps_15 + 1

    @property
    def in_channels(self) -> int:
        return len(self.input_vars)

    @property
    def train_piece_ids(self) -> List[int]:
        held = set(self.test_piece_ids + self.val_piece_ids + self.buffer_piece_ids)
        return [p for p in range(self.n_total_pieces) if p not in held]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        if "name" not in data:
            data["name"] = path.stem
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
