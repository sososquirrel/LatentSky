"""Spatial-field comparison for ONE example (piece/block/frame): what does
each predictor in the bar-chart comparison (docs/figures/
diffusion_refine_vs_baselines_lead3h.png) actually produce, side by side.

Baselines (persistence/climatology/random) use the exact same definitions
as scripts/eval_baselines_lead3h_newsplits.py's patch-mode:
  - persistence: rain_total at the SAME time index as the conditioning X
    (before the 3h lead shift), same location.
  - climatology: per-pixel mean of rain_total over all 40 pieces, all
    frames -- computed here restricted to this one patch for speed.
  - random: rain_total from an unrelated, decorrelated day (piece+20 mod 40),
    same location/time-of-day -- a real, structured field, just from the
    wrong day, not literal noise.

Sample picked (piece 8, block (1,4), frame 0) was one of the rainier
examples seen during the full evaluation run (truth mean ~0.12mm), so the
fields have visible structure rather than being almost entirely dry.
"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch

from latentsky.data import load_piece_arrays
from latentsky.diffusion import GaussianDiffusion
from latentsky.inference import _load_model
from latentsky.models import ConditionalDiffusionUNet
from latentsky.registry import load_run

REPO = pathlib.Path(__file__).resolve().parents[1]
N_TOTAL = 40
PID, ROW, COL, FI = 8, 1, 4, 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)

diff_run = load_run("diffusion_lead3h")
dcfg = diff_run.config
P = dcfg.patch_size
i, j = ROW * P, COL * P

diff_model = ConditionalDiffusionUNet(
    cond_ch=dcfg.in_channels, base_ch=dcfg.base_ch, depth=dcfg.depth,
    dropout_p=dcfg.dropout_p, time_emb_dim=dcfg.diffusion_time_emb_dim,
)
diff_model.load_state_dict(torch.load(diff_run.checkpoint_path, map_location=device))
diff_model = diff_model.to(device).eval()
diffusion = GaussianDiffusion(dcfg.diffusion_timesteps, dcfg.diffusion_schedule, device=device)
scale = dcfg.diffusion_target_scale

reg_run = load_run("patchsplit_twohead_olr_fss_lead3h")
reg_model = _load_model(reg_run, device)

orig_run = load_run("mseuv_rain_resid_lead3h")
orig_model = _load_model(orig_run, device)

# --- truth / persistence / cond for piece 8 ---
X, Y, persisted = load_piece_arrays(
    dcfg.pieces_dir, PID, dcfg.input_vars, dcfg.target_prefix, dcfg.delta_index,
    return_persisted=True,
)
truth = Y[FI, 0, i:i + P, j:j + P]
persistence = persisted[FI, 0, i:i + P, j:j + P]
cond = torch.from_numpy(X[FI:FI + 1, :, i:i + P, j:j + P]).to(device)

# --- climatology: per-pixel mean over all 40 pieces, restricted to this patch ---
clim_sum, clim_n = None, 0
for pid in range(N_TOTAL):
    Yp = np.load(f"{dcfg.pieces_dir}/{dcfg.target_prefix}_piece{pid:02d}.npy", mmap_mode="r")
    patch = np.asarray(Yp[:, i:i + P, j:j + P], dtype=np.float32)
    clim_sum = patch.sum(axis=0) if clim_sum is None else clim_sum + patch.sum(axis=0)
    clim_n += patch.shape[0]
climatology = clim_sum / clim_n

# --- random: same location/time-of-day, decorrelated day (piece+20 mod 40) ---
rand_pid = (PID + 20) % N_TOTAL
Yr = np.load(f"{dcfg.pieces_dir}/{dcfg.target_prefix}_piece{rand_pid:02d}.npy", mmap_mode="r")
random_field = np.asarray(Yr[FI, i:i + P, j:j + P], dtype=np.float32)

# --- regression (patch-split two-head) ---
with torch.no_grad():
    out = reg_model(cond)
    reg_rain = torch.clamp(out[0] if isinstance(out, tuple) else out, min=0.0)
regression = reg_rain.cpu().numpy()[0, 0]

# --- original LatentSky (residual target -> add persisted, clip>=0) ---
with torch.no_grad():
    orig_out = orig_model(cond)
    orig_resid = orig_out[0] if isinstance(orig_out, tuple) else orig_out
orig_pred = np.clip(orig_resid.cpu().numpy()[0, 0] + persistence, a_min=0.0, a_max=None)

# --- diffusion refine at t_start = 50, 150, 400, warm-started from `regression` ---
x0_init = torch.log1p(reg_rain) / scale
refine = {}
for t_start in (50, 150, 400):
    draws = []
    for _ in range(3):
        x0 = diffusion.ddpm_refine_sample(diff_model, cond, x0_init, t_start)
        draws.append(torch.expm1(x0 * scale))
    refine[t_start] = torch.clamp(torch.stack(draws).mean(dim=0), min=0.0).cpu().numpy()[0, 0]

panels = [
    ("Truth", truth),
    ("Persistence", persistence),
    ("Climatology", climatology),
    ("Random\n(decorrelated day)", random_field),
    ("LatentSky\n(original)", orig_pred),
    ("Regression\n(patch-split two-head)", regression),
    ("Diffusion refine\nt_start=50", refine[50]),
    ("Diffusion refine\nt_start=150", refine[150]),
    ("Diffusion refine\nt_start=400", refine[400]),
]

# Rain is extremely heavy-tailed -- a single ~19mm pixel in `truth` would
# otherwise force vmax so high every other pixel looks blank white. Scale
# off the 99th percentile of truth instead, and clip/extend the colorbar so
# the rare higher pixels are still visibly flagged, not silently hidden.
vmax = np.percentile(truth, 99.5)

fig, axes = plt.subplots(3, 3, figsize=(11, 11))
for ax, (title, field) in zip(axes.flat, panels):
    im = ax.imshow(field, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nmean={field.mean():.4f}  max={field.max():.3f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.6, extend="max", label="rain (mm / 15min, lead=3h)")
fig.suptitle(f"Example: piece {PID}, block ({ROW},{COL}), frame {FI}  (patch-split held-out test block)",
             fontsize=12)

out = REPO / "docs/figures/example_prediction_grid_lead3h.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# --- same grid, but (pred - truth) for every panel except Truth itself ---
diff_vmax = np.percentile(np.abs(np.stack([f for _, f in panels[1:]]) - truth[None]), 99)

fig2, axes2 = plt.subplots(3, 3, figsize=(11, 11))
for ax, (title, field) in zip(axes2.flat, panels):
    if title == "Truth":
        ax.imshow(field, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"{title}\n(reference, not a diff -- mean={field.mean():.4f})", fontsize=9)
    else:
        diff = field - truth
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
        ax.set_title(f"{title}\nmean diff={diff.mean():+.4f}  RMSE={np.sqrt((diff**2).mean()):.4f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig2.colorbar(im, ax=[a for a in axes2.flat if a is not axes2.flat[0]], shrink=0.6,
              extend="both", label="prediction - truth (mm)")
fig2.suptitle(f"Prediction error vs. truth: piece {PID}, block ({ROW},{COL}), frame {FI}\n"
              "blue = underprediction, red = overprediction", fontsize=12)

out2 = REPO / "docs/figures/example_prediction_diff_grid_lead3h.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"wrote {out2}")
