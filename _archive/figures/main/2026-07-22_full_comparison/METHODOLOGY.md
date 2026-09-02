# Methodology — cross-model comparison (2026-07-22)

Five model tracks, three baselines, all evaluated on the same
object-based scale-decomposition pipeline. This document specifies
exactly how each was trained and evaluated, so the comparison figures in
this folder can be read without guessing.

## The five model tracks

| # | Track | Run family | Leads | Split | Architecture | Loss |
|---|---|---|---|---|---|---|
| 1 | Regression, old split | `mseuv_lead*` | 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6h | day-level (leaky) | one-head U-Net | masked MSE |
| 2 | Regression, new split | `patchsplit_onehead_lead*` | 0.25, 0.5, 1, 2, 3, 6h | patch-level | one-head U-Net | masked MSE |
| 3 | Regression + two-head | `patchsplit_twohead_olr_fss_lead*` | 0.25, 0.5, 1, 2, 3, 6h | patch-level | two-head U-Net + OLR-FiLM | MSE + BCE + FSS |
| 4 | Regression + two-head + diffusion | `diffusion_lead*` (warm-started from track 3) | 0.25, 0.5, 1, 2, 3, 6h | patch-level | conditional DDPM U-Net | epsilon-MSE |
| 5 | Baselines | persistence / climatology / random | 0.25, 0.5, 1, 2, 3, 6h | n/a | none (no learned params) | n/a |

Track 2 did not exist before today — every previously-trained patch-split
run was two-head. It was trained today (`configs/patchsplit_onehead_lead*.yaml`,
`submit/train_patchsplit_onehead_lead*.pbs`) with **identical** input
variables, architecture hyperparameters, patch size, batch size, epoch
budget, and patch-split settings as track 3's configs — the only
difference is `two_head: false` (plain masked MSE, no OLR-FiLM, no
BCE/FSS terms). This isolates what the two-head/OLR-FiLM/FSS additions
buy you, holding the split and everything else fixed. (Track 1 vs track 3
is *not* a clean architecture ablation by itself — it differs in split,
input variables, and architecture all at once; track 2 is the control
that separates those factors.)

## 1. How the training split is done

Two incompatible conventions exist in this codebase. Which one a run
uses is set by `ExperimentConfig.split_mode` (`"piece"` default, or
`"patch"`).

### Old split — day-level (`split_mode="piece"`, tracks 1 and, historically, most one-head runs)

The 40 DYAMOND days ("pieces") are partitioned by whole day:
`test_piece_ids` (default `[2, 12, 22, 32]`), `val_piece_ids` (default
`[6, 16, 26, 36]`), everything else is train. **Known leak**: adjacent
days are highly autocorrelated (weather persists across a day boundary),
and the default split's held-out days have no buffer — their immediate
neighbors remain in train. A leak-safe variant exists (`scripts/make_split.py`,
used by `twohead_olr_fss_lead3h`/`_finetuned`, not used by any run in
this comparison): it holds out an *interior* day per 5-day block and
excludes both day-neighbors from train entirely as `buffer_piece_ids`.
Track 1 (`mseuv_lead*`) uses the plain, non-buffered default —
its numbers should be read as an *optimistic* upper bound on the old-split
architecture's real skill, not an apples-to-apples baseline against the
patch-split tracks.

### New split — patch-level (`split_mode="patch"`, tracks 2, 3, 4)

Every non-overlapping `128x128`-pixel spatial block, in every piece, is
assigned independently to train/val/test:
`make_patch_split(n_pieces=40, nrows, ncols, seed=1919810, train_frac=0.7, val_frac=0.15)`
shuffles the full list of `(piece, row, col)` blocks with a fixed seed
and slices `70% / 15% / 15%`. Two things make this leak-safe rather than
a repeat of the day-level problem:
1. Patches are non-overlapping (`patch_stride == patch_size`, enforced —
   `train.py` raises `ValueError` otherwise) — two patches on opposite
   sides of the split share zero pixels.
2. The split unit is the **whole spatial block across all time frames**,
   not per-frame — otherwise two 15-minute frames of the same location
   (highly autocorrelated) could land on opposite sides of the split,
   reintroducing the same kind of leak at 15-minute instead of 1-day
   granularity.

Residual, accepted-as-negligible risk: two *adjacent* (touching,
non-overlapping) blocks can still have one storm straddling their shared
border — smaller and different in kind from the day-adjacency leak, not
zero.

**Evaluation gotcha**: for a patch-split run, `cfg.test_piece_ids` is
just the inert day-level default — it does not describe the real test
set, which is scattered blocks within every one of the 40 pieces. Every
patch-split evaluation in this comparison scans all 40 pieces and then
applies `latentsky.patch_split.restrict_to_patch_test_blocks()`, which
keeps only systems whose truth/pred centroid falls in that piece's
actual held-out test blocks. Skipping this filter (an earlier bug this
session, since fixed) would silently evaluate mostly on *training* data.

Diffusion (track 4) trains on the same patch-split machinery
unconditionally, regardless of its config's `split_mode` field value
(a config quirk: `diffusion_lead*.yaml` reports `split_mode: piece` by
default-field-absence, but `train_diffusion.py` always calls
`make_patch_split`) — so it shares the identical `(piece,row,col)`
train/val/test assignment as track 3, since it's warm-started from
track 3's prediction at the matching lead.

## 2. How the two-head model works

`UNet2DTwoHead` (`latentsky/models.py`) is the same encoder/bottleneck/
decoder U-Net as the one-head model, with two additions:

- **OLR-FiLM conditioning at the bottleneck.** The raw OLR (outgoing
  longwave radiation) input channel is summarized per-sample as
  `[mean, min]` over the patch, passed through a small MLP
  (`Linear(2,64)→ReLU→Linear(64, 2*bottleneck_ch)`) to produce
  `(gamma, beta)`, and the bottleneck feature map is affine-modulated:
  `h = h*(1+gamma) + beta`. This is the *only* place OLR conditioning
  enters — one global signal at one scale, not injected at every decoder
  level (contrast with diffusion's timestep-FiLM, applied at every level
  — see below).
- **Two parallel 1x1-conv output heads**: `rain_head` (continuous
  intensity, same activation as the one-head model) and `wetdry_head`
  (raw logits, no activation). The model returns `(rain_pred,
  wetdry_logit)` as a tuple — **the heads do not combine** inside the
  model. `wetdry_logit` is a purely auxiliary training signal; at
  inference, `latentsky/inference.py` discards it entirely
  (`rain_pred = out[0] if isinstance(out, tuple) else out`) and only
  `rain_head`'s continuous output is used as "the" prediction. The
  classification head never gates or masks the regression output at
  eval time — it only shapes what the shared trunk learns during
  training, via the loss below.

## 3. How diffusion works here

`ConditionalDiffusionUNet` takes the noisy target field `x_t`
concatenated with the same conditioning channels used everywhere
(met variables), plus a sinusoidal embedding of the diffusion timestep
injected via FiLM **at every encoder level, the bottleneck, and every
decoder level** (unlike the two-head model's single bottleneck-only
FiLM — here it has to control denoising strength throughout the
network). Output: single-channel predicted noise. `GaussianDiffusion`
uses a cosine beta schedule (Nichol & Dhariwal 2021), 1000 timesteps,
trained with a per-batch random timestep and a target normalized as
`x0 = log1p(rain_total) / diffusion_target_scale` (scale=2.0).

**Inference here is not plain unconditional sampling — it's
warm-started refinement** (`GaussianDiffusion.ddpm_refine_sample`,
`latentsky/diffusion_inference.py`):
1. Take `x0_init` = track 3's (two-head, matching lead) prediction, in
   the same normalized `log1p/scale` space.
2. Forward-noise it to a fixed `t_start` via `q_sample` (adds Gaussian
   noise proportional to how far `t_start` is into the schedule) —
   *not* starting from pure noise at `t=1000`.
3. Run the same reverse ancestral sampling loop as full unconditional
   sampling, but only from `t_start` down to `t=0`.
4. Average `ensemble_size` independent draws, `expm1` back to rain
   units, clamp at 0.

`t_start` trades off fidelity to the two-head warm start (small
`t_start` ≈ small perturbation, safe but limited generative benefit)
against generative freedom (large `t_start` ≈ more room to diverge from
the warm start, but more risk of the reverse chain drifting — observed
in-repo: bulk R² degraded 0.39→0.31 going from `t_start=50` to `400`).
Every diffusion evaluation in this comparison uses `t_start=100`,
`ensemble_size=3`. This requires no retraining of the diffusion network
— it was trained to denoise from *any* noisy `x_t`, not specifically
ones reached from pure noise, so starting the reverse process partway
through is valid by construction.

Domain coverage note: diffusion tiles each piece into non-overlapping
`128x128` patches and runs the above independently per patch — cost is
`t_start+1` sequential forward passes **per patch per frame per
ensemble draw**, i.e. orders of magnitude more expensive than one
regression forward pass. This is why diffusion evaluation always
declares an explicit, bounded scope rather than defaulting to "sweep
everything" the way the regression evaluation does.

## 4. Loss functions, per track

| Track | Loss |
|---|---|
| 1. Regression, old split | `MSELoss` on wet pixels only (`y_target > 0` mask) |
| 2. Regression, new split | Identical: masked `MSELoss`, wet pixels only |
| 3. Two-head | `mse + bce_weight * bce + fss_weight * fss`, weights `1.0` / `0.25` in every `patchsplit_twohead_olr_fss_lead*` config: `mse` = masked MSE (same as tracks 1/2); `bce` = `BCEWithLogitsLoss(wetdry_logit, wet_mask)` over the **full** tensor (wet and dry pixels both, unlike the regression MSE term); `fss` = multiscale Fractions Skill Score term — at each of 6 kernel sizes (`[1,3,5,11,19,33]` px, derived from the project's shared km scale bins via `KM_PER_PIXEL=27.80`), MSE between a differentiable sigmoid-relaxed "wet fraction" field (prediction) and a hard-threshold wet-fraction field (truth), averaged across kernels. FSS explicitly reuses the same scale vocabulary as the evaluation pipeline (`default_scale_edges`), so training and evaluation "speak the same scale language." |
| 4. Diffusion | Epsilon-prediction MSE: `MSE(predicted_noise, actual_noise)`, whole-field, no wet/dry masking — a standard DDPM training objective, unrelated in form to the other three tracks' losses (there is no direct "loss comparability" between diffusion and the regression tracks; only the downstream evaluation metrics are comparable). |
| 5. Baselines | n/a — no learned parameters |

## 5. Scale-decomposition evaluation — what data, what pipeline

Every track (models and baselines alike) is scored with the same
object-based pipeline (`latentsky/scale_analysis.py`), not bulk
pixel RMSE:

1. **Per frame**, truth and prediction are each thresholded at
   `0.1 mm` and connected-component labeled (8-connectivity) into
   discrete precipitating *systems*.
2. **Systems are matched** truth↔prediction by greedy descending IoU;
   a pair is accepted if `IoU >= 0.1` (`iou_min`). An unmatched truth
   system is a **miss**; an unmatched predicted system is a **false
   alarm**.
3. Each system carries `A` (area, km²), `P` (area-weighted precip
   volume, mm·km²), `rho = P/A` (mean intensity, mm), `L = sqrt(A)`
   (characteristic scale, km). For matches, error decomposes exactly
   and log-additively: `dlnP = ln(P_p/P_t) = dlnA + dlnrho` (no cross
   term) — this is the "precipitation decomposition" figure.
4. Systems are binned into 8 log-spaced scale bins,
   `default_scale_edges(min_km=25, max_km=3000, n=8)` — the plotted
   `scale_mid` per bin is the **geometric-mean center**, not the edge
   (so the smallest bin's plotted point is ~34 km, not the 25 km edge;
   the actual grid pixel size is ~27.8 km, so the nominal 25 km edge is
   a rounded anchor, not an exact cell size).
5. Per (run, scale bin): `POD = n_match/(n_match+n_miss)`,
   `FAR = n_fa/(n_match+n_fa)`, plus the precip-weighted mean `dlnA`/
   `dlnrho`/`dlnP` and their area-vs-intensity fractional split.

**Which pieces/frames, per track — this is where the tracks differ,
and why:**

- **Tracks 1 (old split)**: the actual held-out test set is small by
  construction — `test_piece_ids=[2,12,22,32]`, 4 whole days. Evaluation
  sweeps exactly those 4 pieces; no bounding needed, this already *is*
  full coverage of the model's test set.
- **Tracks 2, 3 (patch-split regression)**: the real test set is
  scattered ~15%-of-domain blocks across *every* piece. Evaluation
  scans all 40 pieces (cheap — one forward pass per frame, regardless
  of architecture) then applies `restrict_to_patch_test_blocks` to keep
  only genuine held-out systems. `max_frames_per_piece=16`.
- **Track 4 (diffusion)**: same patch-split test blocks as tracks 2/3,
  but each frame costs `t_start+1` sequential forward passes per patch
  per ensemble draw. Evaluation still scans all 40 pieces with
  `max_frames_per_piece=16` (matching tracks 2/3 exactly, so POD/tau(L)
  are computed from the same amount of underlying data across all three
  patch-split tracks — an earlier bounded 3-piece/2-frame version of
  this evaluation produced visibly noisier, less trustworthy numbers at
  the largest scale bins and has been superseded), `t_start=100`,
  `ensemble_size=3`.
- **Track 5 (baselines)**: persistence/climatology/random are evaluated
  against the same patch-split test blocks as tracks 2/3/4, at the same
  6 leads, using a `predict_piece_baseline()` wrapper built today
  specifically so these baselines plug into the identical evaluation
  code path (no bespoke baseline-only script/metric).

**A note on large scales (>=~670 km)**: at every track, POD saturates
near 1.0 for the largest 1-2 scale bins. This is a real, expected
property of the metric, not a model artifact: (a) IoU>=0.1 is an easy
bar to clear for a system spanning hundreds of km — even a blurry
prediction usually has *some* rain somewhere in that huge envelope, and
(b) very few systems are that large in any sample (single digits to low
tens), so POD is a coarse fraction at that scale for every track,
including the baselines. Any per-scale-bin exponential decay fit
(`POD(t)/POD(0) = exp(-t/tau)`) that saturates at the fit's upper tau
bound (200h) is flagged, not silently reported as a real number, in the
tau(L) scaling-law figure.
