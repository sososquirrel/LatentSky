# FRAME — LatentSky

Fields marked `[to confirm]` are drafts to validate or replace. Nothing enters `results/`
that is not tied to an R-number below.

## Thesis (one sentence)

Precipitation predictability at km scale is set by the size of the
precipitating system: a cluster-based, scale-decomposed evaluation defines a predictability
horizon T(L) that separates detection from area and intensity errors, and shows where a
diffusion model gains over a deterministic one.

## Definitions (locked 2026-09-02)

- **System scale L** = √A of a detected precipitating cluster; systems binned in
  log-spaced L bins (P = ρ·A decomposition; paired-IoU matching; log-additive errors —
  see `docs/scale_decomposition_method.md`).
- **Predictability horizon T(L)** = the POD e-folding time τ(L) from the fit
  POD(t; L) = POD₀(L) · exp(−t/τ(L)) per scale bin, with fit standard errors reported.
- **Lead times**: all six trained leads (0.25, 0.5, 1, 2, 3, 6 h) enter every fit.
  Claims may be stated at any lead; two leads cannot constrain an exponential.
- **Canonical model pair (R3)** (revised 2026-09-02, superseding the one-head choice
  earlier the same day): deterministic = patch-split **two-head** model
  (`patchsplit_twohead_olr_fss_lead*`: wet/dry BCE head over the full field + masked-MSE
  intensity head, OLR-FiLM bottleneck conditioning, FSS multi-scale loss weight 0.25);
  diffusion = the conditional DDPM **warm-started from that same two-head forecast**.
  R3 then reads: the same deterministic forecast, with and without generative refinement
  on top. (Generative side superseded the same evening — see the canonical generative
  representative bullet below: the rectified flow replaces the SDEdit DDPM.) The one-head pair (plain UNet2D +
  one-head warm start) is the architecture sensitivity test, isolating what the two-head
  additions contribute.
  *Split note (2026-09-02, corrected same day)*: the diffusion manifests say
  `split_mode: piece`, but `train_diffusion.py` ignores that field and always builds
  the patch split (same seed 1919810, same 70/15/15) — both canonical families are
  trained on the *identical* leakage-safe patch split. No retraining needed;
  the manifest field is misleading, not the training.
- **Inputs (all models)**: 10 channels at 15-min resolution — `precipw`, `t2m`,
  `olrtoa`, `mse_500hPa`, `mse_850hPa`, `uzonal_500hPa`, `uzonal_850hPa`,
  `umeridional_500hPa`, `umeridional_850hPa`, `rain_log1p`; target `rain_total`
  at the given lead. The current-rain channel `rain_log1p` is *deliberate*: the
  model receives the persistence forecast as an input, so the null is nested —
  any skill above persistence is information the model added, not information it
  lacked. (Empirically this does not make beating persistence automatic in every
  metric: see the IoU-matching pitfall in CLAUDE.md.)
- **Null** = persistence, evaluated under *identical* object detection and matching.
- **Ensemble scoring rule (locked 2026-09-02)**: every diffusion metric is computed
  per draw and then averaged over draws — the mean of the error, never the error of
  the mean. The n-draw *mean field* is a conditional-mean estimate: scoring it turns
  the generative model back into a regression and erases exactly the difference R3
  measures. The mean field may appear only as an explicitly labeled illustration.
  *Caveat on existing numbers*: the July evaluation
  (`predict_piece_diffusion`, ensemble_size = 3) averaged draws before scoring, so all
  legacy diffusion CSVs — including the R3 draft numbers — are error-of-the-mean and
  must be re-derived per-draw.
- **Canonical generative representative (adopted 2026-09-02 evening,
  `experiments/flow_matching_6h`)**: the conditional rectified flow — trained to
  transport (two-head forecast + σ·ε) → truth in log-rain space, per-lead σ = the
  measured wet-masked forecast error rms, sampled with 32 Heun ODE steps, single
  draws scored per-draw. Verdict at 6 h: band ratio 1.115, wet +1.5 % vs truth —
  clears the SDEdit ceiling (0.80–0.85, −20 % area); the DDPM's 6 h limit was its
  sampler's train/test mismatch, not an informational limit. R3 compares two-head vs
  flow; the SDEdit DDPM with the t*(lead) table below becomes a sampling-protocol
  sensitivity.
- **SDEdit/DDPM sampling protocol (measured 2026-09-02;
  `experiments/tstart_protocol` + `sensitivity_tests/tstart_protocol_piece20`)**:
  warm start from the two-head forecast; single draws, scored per-draw;
  lead-dependent noise budget from the measured band-ratio crossings —
  t_start = 200 (0.25 h), 250 (1 h), 550 (3 h), 500 (6 h, the saturation plateau);
  0.5 h and 2 h log-interpolated (≈225, ≈400), to be verified in the R3
  re-derivation. Known limit, reported as a property of the model class:
  at 6 h the model under-disperses — small-scale power ratio ceiling ≈0.8–0.85
  and wet area −20 % regardless of budget (full-mosaic sampling does not lift it).

## Part 1 — Predictability horizon and model class (core, direct comparison)

Setting: LatentSky trained on DYAMOND, two model classes on the same data, same resolution,
same lead times. Null: persistence.

Questions
1. How does the predictability horizon T(L) = τ(L), measured from POD decay, grow with
   system scale L, and does it follow a power law?
2. Which errors does the metric separate — detection, area, intensity — and are area and
   intensity errors compensating in aggregate scores?
3. Does a generative model extend T(L) relative to the deterministic model, and at
   which scales? R3 distinguishes **placement** from **representation**: the horizon
   T(L) measures placement and is expected invariant across model classes (below the
   ≈100 km crossover placement is a predictability limit that binds persistence too);
   the ρ·A decomposition, size/intensity distributions and the spectral decomposition
   measure representation, where the generative model is expected to restore what the
   conditional mean destroys. "Capturing small systems" is claimed only in the
   representation sense; the placement sense is claimed only probabilistically
   (any-draw detection, supplementary). Pre-registered expectations:
   journal/2026-09-03.

| R  | Claim (with a number)                                                                                 | Figure | Status |
|----|--------------------------------------------------------------------------------------------------------|--------|--------|
| R1 | Definition: T(L) = POD e-folding time; it rises from ≈0.7 h (L < 65 km) to ≈8.8 h (L > 200 km); T(L) ∝ L^α with draft α = 1.30 for persistence (two-head: 1.70; committed fit form `metrics.fits.fit_exp_decay`, one block × one day, `docs/make_scaling_assets.py` — crossover ≈ 100 km; small-L decay has a slow non-exponential tail to address), the null every model must beat | fig_R1 | draft |
| R2 | Decomposition: aggregate error hides compensating area (+4 %) and intensity (−9 %) errors (net −5 %; L ≈ 200 km, 0.25 h lead, diffusion) | fig_R2 | draft |
| R3 | Deterministic vs generative (flow): the flow lifts τ(L) above both the two-head and persistence at every scale (α 1.66 vs 1.70; first model to beat the null everywhere — τ/α still one block × one day, `docs/make_flow_decomp_scaling.py`). At +3 h over the **full test record** (132 blocks × 40 days, 0.76 M two-head / 1.25 M flow / 1.30 M persistence systems, per-draw; `docs/make_flow_alltest_figs.py`): flow POD beats the two-head at every scale (0.79 vs 0.72 at L ≈ 193 km, FAR 0.17 vs 0.08); small-L (< 65 km) POD nearly triples (0.078 vs 0.028) at FAR 0.92 vs 0.64; band barycenters collapse — small (+0.53, −1.05) → (+0.21, −0.24), mid (+0.14, −0.75) → (−0.15, −0.15), large (+0.20, −0.39) → (−0.14, −0.04) — every one-day draft within ~0.1 of full record | fig_R3 | draft |

Draft-number provenance (legacy pipeline, to re-derive in `experiments/` before any
status moves past `draft`): R1 e-folding times from `_archive/../docs/predictability_horizon.csv`
(band fits; which model/null produced them must be re-established); R2 from
`docs/scale_summary_diffusion.csv` (dlnA_w = +0.043, dlnrho_w = −0.094, dlnP_w = −0.051 at
scale_mid = 203 km, lead 0.25 h); R3 from `docs/dln_tau_fits_diffusion_dln.csv` vs
`docs/dln_tau_fits_two_head_dln.csv` (τ_ρ per scale bin) — this legacy pairing (two-head
deterministic, diffusion warm-started from it) matches the canonical pair in Definitions.
The persistence-wins-object-detection caveat at short leads (loose IoU matching favors
long-lived storms) is documented in the 2026-07 baseline comparison and must be addressed
in R1's definition of T(L).

## Part 2 — Outlook: the framework applied to WeatherBench2 models

Deliberately shallow. One figure showing the metric applied to global models
(GraphCast, GenCast, FuXi, …) on their native grid and lead times. Not a comparison with
LatentSky — different data, resolution, leads — and the paper says so in one sentence.
Purpose: show the framework is general and point to what a follow-up would ask.

| R  | Claim (with a number)                                                                        | Figure | Status |
|----|-----------------------------------------------------------------------------------------------|--------|--------|
| I1 | Illustration: on WB2 AI forecasts at +6 h (GraphCast/FuXi 2020-08-15, FGN 2022-08-15; ERA5 truth, tropics, identical pipeline) the same scale story holds — POD rises monotonically with L for every model, small systems (L < 50 km) sit at the persistence floor (POD 0.02–0.06) for all three, models separate from the null only above ≈50 km (GraphCast 0.85, FuXi 0.70 near 450 km); FGN (ensemble-mean field — the conditional-mean caveat applies) tracks its null. GenCast has no +6 h forecast. `wb2_benchmark/i1_pod6h_august.py` | fig_I1 | draft |

`I` = illustration: registered in REGISTRY.md with its script and data, exempt from the
sensitivity requirement, phrased as an observation, not a finding.

Status values (R only): `draft` → `tested` (experiment done) → `robust` (sensitivity passed) → `registered`.

## Out of scope for this paper

- Ranking LatentSky against WB2 models (different data, resolution, leads).
- The rain-input ablation (9var vs 10var: how much skill comes from the `rain_log1p`
  channel; separate study — the channel itself is a deliberate design choice, see
  Definitions).
- Architecture changes or retraining beyond what R1–R3 require.
- Case studies of individual events.

## Definition of done

- R1–R3 `registered`, each with ≥ 1 passed sensitivity test; I1 registered.
- `make results` regenerates every figure from `src/` + `data/`.
- `_archive/` deleted; README = this frame + install instructions.
