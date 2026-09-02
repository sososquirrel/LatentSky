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
- **Canonical model pair (R3)**: deterministic = patch-split one-head regression;
  diffusion = the diffusion model on the same one-head backbone, same split.
  The two-head model is not canonical; comparing against it is a sensitivity test.
- **Null** = persistence, evaluated under *identical* object detection and matching.

## Part 1 — Predictability horizon and model class (core, direct comparison)

Setting: LatentSky trained on DYAMOND, two model classes on the same data, same resolution,
same lead times. Null: persistence.

Questions
1. How does the predictability horizon T(L) = τ(L), measured from POD decay, grow with
   system scale L, and does it follow a power law?
2. Which errors does the metric separate — detection, area, intensity — and are area and
   intensity errors compensating in aggregate scores?
3. Does a diffusion model extend T(L) relative to the deterministic (regression) model,
   and at which scales?

| R  | Claim (with a number)                                                                                 | Figure | Status |
|----|--------------------------------------------------------------------------------------------------------|--------|--------|
| R1 | Definition: T(L) = POD e-folding time; it rises from ≈0.7 h (L < 65 km) to ≈8.8 h (L > 200 km); T(L) ∝ L^α with α = [to confirm — power-law fit not yet run] for persistence, the null every model must beat | fig_R1 | draft |
| R2 | Decomposition: aggregate error hides compensating area (+4 %) and intensity (−9 %) errors (net −5 %; L ≈ 200 km, 0.25 h lead, diffusion) | fig_R2 | draft |
| R3 | Deterministic vs diffusion: diffusion extends the intensity e-folding time τ_ρ by ≈2 h for 100 ≤ L ≤ 700 km and ≈5 h at L ≈ 60 km (9.3 h vs 4.3 h); below ≈45 km neither beats persistence [threshold to confirm] | fig_R3 | draft |

Draft-number provenance (legacy pipeline, to re-derive in `experiments/` before any
status moves past `draft`): R1 e-folding times from `_archive/../docs/predictability_horizon.csv`
(band fits; which model/null produced them must be re-established); R2 from
`docs/scale_summary_diffusion.csv` (dlnA_w = +0.043, dlnrho_w = −0.094, dlnP_w = −0.051 at
scale_mid = 203 km, lead 0.25 h); R3 from `docs/dln_tau_fits_diffusion_dln.csv` vs
`docs/dln_tau_fits_two_head_dln.csv` (τ_ρ per scale bin) — note this legacy pair uses the
*two-head* deterministic model; the canonical pair (one-head + one-head-backbone diffusion,
see Definitions) must replace it.
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
| I1 | Illustration: on WB2 models the same decomposition shows small systems (L < [to confirm — read off `_archive/../docs/scale_summary_<model>_*_24h.csv`] km) under-detected and intensity underestimated | fig_I1 | draft |

`I` = illustration: registered in REGISTRY.md with its script and data, exempt from the
sensitivity requirement, phrased as an observation, not a finding.

Status values (R only): `draft` → `tested` (experiment done) → `robust` (sensitivity passed) → `registered`.

## Out of scope for this paper

- Ranking LatentSky against WB2 models (different data, resolution, leads).
- Precipitation as an input channel to LatentSky (persistence leakage; separate study).
- Architecture changes or retraining beyond what R1–R3 require.
- Case studies of individual events.

## Definition of done

- R1–R3 `registered`, each with ≥ 1 passed sensitivity test; I1 registered.
- `make results` regenerates every figure from `src/` + `data/`.
- `_archive/` deleted; README = this frame + install instructions.
