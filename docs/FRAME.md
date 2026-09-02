# FRAME — LatentSky

Fields marked `[to confirm]` are drafts to validate or replace. Nothing enters `results/`
that is not tied to an R-number below.

## Thesis (one sentence)

[to confirm] Precipitation predictability at km scale is set by the size of the
precipitating system: a cluster-based, scale-decomposed evaluation defines a predictability
horizon T(L) that separates detection from area and intensity errors, and shows where a
diffusion model gains over a deterministic one.

## Part 1 — Predictability horizon and model class (core, direct comparison)

Setting: LatentSky trained on DYAMOND, two model classes on the same data, same resolution,
same lead times. Null: persistence.

Questions
1. How is the predictability horizon T(L) defined and measured from cluster detection
   (POD, FAR) as a function of system scale L and lead time?  [to confirm]
2. Which errors does the metric separate — detection, area, intensity — and are area and
   intensity errors compensating in aggregate scores?  [to confirm]
3. Does a diffusion model extend T(L) relative to the deterministic (regression) model,
   and at which scales?  [to confirm]

| R  | Claim (with a number)                                                                                 | Figure | Status |
|----|--------------------------------------------------------------------------------------------------------|--------|--------|
| R1 | Definition: T(L) from POD decay; T(L) ∝ L^α with α = [..] for persistence, the null every model must beat | fig_R1 | draft |
| R2 | Decomposition: aggregate error hides compensating area (+[..]%) and intensity (−[..]%) errors           | fig_R2 | draft |
| R3 | Deterministic vs diffusion: diffusion extends T(L) by [..] h for L > [..] km; below [..] km neither beats persistence | fig_R3 | draft |

## Part 2 — Outlook: the framework applied to WeatherBench2 models

Deliberately shallow. One figure showing the metric applied to global models
(GraphCast, GenCast, FuXi, …) on their native grid and lead times. Not a comparison with
LatentSky — different data, resolution, leads — and the paper says so in one sentence.
Purpose: show the framework is general and point to what a follow-up would ask.

| R  | Claim (with a number)                                                                        | Figure | Status |
|----|-----------------------------------------------------------------------------------------------|--------|--------|
| I1 | Illustration: on WB2 models the same decomposition shows small systems (L < [..] km) under-detected and intensity underestimated | fig_I1 | draft |

`I` = illustration: registered in REGISTRY.md with its script and data, exempt from the
sensitivity requirement, phrased as an observation, not a finding.

Status values (R only): `draft` → `tested` (experiment done) → `robust` (sensitivity passed) → `registered`.

## Out of scope for this paper

- Ranking LatentSky against WB2 models (different data, resolution, leads).
- Precipitation as an input channel to LatentSky (persistence leakage; separate study).
- Sub-3 h lead times.
- Architecture changes or retraining beyond what R1–R3 require.
- Case studies of individual events.

## Definition of done

- R1–R3 `registered`, each with ≥ 1 passed sensitivity test; I1 registered.
- `make results` regenerates every figure from `src/` + `data/`.
- `_archive/` deleted; README = this frame + install instructions.
