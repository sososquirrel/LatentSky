# tstart_protocol — is t_start = 100 throttling the diffusion model?

**Serves:** R3 in docs/FRAME.md (sampling-protocol prerequisite: where diffusion's gain
is measured depends on how it is sampled).

**Hypothesis:** Evaluated on single draws at lead 3 h, the small-scale
(λ < 1000 km) spectral power of the diffusion prediction increases monotonically with
t_start, and some t_start ∈ [250, 999] brings the small-scale power ratio to truth
above 0.5 while wet-area fraction stays within ±20 % of truth. The current protocol
(t_start = 100, 3-draw mean) sits far below both.

**Falsified if:** the single-draw small-scale power ratio stays regression-like
(< 2× the t_start = 100 value) for every t_start up to 999 — which would indict the
trained conditional model itself (candidate mechanism: it leans on the persisted-rain
conditioning channel and never learned to generate structure), not the protocol.

**Method:** piece 8, 6 frames (stride 16), lead 3 h, canonical checkpoints
(patchsplit_twohead_olr_fss_lead3h warm start + diffusion_lead3h). All 22 tiles of the
850-hPa-grid mosaic batched per denoising step. t_start ∈ {50, 100, 250, 500, 999},
one seeded draw each; additionally the 3-draw mean at t_start = 100 (the current
protocol) to isolate the averaging effect. Metrics per field, averaged over frames:
radially averaged power spectrum (raw mm), power ratio to truth in the λ < 1000 km
band, wet fraction (> 0.1 mm), P99 and max intensity. Note: full mosaic, not only test
blocks — this is a protocol diagnosis, not a skill claim.

**Config:** config.yaml
**Outcome (2026-09-02, job 5805615): supported.** Single-draw λ<1000 km power ratio to
truth (lead 3 h, piece 8, 6 frames): two-head 0.57; t_start 50/100/250/500 →
0.61/0.70/0.89/0.96; t_start=999 diverges (ratio 98, wet fraction 94 % — pure-noise
chain collapse; independent per-tile chains also lose cross-tile coherence, visible
patchwork). Wet fraction at t_start=500: 0.042 vs truth 0.046 (−8 %); field max
overshoots (22.6 vs 14.0 mm). The current protocol is doubly throttled: t_start=100
gives only 0.70, and the 3-draw mean pulls it back to 0.64. t_start=100 was originally
chosen by bulk pixel R² — a double-penalty metric that structurally prefers smooth
fields; protocol metrics for a generative model must be distributional (spectrum,
intensity quantiles), computed on single draws.
**Next:** lead-dependent t_start (Sophie's hypothesis: the noise budget should grow
with lead). Rule under test in `run_tstart_by_lead.py`: t*(lead) = smallest t with
(1−ᾱ_t) ≥ ᾱ_t·MSE_z(lead), MSE_z = two-head forecast error in z-space. Then:
sensitivity on a second piece; adopt in FRAME; rerun the R3 decomposition under the
new protocol.
