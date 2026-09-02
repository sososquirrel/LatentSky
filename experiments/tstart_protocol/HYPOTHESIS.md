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
**Part 2 — lead-dependent t_start (jobs 5805622/5805627, `run_tstart_by_lead.py`).**
Sophie's hypothesis (noise budget grows with lead): *supported in shape, analytic rule
too weak in magnitude, and a hard ceiling found at 6 h.*
- Full-field MSE_z rule failed (dry-pixel dilution → t* = 5–25, no gain at
  verification). Wet-masked MSE_z (mask truth>0.1 ∪ det>0.1) grows monotonically with
  lead: 0.010/0.014/0.021/0.029/0.035/0.043-ish → rule t* = 56/67/84/100/110/124.
  Monotone ✓, but ~2–5× below the empirical need, worse at long leads.
- Empirical band-ratio-to-truth vs t_start (single draws, piece 8):
  lead 0.25 h crosses 1 near t≈150 (0.97@100, 1.03@250); 1 h near t≈230
  (0.90@100, 1.02@250); 3 h near t≈550 (0.89@250, 0.96@500); **6 h never crosses —
  saturates at 0.80 (0.80@500 = 0.80@750)** with a persistent wet-area deficit
  (0.038 vs truth 0.047, −20 %).
- Stability ceiling: beyond t≈500 the wet fraction inflates at short leads
  (1 h: 0.056@500 → 0.21@750) — the one-sided clip drift.
- **Interpretation:** protocol fixed for ≤3 h (use the empirical t*(lead) table);
  at 6 h the model itself under-disperses — even with full generative freedom it
  restores only 80 % of truth's small-scale power and cannot create the missing
  rainy area. A perfect conditional generative model should restore ~100 % of
  small-scale variance distributionally even when placement is unpredictable, so
  this saturation is a real R3-relevant property, not a sampling artifact —
  unless it is the per-tile chain decoherence capping it (testable: full-mosaic
  single-chain sampling).
**Next:** (1) sensitivity: repeat empirical crossings on a second piece;
(2) full-mosaic sampling test at 6 h to separate model limit from tile decoherence;
(3) adopt t*(lead) table in FRAME after (1)–(2); (4) re-derive the R3 decomposition
per-draw (FRAME ensemble scoring rule) under the adopted protocol.
