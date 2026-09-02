# flow_matching_6h — does a learned forecast→truth transport clear the SDEdit ceiling?

**Serves:** R3 (the diffusion/generative representative must not be handicapped by its
sampler; follows `experiments/tstart_protocol`, which found a 6 h ceiling).

**Hypothesis:** A conditional rectified flow trained to transport
(z-forecast + σ·ε) → z-truth, conditioned on the 10 met channels, with
σ = 0.204 (the measured 6 h wet-masked forecast error rms in z), restores
single-draw small-scale power at lead 6 h beyond the SDEdit ceiling:
band ratio ≥ 0.90 (SDEdit: 0.80–0.85 tiled, 0.76 full-mosaic) with wet fraction
within 10 % of truth (SDEdit: −20 %).

**Falsified if:** the flow also plateaus at ≤ 0.85 or keeps the ≥15 % area deficit —
then the 6 h limit is informational (the conditioning does not determine where the
missing systems are, and no transport can invent them), and SDEdit + t*(lead)
remains the canonical protocol with confidence.

**Method:** same backbone (ConditionalDiffusionUNet as velocity net), same patch
split (seed 1919810), same data and lead-6 h pairing as the canonical runs; the
frozen two-head lead-6 h forecast computed on the fly per batch; velocity-MSE loss;
AdamW 2e-4, StepLR(8, 0.5), batch 64, early stop patience 8. Post-training in the
same job: single draws (Heun, 32 steps) on piece-8 frames [8,24,40,56], band ratio
and wet fraction against the recorded SDEdit values. Model code:
`src/latentsky/models/flow.py` (tested).

**Config:** config.yaml
**Outcome (2026-09-02, job 5805787): SUPPORTED, decisively.** Early stop at epoch ~24
(best val velocity-MSE 0.0047). Single draws, 32 Heun steps, piece-8 frames
[8,24,40,56]: **band ratio 1.115** (per frame 1.09–1.16) and **wet fraction 0.0477 vs
truth 0.0470 (+1.5 %)** — against the SDEdit ceiling of 0.80–0.85 with −20 % wet area.
The 6 h limit was the SDEdit train/test mismatch, not an informational limit: the
natively-trained transport restores both the small-scale variance and the missing rainy
area. Mild small-scale over-dispersion (+11 %) noted — candidate sensitivity: σ ×½/×2
and step count; do not tune before the R3 re-derivation baseline exists.
**Next:** remaining five leads already training (jobs 5809905–09, per-lead σ);
flow adopted as R3's canonical generative representative (FRAME.md updated); SDEdit +
t*(lead) demoted to sampling-protocol sensitivity. Then: metrics rebuild and the
per-draw R1–R3 re-derivation.
