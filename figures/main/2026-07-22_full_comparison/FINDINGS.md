# Findings — cross-model comparison (2026-07-22)

Companion to METHODOLOGY.md (which documents how each track was trained/
evaluated). This is what the comparison actually shows.

## 1. Architecture ordering is clean and consistent

Across nearly every scale bin and lead (`track_compare_pod_raw.png`,
`pod_far_compare_lead*.png`), POD ranks:
**two-head+diffusion ≈ two-head > regression (new/patch split) >
regression (old split) > persistence >> climatology ≈ random**.
Each architectural addition (patch-split control, classification head +
OLR-FiLM + FSS loss, diffusion refinement) buys real, visible detection
skill. This holds at every lead from 0.25h to 6h.

## 2. Diffusion vs two-head: real signal, small margin

With full-coverage evaluation (this session's main compute cost — see
METHODOLOGY.md §5), diffusion's tau(L) power-law (β=1.11, R²=1.00) is
close to two-head's (β=1.19, R²=0.99) — not the steeper, noisier β=1.74/
R²=0.85 seen under an earlier, deliberately bounded 3-piece sample. The
earlier result was a small-sample artifact, not a property of the model —
worth remembering before trusting any diffusion metric computed on a
scoped-down evaluation.

Where diffusion visibly helps beyond two-head: **system-size
distribution** (`system_size_dist_lead*.png`, plotted as a CCDF —
P(size > L), log-log — not a KDE-smoothed PDF; the KDE version tried
first understated exactly the tail behavior that matters here, so it was
dropped). At lead=3h, checked against the raw (non-smoothed) data
directly:

| track | median L | frac. systems >500 km |
|---|---|---|
| **truth** | 38 km | **0.36%** |
| regression (old split) | 104 km | 5.5% (15x too many) |
| regression (new split) | 106 km | 6.6% (18x too many) |
| two-head | 100 km | 3.8% (10x too many) |
| two-head+diffusion | 28 km | 0.64% (1.8x too many) |

All three regression-family tracks inflate both the median predicted
size (~2.7x) and the large-system tail (10-18x too many systems above
500 km) — classic MSE-blur merging adjacent small storms into single
giant blobs.

**Caution reading the CCDF itself**: a CCDF is normalized to each
track's own total sample size (always starts near 1.0), so it can make
small-L behavior look like it "agrees" across tracks purely from that
normalization, even when absolute counts differ enormously. Checked
directly (lead=3h, systems with L<60km): truth has 96,912 (76% of its
127,501 total); the regression tracks have only 4,846-7,419 (26-28% of
their own, much smaller, totals) — i.e. regression models capture only
~5-13% of the true number of small systems, not "the right number." The
CCDF's apparent small-L agreement is a normalization artifact, not
evidence of correct small-system counts. This is the same blur mechanism
seen from the other side: merging ~97k small storms into far fewer,
larger blobs simultaneously crashes total system count (127k→18-26k)
*and* skews the survivors toward large sizes. Diffusion's absolute
small-system count (90,741, 84% of its own 107,705 total) is close to
truth's, confirming it isn't merging nearly as aggressively.

Diffusion mostly, not perfectly, corrects the large-system side of this
(median is actually *smaller* than truth's, tail still ~2x inflated rather than
10-18x). **Persistence and random track truth's tail almost exactly** —
expected, since both simply reuse real (if temporally/spatially
misaligned) rain fields rather than a regression output — which is a
clean confirmation that the tail-inflation is a genuine model artifact,
not something inherent to "any prediction of an active domain."

## 3. Large-scale POD saturation is a metric property, not a model property

At the largest 1-2 scale bins (>=~670 km), POD saturates near 1.0 for
*every* track, including persistence and (at 1223 km specifically) even
briefly for random. This is not evidence any model is "good" at that
scale — very few systems are that large in any sample (single digits to
low tens per lead), and IoU>=0.1 is an easy bar to clear for a system
spanning hundreds of km. Any tau(L) fit that saturates at the fit's upper
bound (200h) is flagged, not reported as a real number, in
`track_compare_tau_scaling.png`.

## 4. Old-split regression's held-out set isn't the same storms as the other tracks

Old-split (`mseuv_lead*`) evaluates against 4 fixed whole days
([2,12,22,32]); every patch-split track evaluates against scattered
blocks across all 40 days. Keep this in mind when comparing old-split
against the other tracks at the pixel level — they are scored against
genuinely different storm events, not a shared held-out set.

(An earlier, now-superseded version of the confusion-matrix figure —
coarse 4-category dry/light/mod/heavy bins, built from a single piece/
4 frames — showed old-split with a *higher* correct-classification rate
for "heavy rain" than two-head, which looked backwards. That figure has
since been rebuilt twice: first to a fine-grained 30-bin contingency
heatmap, then to accumulate over each track's *actual full held-out test
scope* — all 40 pieces for patch-split tracks/baselines, restricted to
real test blocks, ~34.6M pixels per track — instead of one piece. The
current `confusion_matrix_compare_lead*.png` figures are the trustworthy
version; don't draw conclusions from the coarse/small-sample one, which
no longer exists.)

With the full-scan data, the pixel-level picture is what you'd expect:
all four model tracks show a tight, smoothly-decaying spread around the
diagonal that thins out toward high truth values (under-prediction of
extremes — the same MSE-smoothing bias behind the system-size
tail-inflation finding above). Persistence shows visibly more spread and
extends further along the diagonal at high truth/high-predicted values
than any model track — not a sign it's "bad," but the expected
consequence of reusing an actual observed rain field (full real-world
variance) rather than a smoothed model output. Climatology stays a
tight near-zero vertical stripe (never predicts above light rain, no
local extremes in a mean field). Random shows the expected decorrelated
scatter concentrated at low truth values, no diagonal structure.

## 5. Baselines behave as expected, which is itself a useful check

Persistence is a strong baseline at short lead and large scale (tropical
convective systems evolve slowly enough that "nothing changes" is a
decent 15-60min forecast) but decays faster than any trained model with
lead. Climatology never predicts anything above light rain (a smooth
mean field has no local extremes) and is flat/near-zero everywhere in
POD except where geometric overlap luck dominates. Random is
decorrelated by construction and behaves accordingly. None of the
baselines threaten the model tracks' ranking at any scale/lead that
actually has enough systems to be statistically meaningful.
