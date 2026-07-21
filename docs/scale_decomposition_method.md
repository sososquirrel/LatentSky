# Scale-aware precipitation decomposition — method

How the object-based `P = ρ·A` analysis is computed, end to end. Code lives
in [`latentsky/scale_analysis.py`](../latentsky/scale_analysis.py); the eval
driver is [`scripts/eval_scale_decomposition.py`](../scripts/eval_scale_decomposition.py).

## 1. What we decompose

For every precipitating *system* (a connected blob of rain) we compute three
quantities and use the exact identity

```
P = ρ · A          ln P = ln A + ln ρ
```

- **A** — precipitating area of the system, in km² (latitude-weighted).
- **P** — total precipitation it carries, `Σ rate · cell_area`, units mm·km²
  (a precip "volume").
- **ρ = P / A** — the *area-weighted* mean intensity over the system, mm.
- **L = √A** — its characteristic length scale, km. This is the binning key.

Defining ρ as the area-weighted mean (not the plain pixel mean) is what makes
`P = ρ·A` hold *exactly* rather than approximately — verified on the real data
to machine precision (`|P − ρ·A|` max relative ≈ 1e-15).

## 2. Grid-cell area (latitude weighting)

The data is a regular lat/lon grid (tropical SAM crop, lat ∈ [−40, 40]°,
lon ∈ [0, 360]°, 0.25° cells). On such a grid a "pixel" is **not** constant
area — cells shrink toward the poles by cos(latitude). For row at latitude φ:

```
Δlat_km = deg2rad(Δlat) · R
Δlon_km = deg2rad(Δlon) · R · cos(φ)
cell_area(φ) = Δlat_km · Δlon_km          R = 6371 km
```

`cell_area_km2` returns one value per grid row (longitude-uniform). Over
[−40, 40]° cells shrink ~23 % from equator to edge, so every area and
intensity below is computed against this per-row area, never a raw pixel
count. (`cell_area_km2`)

## 3. Segmenting systems (connected components)

For one 2-D field (truth or prediction) at one time:

1. Threshold: `mask = field > threshold` (default **0.1 mm**) — defines
   "precipitating".
2. Label: `scipy.ndimage.label` with **8-connectivity** (diagonally touching
   wet pixels join the same system).
3. Per label, accumulate with `ndimage.sum` against the 2-D cell-area field:
   - `A   = Σ cell_area`            (km²)
   - `P   = Σ field · cell_area`    (mm·km²)
   - `ρ   = P / A`
   - `L   = √A`
   - centroid (for diagnostics)

This runs independently on the truth field and the predicted field, giving
two lists of systems per frame. (`label_systems`, `_object_table`)

## 4. Matching truth ↔ prediction

To attribute error per system, predicted systems are matched to truth
systems by **spatial overlap** within each frame:

1. Build the intersection-area matrix between every truth label and every
   pred label in one pass (`np.add.at` scatter over the overlapping pixels —
   no O(n²) loop). (`_overlap_matrix`)
2. IoU = `intersection / (A_truth + A_pred − intersection)`.
3. **Greedy assignment**: take pairs in descending IoU order; accept a pair
   if neither system is already used and `IoU ≥ iou_min` (default **0.1**).
   This yields one-to-one matches.
4. Leftovers get a role:
   - **match** — paired truth+pred system (has both A_t/ρ_t/P_t and A_p/ρ_p/P_p)
   - **miss** — a truth system with no predicted counterpart (model missed it)
   - **false_alarm** — a predicted system with no truth counterpart (spurious)

(`match_systems`) Stacked over all frames/pieces this becomes one tidy
table, one row per system, with a `scale` column = L_truth (or L_pred for
false alarms). (`system_table`)

## 5. The log decomposition

For **matched** systems only (miss/false-alarm have P=0 on one side, so ln P
is undefined there):

```
dlnA   = ln(A_pred / A_truth)        area error
dlnρ   = ln(ρ_pred / ρ_truth)        intensity error
dlnP   = ln(P_pred / P_truth) = dlnA + dlnρ      (exact, no cross term)
```

**Sign convention** (applies identically to `dlnA`, `dlnρ`, `dlnP`):

- **negative ⇒ underestimate** — pred < truth (e.g. `dlnρ < 0`: system
  predicted too weak; `dlnA < 0`: too small)
- **positive ⇒ overestimate** — pred > truth (e.g. `dlnA > 0`: too big)
- zero ⇒ unbiased

The log form is scale-free, so a +50 % area error on a 30 km cell and on a
400 km MCS contribute comparably — essential when systems span orders of
magnitude.

## 6. Scale binning

Bin edges are **log-spaced in L** (`default_scale_edges`,
`np.geomspace`). The canonical run uses 8 bins over 25–3000 km; the
**retuned v2** run uses **10 bins over 25–600 km** because almost no real
tropical system exceeds ~600 km — the upper bins were empty in v1 and are
dropped. Each system is assigned a bin with `np.digitize` on its `scale`.

## 7. Distributions

The truth-vs-pred distributions (figure panels (a) size, (f) intensity) are
**unpaired** — they need no matching. From the system table:

- truth systems = rows where the truth side exists (`role ∈ {match, miss}`)
- pred systems  = rows where the pred side exists (`role ∈ {match, false_alarm}`)

Histograms are taken on **log-spaced edges** spanning the combined range
(`np.geomspace(min, max, n)`), plotted on log-x. "Precip by scale" (panel b)
is `Σ P` within each scale bin, truth vs pred — it shows *where the water
is*, not just how many systems there are. (`summarize_by_scale`,
`ScaleSummary.distributions`)

## 8. Per-scale skill (object-based)

Within each scale bin (`summarize_by_scale`):

```
POD = n_match / (n_match + n_miss)         detection rate of real systems
FAR = n_fa    / (n_match + n_fa)           fraction of predicted that were spurious
```

These are **object-based**, not pixel-based: the unit is a whole
precipitating system, not a grid cell. POD high & flat ≈1 and FAR low &
flat ≈0 across scale would be a perfect model; the observed POD rising from
~0 with scale is the quantitative statement of "the model only sees large
systems". (Undefined when a bin has 0 truth systems → NaN, by design.)

## 9. Precip-bias budget

Per scale bin, the matched-system bias is split into how much comes from
getting area wrong vs intensity wrong, precipitation-weighted (weight =
P_truth):

```
area_frac = |Σ w·dlnA| / (|Σ w·dlnA| + |Σ w·dlnρ|)
int_frac  = |Σ w·dlnρ| / (|Σ w·dlnA| + |Σ w·dlnρ|)        area_frac + int_frac = 1
```

Misses and false alarms cannot enter the log decomposition (P=0 on one
side), so the precip they carry is tracked separately as a detection term:

```
miss_P_frac = Σ P_truth(miss)        / Σ P_truth(bin)
fa_P_frac   = Σ P_pred(false_alarm)  / Σ P_pred(bin)
```

These quantify the share of the precip-bias budget that is a *detection*
failure rather than an area/intensity distortion.

## 10. Outputs

`eval_scale_decomposition.py` writes (summary first, so a multi-hour GPU
job is never lost to a serialization error):

- **`docs/scale_summary.csv`** — one row per (run × scale bin): the
  POD/FAR/dln*/budget columns above, carrying `predictor_set` and `lead_h`
  so it joins the lead-time study.
- **`docs/scale_systems.csv.gz`** — the raw per-system table (~1.7 M rows),
  everything else derives from this. (Parquet is used only if `pyarrow` is
  importable; otherwise gzip CSV — no hard dependency.)

`_v2` suffixed files are the retuned-bin re-run; the originals are kept.

## 11. Reproduce

```bash
# eval (GPU, ~2 h) — develop queue starts in seconds
qsub -v SUMMARY_OUT=docs/scale_summary_v2.csv,\
        SYSTEMS_OUT=docs/scale_systems_v2.csv.gz,\
        SCALE_MAX=600,SCALE_BINS=10 submit/eval_scale.pbs

# figures (no GPU, login node)
python scripts/plot_scale_decomposition.py --systems docs/scale_systems_v2.csv.gz \
       --summary docs/scale_summary_v2.csv --run mseuv_lead3h
python scripts/plot_scale_leads.py --summary docs/scale_summary_v2.csv \
       --predictor-set "9var (mse+uv)"
```

Interactive walkthrough: [`notebooks/04_scale_decomposition.ipynb`](../notebooks/04_scale_decomposition.ipynb).
