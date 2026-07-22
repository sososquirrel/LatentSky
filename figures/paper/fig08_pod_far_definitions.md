# POD / FAR: exact definitions and computation

Source: `latentsky/scale_analysis.py` (`label_systems`, `match_systems`, `summarize_by_scale`).

## 1. Object detection (per frame, per field)

A precipitating **system** is a spatially connected region where the field exceeds
a threshold (default 0.1 mm):

```
mask = field > threshold
labels = connected_components(mask, connectivity=8)   # scipy.ndimage.label, 8-connectivity
```

This is run **independently** on the truth field and the prediction field for the
same frame — truth and prediction get their own, separate sets of labeled systems.

For each labeled system `i`, compute (area-weighted, using true grid-cell area in
km² which varies with latitude via cos(lat)):

```
A   = sum(cell_area over system pixels)        [km^2]   -- area
P   = sum(rate * cell_area over system pixels) [mm*km^2] -- precip "volume"
rho = P / A                                    [mm]      -- mean intensity
L   = sqrt(A)                                  [km]      -- characteristic scale
```

By construction, `P = rho * A` exactly for every system.

## 2. Matching truth systems to predicted systems

For every truth system *i* and predicted system *j* in the same frame, compute the
Intersection-over-Union of their pixel masks (in km², not pixel count, so it
respects the latitude-dependent cell area):

```
IoU(i,j) = area(truth_i ∩ pred_j) / area(truth_i ∪ pred_j)
```

Matching is **greedy, descending IoU**: sort all (i,j) pairs by IoU from highest to
lowest; walk down the list; accept a pair as a match if `IoU >= iou_min` (default
0.1) **and** neither system has already been claimed by an earlier, higher-IoU
match. Each system can be matched at most once.

After this pass:
- **match** — a truth system matched to a predicted system (IoU ≥ iou_min).
- **miss** — a truth system with no acceptable predicted match.
- **false_alarm** — a predicted system with no acceptable truth match.

## 3. POD and FAR

Aggregated over all systems in a given scale bin (or overall):

```
POD = n_match / (n_match + n_miss)          =  hits / all truth systems
FAR = n_false_alarm / (n_match + n_false_alarm)  =  false alarms / all predicted systems
```

- **POD** (Probability Of Detection): of all the real systems that existed, what
  fraction did the model find (anywhere with IoU ≥ 0.1)? 1.0 = found every real
  system. 0.0 = found none.
- **FAR** (False Alarm Ratio): of all the systems the model predicted, what
  fraction didn't correspond to anything real? 0.0 = every prediction was real.
  1.0 = every prediction was spurious.

Both are computed **per scale bin** (systems grouped by their characteristic scale
`L`, using the truth system's `L` for matches/misses and the predicted system's `L`
for false alarms) via `default_scale_edges()` — log-spaced bins from 25 km to
3000 km by default.

## 4. What POD/FAR do *not* capture

Neither metric knows anything about intensity (`rho`) — a system can be perfectly
located and shaped (IoU ≥ 0.1) with wildly wrong rainfall intensity, and it still
counts as a "match." Intensity accuracy is captured separately by the
`dlnP = dlnA + dlnrho` decomposition (see the corresponding figures), computed only
for matched systems.
