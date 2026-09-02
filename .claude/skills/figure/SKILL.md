---
name: figure
description: Produce or revise a figure following the project's conventions (one message per panel, typography-forward, log axes for power laws) and register it. Use for any figure destined for results/ or the paper.
---

# Figure

1. One message per panel. Write the caption's first sentence before writing code; the
   figure must make that sentence obvious without reading the caption.
2. Script lives in `results/fig_R<n>.py`, reads only from `data/` and `src/`, writes
   `results/fig_R<n>.pdf`. Deterministic: same data → same figure.
3. Style: `plt.style.use("docs/latentsky.mplstyle")` — serif labels, 9 pt, no top/right
   spines, no titles inside panels (caption carries it), colorblind-safe palette,
   LatentSky = one fixed color, persistence = grey, WB2 models = muted.
   Log axes where the quantity scales as a power law (L, τ). Units on every axis.
4. Fits drawn as thin lines with the exponent printed in the legend, data as markers.
5. Panels labeled (a), (b) top-left. Width 8.5 cm (single column) or 17.5 cm (double).
6. After saving, add or update the `results/REGISTRY.md` row.
