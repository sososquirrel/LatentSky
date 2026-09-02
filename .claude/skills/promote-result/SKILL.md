---
name: promote-result
description: Check whether an experiment's conclusion can be promoted to results/ and write its REGISTRY entry. Use when an experiment is finished.
---

# Promote result

Illustrations (I-numbers in FRAME.md, Part 2) skip step 2: register with script and data,
phrase the claim as an observation, never as a finding.

Usage: `/promote-result <experiment folder>`

1. Read the experiment's `HYPOTHESIS.md`. Its Outcome must be filled with numbers.
2. Find sensitivity tests in `sensitivity_tests/` that name this experiment. Require ≥ 1
   with a filled Outcome that does not change the sign or the leading-order magnitude of
   the claim. If none: list the sensitivity test most likely to break the claim
   (fit form, threshold, subset of days, lead window) and stop.
3. Draft the claim: one declarative sentence with a number, a scale or a lead time.
   No hedging words. Check it against the R-number's row in `docs/FRAME.md`; if it
   contradicts the draft claim, say so plainly — a rejected hypothesis is still a result.
4. Create `results/fig_R<n>.py` following `/figure`, add the `REGISTRY.md` row, update the
   Status column in `docs/FRAME.md`.
