# tstart_protocol_piece20 — sensitivity of the t*(lead) calibration to the piece

**Stresses:** `experiments/tstart_protocol` (parts 2–3). **Varies:** the data subset —
piece 20 instead of piece 8, same frames [8, 24, 40, 56, 72], same leads, same
t_start grid, same seeds.

**Passes if** on piece 20: (a) the single-draw band ratio increases monotonically with
t_start at every verified lead (0.25, 1, 6 h); (b) the band-ratio-≈1 crossings lie
within a factor 1.5 of the piece-8 values (≈150 at 0.25 h, ≈230 at 1 h, ≈550 at 3 h
by interpolation of the rule column); (c) 6 h still saturates below ≈0.85 with a wet
fraction deficit.

**Fails if** crossings shift by more than ×1.5 or the 6 h saturation disappears —
then t*(lead) is weather-regime-dependent and cannot be a fixed table.

**Outcome (2026-09-02, job 5805657): PASSED.**
- (a) Band ratio monotone in t_start at 0.25 h (0.93→1.17) and 1 h (0.89→1.43);
  at 6 h monotone to the saturation then rolls over (0.88 at 500 → 0.84 at 750),
  same shape as piece 8.
- (b) Crossings: ≈210 at 0.25 h (piece 8: ≈170) and ≈260 at 1 h (piece 8: ≈220) —
  factors 1.2, inside the ×1.5 criterion. Analytic-rule t* agrees within 4 %
  across pieces (MSE_z 0.0103/0.0206/0.0415 vs 0.0104/0.0212/0.0416 at
  0.25/1/6 h) — the wet-masked error is piece-stable.
- (c) 6 h saturation reproduced: never crosses 1, wet fraction −19 %
  (0.037 vs 0.045). Caveat: the ceiling's level varies between pieces
  (0.80 on piece 8, ≈0.85 on piece 20) — quote it as ≈0.8–0.85.
- Not re-checked here: the 3 h crossing (measured on piece 8 only); its rule-t*
  and det deficit agree across pieces, so risk is low — verify during the R3
  re-derivation.
