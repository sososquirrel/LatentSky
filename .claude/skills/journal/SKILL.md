---
name: journal
description: Write or update today's lab-notebook entry in journal/YYYY-MM-DD.md documenting every experiment, its motivation, setup, numerical result and conclusion, then regenerate journal/INDEX.md. Use at the end of each session and whenever an experiment finishes.
---

# Journal

1. Determine today's date; open `journal/YYYY-MM-DD.md`, create it from `journal/_template.md`
   if absent. Never edit a previous day's file.
2. Collect what happened this session: experiments launched or finished (check
   `experiments/*/HYPOTHESIS.md` and `sensitivity_tests/*/HYPOTHESIS.md` modified today,
   `qstat -u $USER` for running jobs), files changed (`git diff --stat`, `git log --since=midnight`),
   figures produced, decisions taken in conversation.
3. Fill each section. Standards:
   - Every experiment run today has an entry, including failures ("job died OOM at 3 h").
   - Result lines carry numbers with units, scales, lead times. No adjectives without a number.
   - Conclusion is one of: supported / rejected / inconclusive, with the reason.
   - Decisions record what changed and why, especially changes to `docs/FRAME.md`.
   - Link files by path; do not paste code or tables longer than five rows.
   - Target length: 15–40 lines. Comprehensive means nothing that changed a decision is
     missing, not that everything is transcribed.
4. Update the Status line from `docs/FRAME.md`.
5. Regenerate `journal/INDEX.md`: header with current R-statuses, then one line per day,
   newest first: `- [YYYY-MM-DD](YYYY-MM-DD.md) — <objective, ≤ 15 words>`.
6. At the start of a session, if the last journal entry is not today's, read it and the
   INDEX before doing anything else.
