# GRL manuscript

`main.tex` / `supplement.tex` are the Geophysical Research Letters version
(12 publication units: abstract + text + captions in units of 500 words,
plus one unit per figure; currently 5 figures). The longer AGU Advances
version is kept as `main_agu_advances.tex` / `supplement_agu_advances.tex`.

Compile on the cluster:

```
module load texlive/20250308
pdflatex main.tex   # (run twice + bibtex once refs are real)
```

`agujournal2019.cls` and `trackchanges.sty` are vendored in this folder
(fetched from the myst-templates/agu2019 mirror of the official AGU template).
Note AGU published a newer `agujournal2025.cls`
(github.com/AGU-Publications/agujournal2025-latex-template); switch before
submission if the journal requires it — the environments (keypoints, abstract,
plain-language summary) carry over.

Overleaf also works: its official AGU template ships the same class.

GRL specifics:
- exactly **3 Key Points**, <= 140 characters each, no math markup;
- abstract <= 150 words; **Plain Language Summary** <= 200 words;
- Open Research (data/code availability) section required;
- 12 publication units in total (check with AGU's counter before submission).

Figures are built by the scripts in `figures/` (one script per PDF; shared
constants and helpers come from `src/latentsky`, not from `docs/`):

| Figure | Script | Notes |
|---|---|---|
| 1 | `fig1_overview.py` | domain (`fig1a_domain.py`) + pipeline (`fig2_pipeline.py`, cached run) |
| 2 | `fig3_pairs.py` | runs the +3 h U-Net and flow on CPU |
| 3 | `fig4_decomp.py` | |
| 4 | `fig5_horizon.py` | |
| 5 | `fig6_wb2.py` | maps (`fig7a_wb2maps.py`) + size distributions (`fig5_sizedist.py`) |
| S1, S2 | `fig2_io.py` + `make_arch.py`, `fig3_flowsketch.py` | |
| S3, S4, S5 | `fig7a_wb2maps.py`, `fig7c_clouds.py`, `fig7b_pod.py` | |

They are not yet registered in `results/REGISTRY.md`; promotion
(`/promote-result`) needs the sensitivity tests first.
