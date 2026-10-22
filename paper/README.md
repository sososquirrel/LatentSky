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

Figures: `figures/` holds the five figures of the paper and `figures_si/` the
supplementary ones, named by their number in the paper. They are built by the
scripts in `../paper_figures/` (same names; shared constants and helpers come
from `src/latentsky`). That folder also holds the helper scripts and, in
`extra/`, everything not shown in the paper (intermediate panels, previews,
older figures). Only `paper/` is synced to Overleaf.

| Figure | Script (in `paper_figures/`) | Notes |
|---|---|---|
| 1 | `fig1_overview.py` | domain (`fig1_domain.py`) + pipeline (`fig1_pipeline.py`, cached run) |
| 2 | `fig2_pairs.py` | runs the +3 h U-Net and flow on CPU |
| 3 | `fig3_decomp.py` | |
| 4 | `fig4_horizon.py` | |
| 5 | `fig5_wb2.py` | maps (`fig5_wb2maps.py`) + size distributions (`fig5_sizedist.py`) |
| S1 | `figS1_unet.py` | inputs and outputs from `figS1_io.py`, U-Net glyph from `fig1_pipeline.py` |
| S2 | `fig5_wb2maps.py` | the whole tropical belt |
| S3, S4 | `figS3_wb2_clouds.py`, `figS4_wb2_pod.py` | WB2 constants in `wb2_common.py` |

They are not yet registered in `results/REGISTRY.md`; promotion
(`/promote-result`) needs the sensitivity tests first.
