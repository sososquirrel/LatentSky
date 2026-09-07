# AGU Advances draft

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

AGU Advances specifics honored in the skeleton:
- exactly **3 Key Points**, ≤ 140 characters each, no math markup;
- **Plain Language Summary** required;
- Open Research (data/code availability) section required;
- word budget ~5000 words excluding methods/captions — keep the five acts tight.

Figures come from the page assets (`docs/`, `wb2_benchmark/`); the final versions
will be regenerated as `results/fig_R<n>.py` per repo rules before submission.
