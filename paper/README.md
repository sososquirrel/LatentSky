# AGU Advances draft

`main.tex` targets the official AGU class `agujournal2019.cls`. That class is not
on this login node (no TeX install) and is not fetchable from a stable URL, so:

- **Compile on Overleaf**: open the official "AGU (American Geophysical Union)"
  template on Overleaf, replace its `main.tex` with this one, keep the template's
  `agujournal2019.cls`, `agutexSI2019.cls`, and `trackchanges.sty`.
- Or drop those three files into this folder and compile locally on a node with TeX.

AGU Advances specifics honored in the skeleton:
- exactly **3 Key Points**, ≤ 140 characters each, no math markup;
- **Plain Language Summary** required;
- Open Research (data/code availability) section required;
- word budget ~5000 words excluding methods/captions — keep the five acts tight.

Figures come from the page assets (`docs/`, `wb2_benchmark/`); the final versions
will be regenerated as `results/fig_R<n>.py` per repo rules before submission.
