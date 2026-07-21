"""Figure: masked-MSE loss formula, rendered standalone for a slide."""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"

fig, ax = plt.subplots(figsize=(9, 2.2))
ax.axis("off")

ax.text(
    0.5, 0.62,
    r"$\mathcal{L} \;=\; \dfrac{1}{|\mathcal{M}|} \sum_{i \,\in\, \mathcal{M}} "
    r"\left( \hat{y}_i - y_i \right)^2$",
    ha="center", va="center", fontsize=26, transform=ax.transAxes,
)
ax.text(
    0.5, 0.12,
    r"$\mathcal{M} = \{\, i : y_i > 0 \,\}$",
    ha="center", va="center", fontsize=20, transform=ax.transAxes,
)

for out_name, transparent in [("fig19_loss_formula.png", False),
                              ("fig19_loss_formula_transparent.png", True)]:
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=transparent)
    print(f"wrote {out}", flush=True)
