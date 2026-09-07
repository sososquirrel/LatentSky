"""fig_margins_leads.png: mid-scale (100-400 km) POD margin over each
model's own persistence null, +6 h (5 August days, mean with day-to-day
range) vs +24 h (month-pooled). The lead-dependence of the generative
family, in one picture. Reads multiday_margins.csv + margins_24h.csv."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/glade/u/home/sabramian/LatentSky")
OUT = REPO / "wb2_benchmark"
plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))

d6 = pd.read_csv(OUT / "multiday_margins.csv")
d24 = pd.read_csv(OUT / "margins_24h.csv")
d24 = d24[d24.band == "mid100-400"].set_index("model").margin

MODELS = [("graphcast", "graphcast", "graphcast"),
          ("fuxi", "fuxi", "fuxi"),
          ("gencast\n(mean)", None, "gencast"),
          ("fgn\n(mean)", "fgn-mean", "fgn"),
          ("fgn\n(member)", "fgn-member", None)]
x = np.arange(len(MODELS))
w = 0.36
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, (label, m6, m24) in enumerate(MODELS):
    if m6 is not None:
        g = d6[d6.model == m6].margin
        ax.bar(k - w / 2, g.mean(), w, color="#74add1", label="+6 h (5 days)"
               if k == 0 else None)
        ax.errorbar(k - w / 2, g.mean(),
                    yerr=[[g.mean() - g.min()], [g.max() - g.mean()]],
                    fmt="none", ecolor="0.2", lw=1.2, capsize=3)
    if m24 is not None:
        ax.bar(k + w / 2, d24[m24], w, color="#2166ac",
               label="+24 h (month)" if k == 0 else None)
ax.axhline(0, color="0.3", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in MODELS], fontsize=8.5)
ax.set_ylabel("POD − own persistence null\n(systems 100–400 km)")
ax.grid(True, axis="y", ls=":", lw=0.5, color="0.8")
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8)
ax.set_title("mid-scale skill over the null: the shortest lead is the "
             "generative family's\nhard regime (2020 models vs 2022 FGN; "
             "nulls nearly identical at 24 h)", fontsize=9)
fig.savefig(OUT / "fig_margins_leads.png", dpi=150, bbox_inches="tight")
print("saved fig_margins_leads.png")
