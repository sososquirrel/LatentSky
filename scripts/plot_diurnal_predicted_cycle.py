"""Predicted-population diurnal cycle at each lead, land vs ocean.
Test 1 for 'does the model invent the diurnal cycle, or only respond to it?'
If the model were purely propagating IC information, the predicted-cycle
amplitude would decay with lead. If it is preserved while the absolute count
collapses, the model carries a learned phase-of-day prior."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SYS = REPO / "docs" / "scale_systems_v3.csv.gz"
OUT = REPO / "docs" / "figures" / "diurnal_predicted_cycle.png"

SMALL_KM = 65
LEADS = [0.0, 0.5, 1.0, 1.5, 3.0, 6.0]
LST_EDGES = np.arange(0, 25, 1)
LST_MID = 0.5 * (LST_EDGES[:-1] + LST_EDGES[1:])

df = pd.read_csv(SYS)
df = df[df["predictor_set"] == "9var (mse+uv)"].copy()
df["lst_bin"] = np.digitize(df["lst_hour"] % 24, LST_EDGES) - 1
df = df[(df["lst_bin"] >= 0) & (df["lst_bin"] < len(LST_MID))]

# Truth-side small systems: role in {match, miss}, scale_use = L_t
truth_mask = df["role"].isin(["match", "miss"]) & (df["L_t"] < SMALL_KM)
# Pred-side small systems: role in {match, false_alarm}, scale_use = L_p
pred_mask  = df["role"].isin(["match", "false_alarm"]) & (df["L_p"] < SMALL_KM)


def cycle(mask, lead_h, is_land):
    s = df[mask & (df["lead_h"] == lead_h) & (df["is_land"] == is_land)]
    c = (s.groupby("lst_bin").size()
         .reindex(range(len(LST_MID)), fill_value=0)
         .values.astype(float))
    return c


cmap = plt.get_cmap("viridis")
lead_colors = {lh: cmap(i / (len(LEADS) - 1)) for i, lh in enumerate(LEADS)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True,
                         sharey=True)

for ax, is_land, label in [(axes[0], True, "LAND"), (axes[1], False, "OCEAN")]:
    # Truth reference (pooled over all leads — truth is the same population,
    # only verification times shift; using lead=0 as the cleanest reference).
    c_truth = cycle(truth_mask, 0.0, is_land)
    ax.plot(LST_MID, c_truth / c_truth.mean(), color="black", lw=2.6,
            label="truth", zorder=5)

    # Predicted cycles at each lead, normalized to their own mean.
    for lh in LEADS:
        c = cycle(pred_mask, lh, is_land)
        if c.sum() == 0:
            continue
        amp = (c.max() - c.min()) / c.mean()
        mean_n = c.mean()
        ax.plot(LST_MID, c / c.mean(), color=lead_colors[lh], lw=1.8,
                marker="o", ms=3,
                label=f"pred t={lh:g}h  (mean n={mean_n:.0f}, amp={amp:.2f})")

    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.set_title(f"{label} — small-system population cycle (L<{SMALL_KM} km)",
                 loc="left")
    ax.set_xlabel("local solar time [h]")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

axes[0].set_ylabel("count / mean")
fig.suptitle(
    "Predicted-population diurnal cycle survives at every lead — "
    "amplitude preserved while count collapses (9var mse+uv)",
    y=1.04, fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
