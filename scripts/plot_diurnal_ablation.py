"""Test 3 figure: predicted-population diurnal cycle under input ablations.

Three columns side by side — baseline vs t2m-killed vs solar-killed — for
land (top) and ocean (bottom). If the predicted afternoon land peak survives
every ablation, the diurnal cycle is in the model's weights. If it collapses
to flat, the model was reading the clock off the inputs every step.

Run after the two ablation PBS jobs finish:
  docs/scale_systems_v3.csv.gz                 (baseline, already on disk)
  docs/scale_systems_v3_t2m_killed.csv.gz      (job 6201456)
  docs/scale_systems_v3_solar_killed.csv.gz    (job 6201460)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "diurnal_ablation_predicted_cycle.png"

CASES = [
    ("baseline",        "docs/scale_systems_v3.csv.gz"),
    ("t2m killed",      "docs/scale_systems_v3_t2m_killed.csv.gz"),
    ("solar killed",    "docs/scale_systems_v3_solar_killed.csv.gz"),
]
SMALL_KM = 65
LEADS = [0.0, 0.5, 1.0, 1.5, 3.0, 6.0]
LST_EDGES = np.arange(0, 25, 1)
LST_MID = 0.5 * (LST_EDGES[:-1] + LST_EDGES[1:])
cmap = plt.get_cmap("viridis")
lead_colors = {lh: cmap(i / (len(LEADS) - 1)) for i, lh in enumerate(LEADS)}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / path)
    df = df[df["predictor_set"] == "9var (mse+uv)"].copy()
    df["lst_bin"] = np.digitize(df["lst_hour"] % 24, LST_EDGES) - 1
    return df[(df["lst_bin"] >= 0) & (df["lst_bin"] < len(LST_MID))]


def cycle(df, lead_h, is_land, side):
    """side='truth' uses role∈{match,miss}+L_t; 'pred' uses {match,fa}+L_p."""
    if side == "truth":
        mask = df["role"].isin(["match", "miss"]) & (df["L_t"] < SMALL_KM)
    else:
        mask = df["role"].isin(["match", "false_alarm"]) & (df["L_p"] < SMALL_KM)
    s = df[mask & (df["lead_h"] == lead_h) & (df["is_land"] == is_land)]
    c = (s.groupby("lst_bin").size()
         .reindex(range(len(LST_MID)), fill_value=0).values.astype(float))
    return c


# Load everything that exists; warn on missing files (so the script works
# while jobs are still running — for sanity-checking the layout).
loaded = []
for label, path in CASES:
    full = REPO / path
    if not full.exists():
        print(f"missing: {full} — skipping column.")
        continue
    loaded.append((label, load(path)))

if not loaded:
    raise SystemExit("no CSVs found.")

fig, axes = plt.subplots(2, len(loaded), figsize=(5.2 * len(loaded), 8.0),
                         constrained_layout=True, sharey="row")
if len(loaded) == 1:
    axes = axes[:, None]

for col, (label, df) in enumerate(loaded):
    for row, (is_land, surf_label) in enumerate([(True, "LAND"), (False, "OCEAN")]):
        ax = axes[row, col]
        # truth reference (same for all ablations: truth is truth)
        c_t = cycle(df, 0.0, is_land, "truth")
        ax.plot(LST_MID, c_t / c_t.mean(), color="black", lw=2.4,
                label="truth", zorder=5)
        for lh in LEADS:
            c = cycle(df, lh, is_land, "pred")
            if c.sum() == 0:
                continue
            amp = (c.max() - c.min()) / c.mean()
            mean_n = c.mean()
            ax.plot(LST_MID, c / c.mean(), color=lead_colors[lh], lw=1.7,
                    marker="o", ms=3,
                    label=f"t={lh:g}h  (n̄={mean_n:.0f}, amp={amp:.2f})")
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f"{label}", loc="left", fontsize=11, weight="bold")
        if col == 0:
            ax.set_ylabel(f"{surf_label}\ncount / mean")
        if row == 1:
            ax.set_xlabel("local solar time [h]")
        ax.legend(frameon=False, fontsize=7, loc="upper left")

fig.suptitle(
    f"Predicted-population diurnal cycle under input ablation (L<{SMALL_KM} km, 9var mse+uv)\n"
    "If the land afternoon peak survives all three columns → cycle lives in the weights.",
    y=1.02, fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
