"""Diurnal cycle of small (L<65 km) precipitating systems and of model POD,
split land vs ocean. Reads docs/scale_systems_v3.csv.gz. Login-node OK."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from latentsky.registry import REPO_ROOT as REPO
SYS = REPO / "docs" / "scale_systems_v3.csv.gz"
OUT = REPO / "docs" / "figures" / "diurnal_small_systems.png"

SMALL_KM = 65
LEADS = [0.0, 0.5, 1.0, 1.5, 3.0]
LST_EDGES = np.arange(0, 25, 1)        # 1-h bins, 0..24
LST_MID = 0.5 * (LST_EDGES[:-1] + LST_EDGES[1:])

df = pd.read_csv(SYS)
df = df[df["predictor_set"] == "9var (mse+uv)"].copy()
df["scale_use"] = np.where(df["role"] == "false_alarm", df["L_p"], df["L_t"])
df = df[df["scale_use"] < SMALL_KM]
df["lst_bin"] = np.digitize(df["lst_hour"] % 24, LST_EDGES) - 1
df = df[(df["lst_bin"] >= 0) & (df["lst_bin"] < len(LST_MID))]


def stats_by_bin(g):
    nm = (g["role"] == "match").sum()
    nmiss = (g["role"] == "miss").sum()
    nfa = (g["role"] == "false_alarm").sum()
    n_truth = nm + nmiss
    n_pred = nm + nfa
    return pd.Series({
        "n_truth": n_truth,
        "POD": nm / n_truth if n_truth else np.nan,
        "FAR": nfa / n_pred if n_pred else np.nan,
    })


fig, (axN, axP, axF) = plt.subplots(1, 3, figsize=(15.5, 4.2),
                                    constrained_layout=True)

# --- Left: truth-side population diurnal cycle (lead-independent) ---
# use lead=0h subset, role in {match, miss} → all truth systems
truth = df[(df["lead_h"] == 0.0) & (df["role"].isin(["match", "miss"]))]
pop = (truth.groupby(["lst_bin", "is_land"]).size()
       .unstack(fill_value=0).reindex(range(len(LST_MID)), fill_value=0))
# normalize each curve by its own mean so amplitude is comparable
pop_norm = pop / pop.mean()
axN.plot(LST_MID, pop_norm[False], lw=2, color="#1f77b4", label="ocean")
axN.plot(LST_MID, pop_norm[True],  lw=2, color="#d62728", label="land")
axN.axhline(1.0, color="0.7", lw=0.8, ls=":")
axN.set_xlabel("local solar time [h]")
axN.set_ylabel("system count / mean")
axN.set_title(f"(a) population cycle, L < {SMALL_KM} km", loc="left")
axN.set_xticks(range(0, 25, 3))
axN.set_xlim(0, 24)
axN.legend(frameon=False)
axN.grid(alpha=0.3)

# --- Right: POD vs LST, multiple leads, land/ocean ---
cmap = plt.get_cmap("viridis")
lead_colors = {lh: cmap(i / (len(LEADS) - 1)) for i, lh in enumerate(LEADS)}
for lh in LEADS:
    sub = df[df["lead_h"] == lh]
    for is_land, ls in [(False, "--"), (True, "-")]:
        s = sub[sub["is_land"] == is_land]
        out = (s.groupby("lst_bin").apply(stats_by_bin)
               .reindex(range(len(LST_MID))))
        axP.plot(LST_MID, out["POD"], ls=ls, marker="o", ms=3,
                 color=lead_colors[lh], lw=1.8)
        axF.plot(LST_MID, out["FAR"], ls=ls, marker="o", ms=3,
                 color=lead_colors[lh], lw=1.8)
for ax, ylabel, title in [
    (axP, f"POD  (small systems, L < {SMALL_KM} km)",
     "(b) model POD vs phase of day"),
    (axF, f"FAR  (small systems, L < {SMALL_KM} km)",
     "(c) model FAR vs phase of day"),
]:
    ax.set_xlabel("local solar time [h]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

# Two legends: one for lead (color), one for surface (linestyle).
# Long handlelength so the dash pattern actually renders.
from matplotlib.lines import Line2D
lead_handles = [Line2D([0], [0], color=lead_colors[lh], lw=2,
                       label=f"t = {lh:g} h") for lh in LEADS]
surf_handles = [Line2D([0], [0], color="0.25", lw=2, ls="-",  label="land"),
                Line2D([0], [0], color="0.25", lw=2, ls="--", label="ocean")]
leg1 = axP.legend(handles=lead_handles, loc="upper left", frameon=False,
                  fontsize=9, handlelength=2.5, title="lead", title_fontsize=9)
axP.add_artist(leg1)
axP.legend(handles=surf_handles, loc="upper left", bbox_to_anchor=(0.22, 1.0),
           frameon=False, fontsize=9, handlelength=3.0,
           title="surface", title_fontsize=9)

fig.suptitle("Diurnal cycle of small tropical precipitating systems (9var mse+uv)",
             y=1.04, fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
