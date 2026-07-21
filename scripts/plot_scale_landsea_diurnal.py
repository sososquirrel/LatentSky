"""Small-system detection stratified by land/ocean and time of day.

Answers: among the small precipitating systems the model loses, are the
misses tied to land vs ocean, and to the diurnal cycle (afternoon land
convection)? Uses the v3 table (per-system lat/lon/is_land/lst_hour).

  (a) POD vs scale, land vs ocean            does the small-scale collapse
                                              differ over land and sea?
  (b) small-system POD vs lead, land vs ocean detection of L < small_km
                                              systems as the horizon grows
  (c) diurnal: truth small-system count       the physical cycle — land
      vs local solar time, land vs ocean      convection should peak in the
                                              afternoon
  (d) diurnal: small-system POD vs LST,       does the model specifically
      land vs ocean                           miss the afternoon land peak?

POD = matched / (matched + missed)  (truth systems only).
Note: the time-slip truncates pre-dawn hours at long leads — early-LST
bins are under-sampled for the longer-lead curves.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pod(sub: pd.DataFrame) -> float:
    """matched / (matched + missed) over truth systems in `sub`."""
    nm = (sub["role"] == "match").sum()
    nmiss = (sub["role"] == "miss").sum()
    n = nm + nmiss
    return nm / n if n else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="docs/scale_systems_v3.csv.gz")
    p.add_argument("--summary", default="docs/scale_summary_v3.csv")
    p.add_argument("--predictor-set", default="9var (mse+uv)")
    p.add_argument("--lead", type=float, default=3.0,
                   help="Lead (h) for the scale & diurnal panels (a,c,d).")
    p.add_argument("--small-km", type=float, default=150.0,
                   help="Systems with L < this are 'small' (b,c,d).")
    p.add_argument("--lst-bins", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    sp = repo / args.systems
    df = (pd.read_parquet(sp) if str(sp).endswith(".parquet")
          else pd.read_csv(sp))
    df = df[df["predictor_set"] == args.predictor_set]
    if df.empty:
        raise SystemExit(f"no rows for {args.predictor_set!r} in {sp}")
    if "is_land" not in df.columns:
        raise SystemExit("table has no is_land column — needs the v3 re-eval")

    # truth systems only (match or miss) carry a real location/time
    truth = df[df["role"].isin(["match", "miss"])].copy()
    truth["surf"] = np.where(truth["is_land"], "land", "ocean")
    SURF = {"land": ("C1", "land"), "ocean": ("C0", "ocean")}

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # ---- (a) POD vs scale, land vs ocean, at args.lead --------------------
    g = truth[np.isclose(truth["lead_h"], args.lead)]
    summ = pd.read_csv(repo / args.summary)
    sm = summ[(summ["predictor_set"] == args.predictor_set)
              & np.isclose(summ["lead_h"], args.lead)].sort_values("scale_mid")
    edges = np.r_[sm["scale_lo"].to_numpy(), sm["scale_hi"].to_numpy()[-1]]
    mids = sm["scale_mid"].to_numpy()
    for s, (c, lab) in SURF.items():
        gs = g[g["surf"] == s]
        b = np.clip(np.digitize(gs["scale"], edges) - 1, 0, len(edges) - 2)
        pod = [_pod(gs[b == k]) for k in range(len(edges) - 1)]
        ax[0, 0].plot(mids, pod, "o-", color=c, label=lab)
    ax[0, 0].set_xscale("log"); ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_xlabel("system scale L = √A  (km)"); ax[0, 0].set_ylabel("POD")
    ax[0, 0].set_title(f"(a) POD vs scale — lead {args.lead:g} h")
    ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.25)

    # ---- (b) small-system POD vs lead, land vs ocean ---------------------
    small = truth[truth["scale"] < args.small_km]
    leads = sorted(small["lead_h"].unique())
    for s, (c, lab) in SURF.items():
        ss = small[small["surf"] == s]
        pod = [_pod(ss[np.isclose(ss["lead_h"], ld)]) for ld in leads]
        ax[0, 1].plot(leads, pod, "o-", color=c, label=lab)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_xlabel("lead time (h)")
    ax[0, 1].set_ylabel(f"POD  (systems L < {args.small_km:g} km)")
    ax[0, 1].set_title("(b) small-system detection vs lead")
    ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)

    # ---- (c) diurnal: truth small-system count vs LST --------------------
    lst_edges = np.linspace(0, 24, args.lst_bins + 1)
    lst_mid = 0.5 * (lst_edges[:-1] + lst_edges[1:])
    sm_l = small[np.isclose(small["lead_h"], args.lead)]
    for s, (c, lab) in SURF.items():
        ss = sm_l[sm_l["surf"] == s]
        cnt, _ = np.histogram(ss["lst_hour"], bins=lst_edges)
        ax[1, 0].plot(lst_mid, cnt, "o-", color=c, label=lab)
    ax[1, 0].set_xlabel("local solar time (h)")
    ax[1, 0].set_ylabel(f"# truth systems  (L < {args.small_km:g} km)")
    ax[1, 0].set_title(f"(c) diurnal cycle of small systems — lead {args.lead:g} h")
    ax[1, 0].set_xticks(np.arange(0, 25, 6))
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    # ---- (d) diurnal: small-system POD vs LST ---------------------------
    for s, (c, lab) in SURF.items():
        ss = sm_l[sm_l["surf"] == s]
        b = np.clip(np.digitize(ss["lst_hour"], lst_edges) - 1,
                    0, len(lst_edges) - 2)
        pod = [_pod(ss[b == k]) for k in range(len(lst_edges) - 1)]
        ax[1, 1].plot(lst_mid, pod, "o-", color=c, label=lab)
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_xlabel("local solar time (h)")
    ax[1, 1].set_ylabel(f"POD  (L < {args.small_km:g} km)")
    ax[1, 1].set_title(f"(d) small-system POD by time of day — lead {args.lead:g} h")
    ax[1, 1].set_xticks(np.arange(0, 25, 6))
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Small-system detection by land/ocean & diurnal cycle — "
        f"{args.predictor_set}", fontsize=14)
    fig.tight_layout()
    out = (repo / (args.out
           or f"docs/figures/scale_landsea_diurnal_lead{args.lead:g}h.png")
           ).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
