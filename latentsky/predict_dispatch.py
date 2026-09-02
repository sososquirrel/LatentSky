"""Unified (X, Y, Y_pred) dispatcher across the three prediction kinds
used in the cross-model comparison: a trained regression run, a trained
diffusion run (warm-started from its paired regression run), or a
non-learned baseline (persistence/climatology/random). Lets comparison
scripts loop over a list of tracks without each one re-implementing the
branch on kind.

Also centralizes the one place that knows how to name a run at a given
lead for every track family -- `tracks_for_lead()` -- so every comparison
script (confusion matrix, qualitative crops, precip decomposition, ...)
enumerates the same five models + three baselines the same way.
"""
from __future__ import annotations

from latentsky.baselines import compute_climatology_mean, predict_piece_baseline
from latentsky.diffusion_inference import predict_piece_diffusion
from latentsky.inference import predict_piece
from latentsky.registry import load_run

PIECES_DIR = "/glade/derecho/scratch/sabramian/latentsky_pieces"


def tracks_for_lead(lead_h: float) -> list[dict]:
    """The five model tracks + three baselines, named as of 2026-07-22's
    cross-model comparison. A track dict is consumed by
    `predict_piece_track` below; `label`/`color` are just for plotting.
    """
    lead = f"{lead_h:g}h"
    return [
        dict(kind="regression", run=f"mseuv_lead{lead}",
             label="regression (old split)", color="tab:blue"),
        dict(kind="regression", run=f"patchsplit_onehead_lead{lead}",
             label="regression (new split)", color="tab:purple"),
        dict(kind="regression", run=f"patchsplit_twohead_olr_fss_lead{lead}",
             label="two-head", color="tab:orange"),
        dict(kind="diffusion", run=f"diffusion_lead{lead}",
             label="two-head+diffusion", color="tab:green"),
        dict(kind="baseline", baseline_kind="persistence", lead_hours=lead_h,
             label="persistence", color="gray"),
        dict(kind="baseline", baseline_kind="climatology", lead_hours=lead_h,
             label="climatology", color="black"),
        dict(kind="baseline", baseline_kind="random", lead_hours=lead_h,
             label="random", color="brown"),
    ]


def summary_tracks_for_lead(lead_h: float) -> list[dict]:
    """Same five tracks + three baselines as `tracks_for_lead`, but
    pointing at precomputed scale_summary_*.csv files (for figures built
    from the full test-set scan, not one live-inference crop) instead of
    live run names."""
    lead = f"{lead_h:g}h"
    return [
        dict(label="regression (old split)", color="tab:blue",
             summary="docs/scale_summary_v3_rebinned8.csv", name=f"mseuv_lead{lead}"),
        dict(label="regression (new split)", color="tab:purple",
             summary="docs/scale_summary_onehead_patchsplit.csv",
             name=f"patchsplit_onehead_lead{lead}"),
        dict(label="two-head", color="tab:orange",
             summary="docs/scale_summary_modelcompare.csv",
             name=f"patchsplit_twohead_olr_fss_lead{lead}"),
        dict(label="two-head+diffusion", color="tab:green",
             summary="docs/scale_summary_diffusion.csv", name=f"diffusion_lead{lead}"),
        dict(label="persistence", color="gray",
             summary="docs/scale_summary_baselines_patchsplit.csv", name=f"persistence_lead{lead}"),
        dict(label="climatology", color="black",
             summary="docs/scale_summary_baselines_patchsplit.csv", name=f"climatology_lead{lead}"),
        dict(label="random", color="brown",
             summary="docs/scale_summary_baselines_patchsplit.csv", name=f"random_lead{lead}"),
    ]


def predict_piece_track(track: dict, piece_id: int, max_frames: int, device=None):
    """Return (X, Y, Y_pred) for any track dict from `tracks_for_lead`
    (or hand-built the same way). Raises (does not silently skip) if the
    underlying run/checkpoint doesn't exist yet -- callers should catch
    and skip, since which tracks are ready varies while training/eval
    jobs are still in flight.
    """
    kind = track["kind"]
    if kind == "regression":
        run = load_run(track["run"])
        return predict_piece(run, piece_id, max_frames=max_frames, device=device)
    if kind == "diffusion":
        run = load_run(track["run"])
        return predict_piece_diffusion(
            run, piece_id, max_frames=max_frames,
            t_start=track.get("t_start", 100),
            ensemble_size=track.get("ensemble_size", 3),
            device=device,
        )
    if kind == "baseline":
        pieces_dir = track.get("pieces_dir", PIECES_DIR)
        n_total = track.get("n_total_pieces", 40)
        clim_mean = track.get("_clim_mean")
        if track["baseline_kind"] == "climatology" and clim_mean is None:
            clim_mean = compute_climatology_mean(pieces_dir, range(n_total))
            track["_clim_mean"] = clim_mean  # cache on the dict across repeated calls
        rand_pid = (piece_id + track.get("rand_offset", 20)) % n_total
        return predict_piece_baseline(
            track["baseline_kind"], pieces_dir, piece_id, track["lead_hours"],
            max_frames=max_frames, rand_piece_id=rand_pid, clim_mean=clim_mean,
        )
    raise ValueError(f"unknown track kind {kind!r}")
