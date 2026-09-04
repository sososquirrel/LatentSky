"""Illustrate the core intuition of rectified flow matching: source and target
populations in a shared PCA phase space, with a few actual ODE-integration
trajectories drawn between them.

Rectified flow is trained on straight lines x_t = (1-t)x0 + t*x1 between PAIRED
(source, target) points, with velocity target v = x1 - x0. But the network only
sees (x_t, t), never which x1 a given x0 is "supposed to" reach -- so the learned
v(x, cond, t) is the AVERAGE direction over all training pairs passing near
(x, t), not any single pair's exact line. At inference, integrating that learned
field from a NEW x0 traces a path that approximates the marginal transport from
the source distribution to the target distribution as a whole, even though no
single integrated path is exactly the straight line its own x0 was ever paired
with in training. That's the piece of intuition this figure exists to show.

Method: gather many (two-head forecast, truth) pairs from held-out test blocks
at lead 6h, flatten each 128x128 z-space field to a vector, fit one PCA basis on
the pooled forecast+truth population, and project everything into its top two
components. For a handful of examples, run the actual Heun ODE integration
(latentsky.models.flow.RectifiedFlow.sample's exact loop, instrumented to keep
every intermediate state) and project the whole trajectory into the same basis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow, T_EMB_RANGE
from latentsky.training.patch_split import compute_patch_grid, make_patch_split

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"
sys.path.insert(0, str(REPO / "docs"))
from make_model_reference_assets import INPUT_VARS  # noqa: E402

P = 128
DELTA6 = 25
SCALE, SIGMA, NSTEPS = 2.0, 0.204, 32
N_CLOUD = 160          # (piece, frame, block) examples for the background clouds
N_TRAJ = 6             # examples that also get a full recorded trajectory
PIECES_USED = [1, 8, 15, 20, 26, 33]
FRAME_STRIDE = 11


def patch_test_blocks():
    nrows, ncols = compute_patch_grid(320, 1440, P)
    _, _, test_b = make_patch_split(40, nrows, ncols, 1919810, 0.7, 0.15)
    return test_b, nrows, ncols


@torch.no_grad()
def heun_trajectory(rf: RectifiedFlow, model, cond, z_forecast, generator=None):
    """Same math as RectifiedFlow.sample, instrumented to return every state
    x_0, x_1, ..., x_{n_steps} (n_steps+1 snapshots) instead of only the last."""
    x = rf.source(z_forecast, generator=generator)
    states = [x.clone()]
    dt = 1.0 / NSTEPS
    for i in range(NSTEPS):
        t0 = torch.full((x.shape[0],), i * dt, device=x.device)
        v0 = model(x, cond, t0 * (T_EMB_RANGE - 1))
        x_pred = x + dt * v0
        if i + 1 < NSTEPS:
            t1 = torch.full((x.shape[0],), (i + 1) * dt, device=x.device)
            v1 = model(x_pred, cond, t1 * (T_EMB_RANGE - 1))
            x = x + dt * 0.5 * (v0 + v1)
        else:
            x = x_pred
        states.append(x.clone())
    return states


def main():
    device = torch.device("cpu")
    rng = np.random.default_rng(0)

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead6h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()
    flow_net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    flow_net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead6h.checkpoint.pt",
        map_location=device, weights_only=True))
    flow_net = flow_net.to(device).eval()
    rf = RectifiedFlow(sigma=SIGMA)

    test_b, nrows, ncols = patch_test_blocks()

    # ---- gather the background cloud: many (forecast, truth) pairs ----
    z_forecasts, z_truths, wets = [], [], []
    for pid in PIECES_USED:
        rain = np.load(PIECES / f"rain_total_piece{pid:02d}.npy", mmap_mode="r")
        chans = {v: np.load(PIECES / f"{v}_piece{pid:02d}.npy", mmap_mode="r") for v in INPUT_VARS}
        blocks = sorted((r, c) for (p, r, c) in test_b if p == pid)
        for t in range(0, 96 - DELTA6, FRAME_STRIDE):
            for (r, c) in blocks:
                sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
                x = np.stack([np.asarray(chans[v][t], dtype=np.float32)[sl] for v in INPUT_VARS])
                cond = torch.from_numpy(x[None]).to(device)
                with torch.no_grad():
                    det, _ = twohead(cond)
                z_fc = (torch.log1p(torch.clamp(det, min=0.0)) / SCALE).numpy()[0, 0]
                y = np.asarray(rain[t + DELTA6], dtype=np.float32)[sl]
                z_tr = np.log1p(np.clip(y, 0, None)) / SCALE
                z_forecasts.append(z_fc.ravel())
                z_truths.append(z_tr.ravel())
                wets.append(float((y > 0.1).mean()))
                if len(z_forecasts) >= N_CLOUD:
                    break
            if len(z_forecasts) >= N_CLOUD:
                break
        if len(z_forecasts) >= N_CLOUD:
            break
    z_forecasts = np.array(z_forecasts)
    z_truths = np.array(z_truths)
    wets = np.array(wets)
    print(f"cloud: {len(z_forecasts)} examples, wet range {wets.min():.3f}-{wets.max():.3f}")
    np.savez_compressed(OUT / "flow_pca_cloud_cache.npz",
                         z_forecasts=z_forecasts, z_truths=z_truths, wets=wets)

    # ---- PCA basis fit on TRUTH ONLY: the question this figure asks is "how
    # far is the source from the real data manifold, in that manifold's own
    # coordinates" -- fitting on the pooled forecast+truth cloud instead would
    # let the forecast's systematic blur (lower pixel-to-pixel variance, not a
    # physically meaningful difference) distort the shared basis. ----
    pca = PCA(n_components=2, random_state=0)
    pca.fit(z_truths)
    fc_2d = pca.transform(z_forecasts)
    tr_2d = pca.transform(z_truths)
    print("explained variance ratio (truth-only basis):", pca.explained_variance_ratio_)

    fc_norm = np.linalg.norm(z_forecasts, axis=1)
    tr_norm = np.linalg.norm(z_truths, axis=1)
    tr_wet = wets  # truth wet fraction, already computed
    for name, pc_vals, other in [("PC1", fc_2d[:, 0], None), ("PC2", fc_2d[:, 1], None)]:
        r_wet = np.corrcoef(pc_vals, tr_wet)[0, 1]
        r_norm = np.corrcoef(pc_vals, fc_norm)[0, 1]
        print(f"forecast {name} vs truth-wet-fraction: r={r_wet:.3f}   vs forecast-vector-norm: r={r_norm:.3f}")
    for name, pc_vals in [("PC1", tr_2d[:, 0]), ("PC2", tr_2d[:, 1])]:
        r_wet = np.corrcoef(pc_vals, tr_wet)[0, 1]
        r_norm = np.corrcoef(pc_vals, tr_norm)[0, 1]
        print(f"truth {name} vs truth-wet-fraction: r={r_wet:.3f}   vs truth-vector-norm: r={r_norm:.3f}")

    # ---- pick examples for full trajectories: mostly wet ones (visible
    # transport), plus one near-dry case for contrast (near-zero movement) ----
    order = np.argsort(wets)
    idx_traj = np.concatenate([order[-(N_TRAJ - 1):], order[:1]])

    # ---- re-run those examples with cond kept, to get full trajectories ----
    traj_2d_list = []
    ii = 0
    for pid in PIECES_USED:
        chans = {v: np.load(PIECES / f"{v}_piece{pid:02d}.npy", mmap_mode="r") for v in INPUT_VARS}
        blocks = sorted((r, c) for (p, r, c) in test_b if p == pid)
        for t in range(0, 96 - DELTA6, FRAME_STRIDE):
            for (r, c) in blocks:
                if ii in idx_traj:
                    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
                    x = np.stack([np.asarray(chans[v][t], dtype=np.float32)[sl] for v in INPUT_VARS])
                    cond = torch.from_numpy(x[None]).to(device)
                    with torch.no_grad():
                        det, _ = twohead(cond)
                    z_fc = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
                    g = torch.Generator(device=device).manual_seed(42)
                    states = heun_trajectory(rf, flow_net, cond, z_fc, generator=g)
                    traj = np.stack([s.numpy()[0, 0].ravel() for s in states])
                    traj_2d_list.append(pca.transform(traj))
                ii += 1
                if ii > idx_traj.max():
                    break
            if ii > idx_traj.max():
                break
        if ii > idx_traj.max():
            break

    # ---- figure ----
    plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.scatter(fc_2d[:, 0], fc_2d[:, 1], s=22, color="#2166ac", alpha=0.45,
               label=r"source $x_0$ = two-head forecast + $\sigma\epsilon$", zorder=2)
    ax.scatter(tr_2d[:, 0], tr_2d[:, 1], s=22, color="#1a1a1a", alpha=0.45,
               marker="^", label=r"target $x_1$ = truth", zorder=2)

    cmap = plt.get_cmap("YlOrBr")
    for k, traj in enumerate(traj_2d_list):
        c = cmap(0.35 + 0.55 * k / max(len(traj_2d_list) - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=1.8, zorder=4, alpha=0.9)
        ax.annotate("", xy=traj[-1], xytext=traj[-2],
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
        ax.scatter(*traj[0], marker="o", s=45, color=c, edgecolor="k", linewidths=0.6, zorder=6)
        ax.scatter(*traj[-1], marker="s", s=50, color=c, edgecolor="k", linewidths=0.6, zorder=6)
        # the straight line this x0's OWN training target would have been,
        # for comparison -- not available at inference (needs true x1), shown
        # only descriptively for the nearest truth point in this projection
        nearest = tr_2d[np.argmin(np.sum((tr_2d - traj[-1]) ** 2, axis=1))]
        ax.plot([traj[0, 0], nearest[0]], [traj[0, 1], nearest[1]], ":",
                 color=c, lw=1.0, alpha=0.6, zorder=3)

    ax.scatter([], [], marker="o", s=45, color="0.4", edgecolor="k", label="trajectory start ($t{=}0$)")
    ax.scatter([], [], marker="s", s=50, color="0.4", edgecolor="k", label="trajectory end ($t{=}1$, 32 Heun steps)")
    ax.plot([], [], ":", color="0.4", lw=1.0, label="nearest truth point (illustrative straight line)")

    ax.set_xlabel(f"PC1 of truth ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 of truth ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.set_title("Rectified flow toward the truth manifold (lead 6h, held-out test blocks)\n"
                 "PCA basis fit on truth only; forecast/source/trajectories projected into it",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "flow_pca_phase_space.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'flow_pca_phase_space.png'}")


if __name__ == "__main__":
    main()
