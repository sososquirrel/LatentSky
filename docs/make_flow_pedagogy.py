"""Three teaching figures for the flow-matching section of model_reference.html
(replacing the PCA phase-space figure):

  A. flow_schematic_lines.png  — 2D toy geometry: straight training lines between
     paired (source, target) points, the kernel-averaged velocity field they imply,
     and one genuinely integrated path from a new source point.
  B. flow_training_pair.png    — one real training pair at +3 h on the standard
     example block: forecast → source → midpoint x_t → truth, and below them the
     velocity target v = x1 − x0 next to the network's actual prediction at t = 0.5.
  C. flow_sampling_path.png    — the real 32-step Heun integration, states shown
     at t = 0, 0.25, 0.5, 0.75, 1 (inverted to rain), with truth for reference.

CPU, ~1 min. Same example case as everywhere: piece 8, frame 64, block (1,4), 3 h."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from latentsky.models import VelocityUNet, UNet2DTwoHead
from latentsky.layout import rain_from_z
from latentsky.models.flow import T_EMB_RANGE, RectifiedFlow

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = REPO / "docs" / "assets_model_reference"
INPUT_VARS = ["precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
              "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
              "umeridional_850hPa", "rain_log1p"]
P, PIECE, FRAME, BLOCK = 128, 8, 64, (1, 4)
SCALE, SIGMA, NSTEPS = 2.0, 0.1876, 32

plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))


def toy_schematic():
    rng = np.random.default_rng(11)
    n = 150
    th = rng.uniform(0.15 * np.pi, 0.85 * np.pi, n)
    r = rng.normal(1.0, 0.04, n)
    x1 = np.stack([r * np.cos(th), r * np.sin(th)], 1)          # truth manifold
    centroid = np.array([0.0, 0.35])
    fc = 0.5 * x1 + 0.5 * centroid                               # "smoothed" forecast
    x0 = fc + rng.normal(0, 0.05, fc.shape)                      # source = fc + sigma*eps

    def vfield(x, t, h=0.12):
        xt = (1 - t) * x0 + t * x1
        w = np.exp(-((xt - x) ** 2).sum(1) / (2 * h * h))
        w /= w.sum() + 1e-12
        return (w[:, None] * (x1 - x0)).sum(0)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    sub = rng.choice(n, 45, replace=False)
    for i in sub:
        ax.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], "-", color="0.78",
                lw=0.7, zorder=1)
    ax.plot(x0[:, 0], x0[:, 1], "o", ms=3.2, color="#2166ac", alpha=0.55,
            mec="none", label="sources  x₀ = forecast + σ·ε", zorder=2)
    ax.plot(x1[:, 0], x1[:, 1], "o", ms=3.2, color="0.15", alpha=0.65,
            mec="none", label="targets  x₁ = truth", zorder=2)

    # integrate one NEW source point through the average field (Heun)
    start = np.array([-0.28, 0.52])
    path = [start.copy()]
    x = start.copy()
    ns = 40
    for i in range(ns):
        t0 = i / ns
        v0 = vfield(x, t0)
        xp = x + v0 / ns
        v1 = vfield(xp, (i + 1) / ns)
        x = x + 0.5 * (v0 + v1) / ns
        path.append(x.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-", color="#b05e0a", lw=2.8, zorder=4,
            label="integrated path of a NEW source (Heun)")
    ax.plot(*start, "*", ms=16, color="#b05e0a", mec="0.1", mew=0.8, zorder=5)
    ax.annotate("", xy=path[-1], xytext=path[-4],
                arrowprops=dict(arrowstyle="-|>", color="#b05e0a", lw=2.4), zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_ylim(0.05, 1.28)
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=1)
    ax.set_title("training supervises straight lines between pairs;\n"
                 "sampling rides their local average to the truth manifold",
                 fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "flow_schematic_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved flow_schematic_lines.png")


def real_figures():
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME], dtype=np.float32)[sl]
                  for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13], dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = VelocityUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
        map_location="cpu", weights_only=True))
    net.eval()
    rf = RectifiedFlow(sigma=SIGMA)


    with torch.no_grad():
        det_t, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        z1 = torch.from_numpy(np.log1p(truth)[None, None] / SCALE)
        g = torch.Generator().manual_seed(7)
        x0 = rf.source(zf, generator=g)
        t_mid = 0.5
        x_t = (1 - t_mid) * x0 + t_mid * z1
        v_target = (z1 - x0).numpy()[0, 0]
        tt = torch.full((1,), t_mid)
        v_pred = net(x_t, cond, tt * (T_EMB_RANGE - 1)).numpy()[0, 0]

        # --- B: one training pair
        rain_panels = [("two-head forecast", rain_from_z(zf.numpy()[0, 0])),
                       ("source  x₀ = forecast + σ·ε", rain_from_z(x0.numpy()[0, 0])),
                       ("midpoint  x_t  (t = 0.5)", rain_from_z(x_t.numpy()[0, 0])),
                       ("truth  x₁", truth)]
        vmax = max(truth.max(), 1.0)
        norm = LogNorm(vmin=0.1, vmax=vmax)
        cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")
        fig, axs = plt.subplots(2, 4, figsize=(12.6, 6.6))
        for ax, (ttl, f) in zip(axs[0], rain_panels):
            im = ax.imshow(f, origin="lower", norm=norm, cmap=cmap,
                           interpolation="nearest")
            ax.set_title(ttl, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axs[0], shrink=0.85, pad=0.012).set_label(
            "rain (mm)", fontsize=8)
        vlim = float(np.abs(v_target).max()) * 0.85
        vpan = [("velocity target  v = x₁ − x₀", v_target),
                ("network prediction  v(x_t, cond, t)", v_pred),
                ("difference (MSE loss)", v_target - v_pred)]
        for ax, (ttl, f) in zip(axs[1, :3], vpan):
            imv = ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                            interpolation="nearest")
            ax.set_title(ttl, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        axs[1, 3].axis("off")
        fig.colorbar(imv, ax=axs[1], shrink=0.85, pad=0.012).set_label(
            "velocity (z units)", fontsize=8)
        fig.savefig(OUT / "flow_training_pair.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        rmse = float(np.sqrt(np.mean((v_target - v_pred) ** 2)))
        print(f"saved flow_training_pair.png  (v rms {np.sqrt((v_target**2).mean()):.3f},"
              f" residual rms {rmse:.3f})")

        # --- C: the sampling path (Heun, capture intermediate states)
        g = torch.Generator().manual_seed(7)
        xs = rf.source(zf, generator=g)
        snaps = {0.0: xs.numpy()[0, 0].copy()}
        dt = 1.0 / NSTEPS
        xcur = xs
        for i in range(NSTEPS):
            t0v = torch.full((1,), i * dt)
            v0 = net(xcur, cond, t0v * (T_EMB_RANGE - 1))
            x_pred = xcur + dt * v0
            if i + 1 < NSTEPS:
                t1v = torch.full((1,), (i + 1) * dt)
                v1 = net(x_pred, cond, t1v * (T_EMB_RANGE - 1))
                xcur = xcur + dt * 0.5 * (v0 + v1)
            else:
                xcur = x_pred
            tnext = (i + 1) * dt
            for mark in (0.25, 0.5, 0.75, 1.0):
                if abs(tnext - mark) < dt / 2:
                    snaps[mark] = xcur.numpy()[0, 0].copy()
        fig, axs = plt.subplots(1, 6, figsize=(15.2, 3.0))
        for ax, tmark in zip(axs[:5], sorted(snaps)):
            im = ax.imshow(rain_from_z(snaps[tmark]), origin="lower", norm=norm,
                           cmap=cmap, interpolation="nearest")
            ax.set_title(f"t = {tmark:g}" + ("  (source)" if tmark == 0 else
                                             "  (final draw)" if tmark == 1 else ""),
                         fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        axs[5].imshow(truth, origin="lower", norm=norm, cmap=cmap,
                      interpolation="nearest")
        axs[5].set_title("truth", fontsize=8.5, fontweight="bold")
        axs[5].set_xticks([]); axs[5].set_yticks([])
        fig.colorbar(im, ax=axs, shrink=0.8, pad=0.008).set_label("rain (mm)",
                                                                  fontsize=8)
        fig.savefig(OUT / "flow_sampling_path.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("saved flow_sampling_path.png  draw max",
              round(float(rain_from_z(snaps[1.0]).max()), 1), "truth max",
              round(float(truth.max()), 1))





def pixel_trajectories():
    """Replacement for the 2-D toy: real pixels' z-values along the actual
    32-step integration — each pixel rides a near-straight ramp whose slope is
    the network's predicted correction; ends land near truth."""
    import torch
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME], dtype=np.float32)[sl]
                  for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13], dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])
    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = VelocityUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
        map_location="cpu", weights_only=True))
    net.eval()
    rf = RectifiedFlow(sigma=SIGMA)
    z1 = np.log1p(truth) / SCALE

    with torch.no_grad():
        det_t, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        g = torch.Generator().manual_seed(7)
        xcur = rf.source(zf, generator=g)
        states = [xcur.numpy()[0, 0].copy()]
        dt = 1.0 / NSTEPS
        for i in range(NSTEPS):
            t0v = torch.full((1,), i * dt)
            v0 = net(xcur, cond, t0v * (T_EMB_RANGE - 1))
            x_pred = xcur + dt * v0
            if i + 1 < NSTEPS:
                t1v = torch.full((1,), (i + 1) * dt)
                v1 = net(x_pred, cond, t1v * (T_EMB_RANGE - 1))
                xcur = xcur + dt * 0.5 * (v0 + v1)
            else:
                xcur = x_pred
            states.append(xcur.numpy()[0, 0].copy())
    traj = np.stack(states)                      # (33, 128, 128)
    zf0 = zf.numpy()[0, 0]
    err = z1 - zf0

    # pick pixels: strong under-forecast, strong over-forecast, well-forecast wet, dry
    wet = (truth > 0.5) & (np.expm1(SCALE * zf0) > 0.05)
    picks = []
    m = np.where(wet & (err > 0), err, -np.inf); picks.append(
        ("forecast too weak", np.unravel_index(np.argmax(m), m.shape), "#b2182b"))
    m = np.where(wet & (err < 0), -err, -np.inf); picks.append(
        ("forecast too strong", np.unravel_index(np.argmax(m), m.shape), "#2166ac"))
    m = np.where(wet, -np.abs(err), -np.inf); picks.append(
        ("forecast about right", np.unravel_index(np.argmax(m), m.shape), "#4daf4a"))
    dry = (truth < 0.01) & (np.expm1(SCALE * zf0) < 0.01)
    ij = np.argwhere(dry)[137]
    picks.append(("dry (only noise to remove)", tuple(ij), "0.45"))

    ts = np.linspace(0, 1, NSTEPS + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for name, (pi, pj), colr in picks:
        ax.plot(ts, traj[:, pi, pj], "-", color=colr, lw=2.0, label=name)
        ax.plot(0, traj[0, pi, pj], "o", color=colr, ms=5)
        ax.plot([1, 1], [traj[-1, pi, pj], z1[pi, pj]], ":", color=colr, lw=1.1)
        ax.plot(1, z1[pi, pj], "*", color=colr, ms=13, mec="0.15", mew=0.6)
        ax.plot(0, zf0[pi, pj], "s", color=colr, ms=5, mfc="none", mew=1.4)
    ax.set_xlabel("integration time t")
    ax.set_ylabel("pixel value in z = log1p(rain)/2")
    ax.set_title("four real pixels riding the flow (+3 h, 32 Heun steps)\n"
                 "□ forecast · ● source (forecast + σ·ε) · ★ truth", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    fig.savefig(OUT / "flow_pixel_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for name, (pi, pj), _ in picks:
        print(f"{name}: forecast z {zf0[pi,pj]:.3f} source {traj[0,pi,pj]:.3f} "
              f"final {traj[-1,pi,pj]:.3f} truth {z1[pi,pj]:.3f}")
    print("saved flow_pixel_trajectories.png")


if __name__ == "__main__":
    real_figures()
    pixel_trajectories()
