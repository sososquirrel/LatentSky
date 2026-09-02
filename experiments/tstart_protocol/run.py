"""t_start protocol sweep — see HYPOTHESIS.md."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from latentsky.models import ConditionalDiffusionUNet, GaussianDiffusion, UNet2DTwoHead

REPO = Path("/glade/u/home/sabramian/LatentSky")
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
PIECES = Path("/glade/derecho/scratch/sabramian/latentsky_pieces")
OUT = Path(__file__).parent / "out"

INPUT_VARS = [
    "precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
    "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
    "umeridional_850hPa", "rain_log1p",
]
P = 128
LEAD_STEPS = 12  # 3 h
DELTA = LEAD_STEPS + 1
SCALE = 2.0
CLIP = (0.0, 2.5)
PIECE = 8
FRAMES = list(range(8, 96 - DELTA, 16))  # 6 frames
T_STARTS = [50, 100, 250, 500, 999]
KM_PER_PX = 27.8
SMALL_SCALE_KM = 1000.0
H_USE, W_USE = 256, 1408  # 2x11 tiles


def load_frame(t):
    chans = []
    for v in INPUT_VARS:
        a = np.load(PIECES / f"{v}_piece{PIECE:02d}.npy", mmap_mode="r")
        chans.append(np.asarray(a[t], dtype=np.float32))
    x = np.stack(chans, axis=0)
    y = np.asarray(
        np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy", mmap_mode="r")[t + DELTA],
        dtype=np.float32,
    )
    return x[:, :H_USE, :W_USE], y[:H_USE, :W_USE]


def tiles(x):
    """(C,H,W) -> (22,C,P,P) batch of non-overlapping tiles."""
    c = x.shape[0]
    t = x.reshape(c, 2, P, 11, P)
    return torch.from_numpy(np.ascontiguousarray(t.transpose(1, 3, 0, 2, 4).reshape(22, c, P, P)))


def mosaic(t):
    """(22,1,P,P) -> (H,W)"""
    a = t.reshape(2, 11, P, P).transpose(0, 2, 1, 3).reshape(H_USE, W_USE)
    return a


def refine(gd, model, cond, x0_init, t_start, seed):
    torch.manual_seed(seed)
    tb = torch.full((cond.shape[0],), t_start, device=cond.device, dtype=torch.long)
    x = gd.q_sample(x0_init, tb, torch.randn_like(x0_init))
    for tt in reversed(range(t_start + 1)):
        tb.fill_(tt)
        eps = model(x, cond, tb)
        ab = gd.alphas_cumprod[tt]
        x0p = torch.clamp((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab), *CLIP)
        mean = gd.posterior_mean_coef1[tt] * x0p + gd.posterior_mean_coef2[tt] * x
        x = mean + torch.sqrt(gd.posterior_variance[tt]) * torch.randn_like(x) if tt > 0 else mean
    return x


def radial_spectrum(field):
    f = np.fft.rfft2(field)
    p2 = np.abs(f) ** 2
    ky = np.fft.fftfreq(field.shape[0], d=KM_PER_PX)
    kx = np.fft.rfftfreq(field.shape[1], d=KM_PER_PX)
    kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kbins = np.linspace(kk[kk > 0].min(), kk.max(), 60)
    idx = np.digitize(kk.ravel(), kbins)
    ps = np.bincount(idx, weights=p2.ravel(), minlength=len(kbins) + 1)
    ns = np.bincount(idx, minlength=len(kbins) + 1)
    with np.errstate(invalid="ignore"):
        prof = ps / np.maximum(ns, 1)
    kmid = 0.5 * (kbins[:-1] + kbins[1:])
    return kmid, prof[1:len(kbins)]


def band_power(field, lam_max_km=SMALL_SCALE_KM, lam_min_km=2 * KM_PER_PX):
    k, p = radial_spectrum(field)
    sel = (k > 1.0 / lam_max_km) & (k < 1.0 / lam_min_km)
    return float(np.nansum(p[sel]))


def main():
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda")

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()
    diff = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    diff.load_state_dict(torch.load(
        RUNS / "gen_models/10var/diffusion_lead3h/checkpoint.pt",
        map_location=device, weights_only=True))
    diff = diff.to(device).eval()
    gd = GaussianDiffusion(1000, "cosine", device=device)

    def to_mm(z):
        return np.clip(np.expm1(SCALE * np.clip(z.cpu().numpy()[:, 0], 0.0, 2.5)), 0, None)

    fields = {}       # name -> list of mosaics over frames
    names = ["truth", "two-head"] + [f"ts{ts}" for ts in T_STARTS] + ["ts100_mean3"]
    for n in names:
        fields[n] = []

    with torch.no_grad():
        for fi, t in enumerate(FRAMES):
            x, y = load_frame(t)
            cond = tiles(x).to(device)
            det, _ = twohead(cond)
            x0_init = torch.log1p(torch.clamp(det, min=0.0)) / SCALE
            fields["truth"].append(y)
            fields["two-head"].append(mosaic(np.clip(det.cpu().numpy()[:, 0], 0, None)))
            for ts in T_STARTS:
                z = refine(gd, diff, cond, x0_init, ts, seed=1000 * fi + ts)
                fields[f"ts{ts}"].append(mosaic(to_mm(z)))
            zm = np.mean([to_mm(refine(gd, diff, cond, x0_init, 100, seed=7000 + 10 * fi + s))
                          for s in range(3)], axis=0)
            fields["ts100_mean3"].append(mosaic(zm))
            print(f"frame {t} done", flush=True)

    # metrics
    truth_band = np.mean([band_power(f) for f in fields["truth"]])
    rows = []
    for n in names:
        fl = fields[n]
        rows.append(dict(
            name=n,
            band_ratio=float(np.mean([band_power(f) for f in fl]) / truth_band),
            wet_frac=float(np.mean([(f > 0.1).mean() for f in fl])),
            p99=float(np.mean([np.percentile(f, 99) for f in fl])),
            fmax=float(np.mean([f.max() for f in fl])),
        ))
    import csv
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r, flush=True)

    # spectra figure
    plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for n, style in [("truth", dict(color="k", lw=2)), ("two-head", dict(color="#2166ac", ls="--"))] + \
                    [(f"ts{ts}", dict(alpha=0.85)) for ts in T_STARTS] + \
                    [("ts100_mean3", dict(color="#2166ac", ls=":", alpha=0.9))]:
        specs = [radial_spectrum(f) for f in fields[n]]
        k = specs[0][0]
        p = np.nanmean([s[1] for s in specs], axis=0)
        ax.loglog(1.0 / k, p, label=n, **style)
    ax.axvline(SMALL_SCALE_KM, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("wavelength (km)"); ax.set_ylabel("power")
    ax.set_title("lead 3 h, piece 8, 6 frames — single draws")
    ax.legend(fontsize=7)
    fig.savefig(OUT / "spectra_tstart.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # montage of one frame
    show = ["truth", "two-head", "ts100", "ts250", "ts500", "ts999"]
    vmax = max(fields["truth"][3].max(), 1.0)
    norm = LogNorm(vmin=0.1, vmax=vmax)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("#f7f9fb")
    fig, axes = plt.subplots(len(show), 1, figsize=(9.5, 1.9 * len(show)))
    for ax, n in zip(axes, show):
        im = ax.imshow(fields[n][3][:, :704], norm=norm, cmap=cmap, origin="lower",
                       interpolation="nearest", aspect="auto")
        ax.set_title(n, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01); cb.set_label("rain (mm)", fontsize=8)
    fig.savefig(OUT / "montage_tstart.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
