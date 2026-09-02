"""Radially-averaged power spectrum, truth vs diffusion prediction, one line
per lead time -- the decisive diagnostic for whether diffusion's texture is
real (spectrum matches truth, POD failure is a placement problem) or fake
(dissipative rollout: small-scale power bleeds away with lead, large scales
hold -- the "blurry" failure mode in frequency space).

Reads the already-saved full-field npz dumps from
evaluate_run_systems_diffusion(..., save_fields_dir=...) -- no new diffusion
sampling needed. Currently only the "oneheadbackbone" (plain-regression
warm-start) variant has these saved, across all 6 leads
(/glade/derecho/scratch/${USER}/LatentSky_diffusion_fields/oneheadbackbone).

Per piece/frame: 2D FFT of the rain field (zero-mean removed, no window --
domain is a fixed rectangular tile, not periodic, so edge leakage exists but
is the same for truth and prediction and mostly affects the lowest
wavenumbers, not the small-scale falloff this diagnostic cares about),
radially-averaged into wavenumber bins (assuming square-ish pixels; the
domain is 320x1440 -- rectangular, so radial bins use min(H,W)/2 as the
Nyquist cutoff and elliptical (kx,ky) radii are still pooled by their
Euclidean magnitude, a standard approximation for non-square domains).
Power spectra from all pieces/frames at a given lead are averaged together
(in log-power space would bias toward low values; averaged in linear power,
consistent with how "spectral density" is usually reported).
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentsky.paths import FIGURES_DIR

_LEADS = ["0.25h", "0.5h", "1h", "2h", "3h", "6h"]
GRID_KM = 25.0  # ERA5-derived grid spacing used throughout this project's scale analysis

EXTRA_TRACKS = [
    dict(label="regression pred", color="tab:purple",
         dir="/glade/derecho/scratch/sabramian/LatentSky_regression_fields/onehead",
         glob_tmpl="patchsplit_onehead_lead{lead}h_piece*.npz"),
    dict(label="two-head pred", color="tab:blue",
         dir="/glade/derecho/scratch/sabramian/LatentSky_regression_fields/two_head",
         glob_tmpl="patchsplit_twohead_olr_fss_lead{lead}h_piece*.npz"),
]


def _radial_bin_index(H: int, W: int) -> tuple[np.ndarray, int]:
    """Precompute, once per (H,W), the integer radial-wavenumber bin index
    for every FFT cell -- reused via np.bincount across every frame instead
    of a per-bin boolean-mask loop (which is O(kmax * H*W) per frame and
    was the actual bottleneck: ~40 pieces * 16 frames * 2 fields * 6 leads
    each paying that cost independently)."""
    kx = np.fft.fftfreq(W)
    ky = np.fft.fftfreq(H)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2) * min(H, W)
    kmax = min(H, W) // 2
    idx = np.clip(np.round(K).astype(int), 0, kmax)
    return idx, kmax


def radial_power(field: np.ndarray, bin_idx: np.ndarray, kmax: int) -> np.ndarray:
    """2D FFT power spectrum of a single (H,W) field, radially averaged
    into integer-wavenumber bins via bin_idx (from _radial_bin_index).
    Returns power for k=1..kmax-1 (drops k=0, the mean)."""
    H, W = field.shape
    f = field - field.mean()
    F = np.fft.fft2(f)
    P = (np.abs(F) ** 2 / (H * W)).ravel()
    sums = np.bincount(bin_idx.ravel(), weights=P, minlength=kmax + 1)
    counts = np.bincount(bin_idx.ravel(), minlength=kmax + 1)
    power = sums / np.maximum(counts, 1)
    return power[1:kmax]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fields-dir", default="/glade/derecho/scratch/sabramian/LatentSky_diffusion_fields/oneheadbackbone")
    p.add_argument("--tag", default="oneheadbackbone")
    p.add_argument("--out", default=str(FIGURES_DIR / "power_spectrum_diffusion_vs_truth.png"))
    p.add_argument("--max-pieces", type=int, default=None,
                    help="cap pieces read per lead -- each npz is ~30-38MB compressed; "
                         "this GPFS scratch dir has been slow to read in full (240 files, "
                         "~8GB total), so bound it for a fast first pass.")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")

    for ax, lead in zip(axes.ravel(), _LEADS):
        files = sorted(glob.glob(f"{args.fields_dir}/diffusion_lead{lead}_piece*.npz"))
        if args.max_pieces is not None:
            files = files[:args.max_pieces]
        if not files:
            ax.set_title(f"lead={lead} (no data)")
            continue
        truth_powers, pred_powers = [], []
        bin_idx = kmax = k_bins_ref = None
        for fpath in files:
            d = np.load(fpath)
            Y, Yp = d["Y"], d["Yp"]  # (T,1,H,W)
            H, W = Y.shape[-2:]
            if bin_idx is None:
                bin_idx, kmax = _radial_bin_index(H, W)
                k_bins_ref = np.arange(1, kmax)
            for t in range(Y.shape[0]):
                truth_powers.append(radial_power(Y[t, 0], bin_idx, kmax))
                pred_powers.append(radial_power(Yp[t, 0], bin_idx, kmax))
        truth_mean = np.nanmean(np.stack(truth_powers), axis=0)
        pred_mean = np.nanmean(np.stack(pred_powers), axis=0)

        # convert cycles/domain-pixel to physical wavelength (km) for the x-axis
        wavelength_km = (max(Y.shape[-2], Y.shape[-1]) * GRID_KM) / k_bins_ref

        ax.loglog(wavelength_km, truth_mean, "-", color="k", lw=2, label="truth")
        ax.loglog(wavelength_km, pred_mean, "--", color="tab:red", lw=2, label="diffusion pred")

        for extra in EXTRA_TRACKS:
            lead_num = lead.rstrip("h")
            efiles = sorted(glob.glob(f"{extra['dir']}/{extra['glob_tmpl'].format(lead=lead_num)}"))
            if args.max_pieces is not None:
                efiles = efiles[:args.max_pieces]
            if not efiles:
                continue
            epowers = []
            for fpath in efiles:
                d = np.load(fpath)
                Yp_e = d["Yp"]
                for t in range(Yp_e.shape[0]):
                    epowers.append(radial_power(Yp_e[t, 0], bin_idx, kmax))
            emean = np.nanmean(np.stack(epowers), axis=0)
            ax.loglog(wavelength_km, emean, "--", color=extra["color"], lw=1.6, alpha=0.85,
                      label=extra["label"])

        ax.set_title(f"lead={lead}  (n={len(truth_powers)} frames)", fontsize=10.5)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, frameon=False)

    for ax in axes[-1]:
        ax.set_xlabel("wavelength (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("power")
    fig.suptitle("Radially-averaged power spectrum, truth vs diffusion prediction, by lead\n"
                 f"({args.tag} warm-start; small scale/short wavelength on the LEFT, large scale on the RIGHT)", y=1.02)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
