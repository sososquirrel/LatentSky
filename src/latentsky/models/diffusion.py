"""Conditional DDPM: denoiser architecture + schedule/sampling machinery
(recovered from _archive/latentsky/{models.py,diffusion.py}, 2026-09-02).

Role per docs/FRAME.md Definitions: the canonical diffusion model for R3 is
ConditionalDiffusionUNet on the same backbone family as UNet2D, sampled with
ddpm_refine_sample warm-started from the one-head regression prediction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .unet import DoubleConv


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of the diffusion
    timestep t (integer, 0..T-1) into a `dim`-length vector, followed by a
    small MLP -- the usual DDPM time-conditioning input."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConditionalDiffusionUNet(nn.Module):
    """Denoising network for a conditional DDPM (see latentsky/diffusion.py).

    Input: the noisy target field x_t (1 channel) concatenated with the
    conditioning fields (the same `cond_ch` met variables used everywhere
    else in this project) -> cond_ch+1 input channels. Timestep is injected
    at every encoder/decoder/bottleneck level via FiLM (affine modulation
    from a projected sinusoidal time embedding) -- same mechanism already
    used for OLR conditioning in UNet2DTwoHead, just applied at every scale
    instead of only the bottleneck, since here it needs to control the
    denoising strength throughout the network, not just gate on one signal.

    Output: single-channel predicted noise (same shape as x_t), no
    activation (noise is unconstrained, unlike rain intensity).
    """

    def __init__(
        self,
        cond_ch: int,
        base_ch: int = 64,
        depth: int = 4,
        dropout_p: float = 0.05,
        groups: int = 8,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        time_ch = time_emb_dim * 4

        self.enc_convs = nn.ModuleList()
        self.enc_film = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = cond_ch + 1
        for d in range(depth):
            outc = base_ch * (2 ** d)
            self.enc_convs.append(DoubleConv(ch, outc, dropout_p, groups))
            self.enc_film.append(nn.Linear(time_ch, 2 * outc))
            self.pools.append(nn.MaxPool2d(2))
            ch = outc

        self.bottleneck = DoubleConv(ch, ch * 2, dropout_p, groups)
        self.bottleneck_film = nn.Linear(time_ch, 2 * ch * 2)
        ch = ch * 2

        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_film = nn.ModuleList()
        for d in reversed(range(depth)):
            outc = base_ch * (2 ** d)
            self.upconvs.append(nn.ConvTranspose2d(ch, outc, kernel_size=2, stride=2))
            self.dec_convs.append(DoubleConv(outc * 2, outc, dropout_p, groups))
            self.dec_film.append(nn.Linear(time_ch, 2 * outc))
            ch = outc

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    @staticmethod
    def _film(h: torch.Tensor, film_layer: nn.Linear, temb: torch.Tensor) -> torch.Tensor:
        gamma, beta = film_layer(temb).chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t)
        h = torch.cat([x_t, cond], dim=1)

        skips = []
        for conv, film, pool in zip(self.enc_convs, self.enc_film, self.pools):
            h = conv(h)
            h = self._film(h, film, temb)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)
        h = self._film(h, self.bottleneck_film, temb)

        for up, conv, film in zip(self.upconvs, self.dec_convs, self.dec_film):
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = conv(h)
            h = self._film(h, film, temb)

        return self.out_conv(h)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021) cosine schedule -- better-behaved than linear
    at both ends (less aggressive near t=0, doesn't fully saturate near T)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class GaussianDiffusion:
    """Holds the precomputed schedule and the two operations training needs:
    q_sample (forward noising) and the simplified epsilon-prediction loss
    target. All buffers are 1-D, length `timesteps`."""

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device=None):
        if schedule != "cosine":
            raise ValueError(f"Unknown schedule: {schedule}")
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.timesteps = timesteps
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # Posterior q(x_{t-1} | x_t, x0) mean coefficients and variance
        # (Ho et al. 2020, eq. 6-7) -- used by ddpm_sample. Dividing by
        # alphas_t (single-step, stays close to 1) rather than
        # sqrt(alphas_cumprod_t) (vanishes near t=T) is what makes this
        # numerically stable where the DDIM x0-parameterization isn't.
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.
        t: (B,) int64 timestep indices, one per sample in the batch."""
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_omac * noise

    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def ddim_timesteps(self, num_steps: int):
        """Evenly-spaced subset of [0, T-1], descending (e.g. [999,...,0])
        -- the strided schedule DDIM samples along instead of every step."""
        idx = torch.linspace(0, self.timesteps - 1, num_steps).round().long()
        idx = torch.unique(idx)
        return idx.flip(0).tolist()

    @torch.no_grad()
    def ddim_sample(
        self, model, cond: torch.Tensor, num_steps: int = 50,
        clip_range: tuple[float, float] = (0.0, 2.5),
    ) -> torch.Tensor:
        """Deterministic DDIM sampling (eta=0): iteratively denoise from pure
        Gaussian noise to an x0 estimate, conditioned on `cond`. Far fewer
        forward passes than full ancestral DDPM sampling (num_steps vs
        self.timesteps) while remaining faithful to the trained model.
        Returns x0 in the same (scaled log1p-rain) space training used --
        caller is responsible for inverting the transform.

        `clip_range`: the model's x0 estimate is clamped to this range at
        EVERY step (standard DDPM/DDIM practice) -- early steps (still
        mostly noise) can produce wildly inaccurate x0 estimates, and
        without clamping those errors compound across steps and diverge.
        x0 = log1p(rain)/scale is never negative by construction (rain>=0),
        so 0 is a hard physical lower bound, not just a numerical safety
        net. Default upper bound 2.5 (with scale=2.0, i.e. rain up to
        expm1(5)~=147mm) is set from the actual training data's x0
        distribution (observed max ~1.6 for a 22mm rain event), generous
        for rare extremes without allowing runaway divergence.
        """
        device = cond.device
        B, _, H, W = cond.shape
        x = torch.randn(B, 1, H, W, device=device)
        timesteps = self.ddim_timesteps(num_steps)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = model(x, cond, t_batch)
            alpha_bar_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, *clip_range)
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps
            else:
                x = x0_pred
        return x

    @torch.no_grad()
    def ddpm_sample(
        self, model, cond: torch.Tensor,
        clip_range: tuple[float, float] = (0.0, 2.5),
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Full ancestral DDPM sampling (Ho et al. 2020): iterates over
        EVERY timestep t=T-1..0 (self.timesteps forward passes, no striding
        like ddim_sample). Slower than DDIM, but numerically stable -- the
        posterior-mean update divides by sqrt(alpha_t) (single-step, stays
        near 1) rather than sqrt(alpha_bar_t) (vanishes near t=T), which is
        what made ddim_sample's x0-parameterization blow up at high noise
        levels (~20000x error amplification near t=999 was diagnosed via
        scripts/debug_diffusion_patch_scale.py). x0 is still estimated and
        clipped each step (same physical justification as ddim_sample), but
        the NEXT sample is drawn from the clipped-x0 posterior directly,
        not recombined with the raw (unclipped-x0-implied) epsilon.

        `noise_scale`: multiplies the per-step stochastic noise (0 = fully
        deterministic ancestral sampling, using only the posterior mean).
        The x0 lower clip at 0 is a hard floor but not a ceiling, so 1000
        steps of full-scale noise injection can drift the sample upward on
        average (excursions below 0 get truncated back up, excursions
        above pass through freely) -- reducing/zeroing this checks whether
        that's the source of the systematic wet bias seen at noise_scale=1.
        """
        device = cond.device
        B, _, H, W = cond.shape
        x = torch.randn(B, 1, H, W, device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = model(x, cond, t_batch)
            alpha_bar_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, *clip_range)

            mean = self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x
            if t > 0 and noise_scale > 0:
                noise = torch.randn_like(x)
                x = mean + noise_scale * torch.sqrt(self.posterior_variance[t]) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddpm_refine_sample(
        self, model, cond: torch.Tensor, x0_init: torch.Tensor, t_start: int,
        clip_range: tuple[float, float] = (0.0, 2.5), noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """SDEdit-style warm-started sampling ('residual/refinement
        diffusion'): rather than starting the reverse chain from pure
        Gaussian noise at t=timesteps-1 (ddpm_sample), forward-noise a
        given point estimate `x0_init` (e.g. a regression model's
        prediction, already in the scaled-log1p x0 space) out to `t_start`
        via q_sample, then run the SAME posterior-mean reverse loop as
        ddpm_sample but only for t=t_start..0.

        This directly targets the failure mode found in ddpm_sample on
        held-out test blocks: starting from x0_init instead of pure noise
        means the "is it raining, roughly how much" structure comes from
        the (independently-validated, accurate) regression model rather
        than something the diffusion chain has to hallucinate; and cutting
        the chain from `timesteps` steps down to `t_start+1` steps directly
        bounds how much compounding drift the reverse process can
        accumulate before returning an answer. Requires no retraining --
        the network was trained to predict noise at an arbitrary timestep
        given ANY x_t, not specifically ones reached from pure noise, so
        the existing checkpoint applies unchanged.
        """
        device = cond.device
        B = x0_init.shape[0]
        t_batch0 = torch.full((B,), t_start, device=device, dtype=torch.long)
        init_noise = torch.randn_like(x0_init)
        x = self.q_sample(x0_init, t_batch0, init_noise)

        for t in reversed(range(t_start + 1)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = model(x, cond, t_batch)
            alpha_bar_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, *clip_range)

            mean = self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x
            if t > 0 and noise_scale > 0:
                noise = torch.randn_like(x)
                x = mean + noise_scale * torch.sqrt(self.posterior_variance[t]) * noise
            else:
                x = mean
        return x
