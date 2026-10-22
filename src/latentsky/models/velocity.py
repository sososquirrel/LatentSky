"""Velocity network of the rectified flow (latentsky.models.flow).

A U-Net with skip connections: the encoder sees the current flow field x_s
stacked with the ten input fields (the condition); the flow time s enters every
encoder, bottleneck and decoder level through FiLM (an affine modulation from a
sinusoidal time embedding); the output is one field, the velocity.

Recovered from the diffusion backbone the project first used (archived with the
DDPM in _archive/ddpm_2026-09/); the layer names are unchanged, so every trained
flow checkpoint loads into it strictly.
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


class VelocityUNet(nn.Module):
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
