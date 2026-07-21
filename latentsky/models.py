"""U-Net used across all rain experiments.

Cherry-picked from tzhou/LatentSky/260206_newtrain.py. Activation is ReLU
(matches all recent 1472_* experiments); switching to Softplus is a one-line
change and can be made config-driven later if needed.
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.05, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        base_ch: int = 64,
        depth: int = 4,
        dropout_p: float = 0.05,
        groups: int = 8,
        out_activation: str = "softplus",
    ):
        super().__init__()
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            outc = base_ch * (2 ** d)
            self.enc_convs.append(DoubleConv(ch, outc, dropout_p, groups))
            self.pools.append(nn.MaxPool2d(2))
            ch = outc

        self.bottleneck = DoubleConv(ch, ch * 2, dropout_p, groups)
        ch = ch * 2

        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            outc = base_ch * (2 ** d)
            self.upconvs.append(nn.ConvTranspose2d(ch, outc, kernel_size=2, stride=2))
            self.dec_convs.append(DoubleConv(outc * 2, outc, dropout_p, groups))
            ch = outc

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        if out_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0, threshold=20.0)
        elif out_activation == "identity":
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {out_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for conv, pool in zip(self.enc_convs, self.pools):
            x = conv(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, conv in zip(self.upconvs, self.dec_convs):
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = conv(x)
        return self.out_act(self.out_conv(x))


class UNet2DTwoHead(nn.Module):
    """Same encoder/bottleneck/decoder topology as UNet2D, plus:

    - OLR-FiLM conditioning at the bottleneck: a per-patch [mean, min] OLR
      summary (from the raw input channel at `olr_channel_idx`) is passed
      through a small MLP to produce (gamma, beta) that affine-modulate the
      bottleneck feature map. Lets the model condition on convective regime
      (cold/deep vs. warm/shallow) without a separate classifier network.
    - Two parallel 1x1-conv heads off the final decoder feature instead of
      one: `rain_head` (intensity regression, same activation as UNet2D's
      single head) and `wetdry_head` (raw logits, no activation -- meant for
      BCEWithLogitsLoss). forward() returns (rain_pred, wetdry_logit).
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int = 64,
        depth: int = 4,
        dropout_p: float = 0.05,
        groups: int = 8,
        out_activation: str = "softplus",
        olr_channel_idx: int = 2,
    ):
        super().__init__()
        self.olr_channel_idx = olr_channel_idx
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            outc = base_ch * (2 ** d)
            self.enc_convs.append(DoubleConv(ch, outc, dropout_p, groups))
            self.pools.append(nn.MaxPool2d(2))
            ch = outc

        self.bottleneck = DoubleConv(ch, ch * 2, dropout_p, groups)
        ch = ch * 2
        bottleneck_ch = ch

        self.film_gen = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * bottleneck_ch),
        )

        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            outc = base_ch * (2 ** d)
            self.upconvs.append(nn.ConvTranspose2d(ch, outc, kernel_size=2, stride=2))
            self.dec_convs.append(DoubleConv(outc * 2, outc, dropout_p, groups))
            ch = outc

        self.rain_head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.wetdry_head = nn.Conv2d(base_ch, 1, kernel_size=1)
        if out_activation == "softplus":
            self.rain_act = nn.Softplus(beta=1.0, threshold=20.0)
        elif out_activation == "identity":
            self.rain_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {out_activation}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        olr = x[:, self.olr_channel_idx : self.olr_channel_idx + 1]
        olr_summary = torch.cat([olr.mean(dim=(2, 3)), olr.amin(dim=(2, 3))], dim=1)  # (B,2)

        skips = []
        h = x
        for conv, pool in zip(self.enc_convs, self.pools):
            h = conv(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)

        gamma, beta = self.film_gen(olr_summary).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        for up, conv in zip(self.upconvs, self.dec_convs):
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = conv(h)

        return self.rain_act(self.rain_head(h)), self.wetdry_head(h)


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
