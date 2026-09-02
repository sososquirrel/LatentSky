"""Conditional rectified flow from the deterministic forecast to truth.

The alternative to the SDEdit retrofit (see experiments/tstart_protocol): instead
of pretending the two-head forecast is partially-noised truth, learn the transport
directly. Source samples are x0 = z_forecast + sigma * eps (the noise keeps the
map generative — a deterministic pair-map would collapse to a regression; sigma
is set to the measured wet-masked forecast error in z-space). Target x1 = z_truth.
Straight-line interpolation x_t = (1 - t) * x0 + t * x1 with velocity target
v = x1 - x0; the network v(x_t, cond, t) reuses ConditionalDiffusionUNet (its
timestep embedding takes t * (T_EMB_RANGE - 1), mapping t in [0, 1] onto the
embedding range it was designed for).

Sampling: Heun (2nd order) ODE integration from a fresh x0 draw; an ensemble is
different eps draws. All fields in z = log1p(rain) / 2 space, as everywhere else.
"""
from __future__ import annotations

import torch

T_EMB_RANGE = 1000  # scale of the sinusoidal embedding the backbone expects


class RectifiedFlow:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def source(self, z_forecast: torch.Tensor, generator=None) -> torch.Tensor:
        noise = torch.randn(z_forecast.shape, device=z_forecast.device, generator=generator)
        return z_forecast + self.sigma * noise

    def training_pairs(self, z_forecast: torch.Tensor, z_truth: torch.Tensor):
        """Returns (x_t, t, v_target) for one batch: t ~ U(0,1) per sample."""
        x0 = self.source(z_forecast)
        t = torch.rand(z_forecast.shape[0], device=z_forecast.device)
        tb = t.view(-1, 1, 1, 1)
        x_t = (1 - tb) * x0 + tb * z_truth
        v = z_truth - x0
        return x_t, t, v

    @torch.no_grad()
    def sample(self, model, cond: torch.Tensor, z_forecast: torch.Tensor,
               n_steps: int = 32, generator=None) -> torch.Tensor:
        """Heun integration of dx/dt = v(x, cond, t) from t=0 to t=1."""
        x = self.source(z_forecast, generator=generator)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0 = torch.full((x.shape[0],), i * dt, device=x.device)
            v0 = model(x, cond, t0 * (T_EMB_RANGE - 1))
            x_pred = x + dt * v0
            if i + 1 < n_steps:
                t1 = torch.full((x.shape[0],), (i + 1) * dt, device=x.device)
                v1 = model(x_pred, cond, t1 * (T_EMB_RANGE - 1))
                x = x + dt * 0.5 * (v0 + v1)
            else:
                x = x_pred
        return x
