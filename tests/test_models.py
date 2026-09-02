"""Architecture tests + strict state_dict compatibility against the canonical
trained checkpoints (docs/FRAME.md Definitions), proving the recovery from
_archive/ is faithful to what was actually trained."""

from pathlib import Path

import pytest
import torch

from latentsky.models import (
    ConditionalDiffusionUNet,
    GaussianDiffusion,
    UNet2D,
    UNet2DTwoHead,
)

RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
ONEHEAD_CKPT = RUNS / "unet_models/one_head/10var/patchsplit_onehead_lead1h/checkpoint.pt"
TWOHEAD_CKPT = RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead1h/checkpoint.pt"
DIFFUSION_CKPT = RUNS / "gen_models/10var/diffusion_lead1h/checkpoint.pt"


def test_unet2d_forward_shape_and_nonnegative():
    m = UNet2D(in_ch=10, depth=3, base_ch=8)
    y = m(torch.randn(2, 10, 32, 32))
    assert y.shape == (2, 1, 32, 32)
    assert (y >= 0).all()  # softplus head: rain intensity is non-negative


def test_unet2d_twohead_forward_shapes():
    m = UNet2DTwoHead(in_ch=10, depth=3, base_ch=8)
    rain, logits = m(torch.randn(2, 10, 32, 32))
    assert rain.shape == (2, 1, 32, 32)
    assert logits.shape == (2, 1, 32, 32)
    assert (rain >= 0).all()


def test_diffusion_unet_forward_shape():
    m = ConditionalDiffusionUNet(cond_ch=10, depth=2, base_ch=8, time_emb_dim=16)
    eps = m(torch.randn(2, 1, 16, 16), torch.randn(2, 10, 16, 16), torch.tensor([3, 900]))
    assert eps.shape == (2, 1, 16, 16)


def test_q_sample_is_near_identity_at_t0():
    gd = GaussianDiffusion(timesteps=1000)
    x0 = torch.randn(4, 1, 8, 8)
    xt = gd.q_sample(x0, torch.zeros(4, dtype=torch.long), torch.randn_like(x0))
    # alpha_bar_0 ~ 1 under the cosine schedule: almost no noise at t=0
    assert torch.allclose(xt, x0, atol=0.1)


def test_ddim_timesteps_descend_to_zero():
    gd = GaussianDiffusion(timesteps=1000)
    ts = gd.ddim_timesteps(50)
    assert ts[0] == 999 and ts[-1] == 0
    assert all(a > b for a, b in zip(ts, ts[1:]))


def test_ddpm_refine_sample_shape_and_clip():
    torch.manual_seed(0)
    gd = GaussianDiffusion(timesteps=1000)
    model = ConditionalDiffusionUNet(cond_ch=3, depth=2, base_ch=8, time_emb_dim=16)
    model.eval()
    cond = torch.randn(1, 3, 16, 16)
    x0_init = torch.rand(1, 1, 16, 16)
    out = gd.ddpm_refine_sample(model, cond, x0_init, t_start=5, clip_range=(0.0, 2.5))
    assert out.shape == (1, 1, 16, 16)
    assert out.min() >= -1.0  # last step is a posterior mean, near the clipped x0


@pytest.mark.skipif(not ONEHEAD_CKPT.exists(), reason="canonical one-head checkpoint not on scratch")
def test_unet2d_loads_canonical_onehead_checkpoint_strict():
    m = UNet2D(in_ch=10, out_ch=1, base_ch=64, depth=5, out_activation="softplus")
    state = torch.load(ONEHEAD_CKPT, map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=True)


@pytest.mark.skipif(not TWOHEAD_CKPT.exists(), reason="canonical two-head checkpoint not on scratch")
def test_twohead_loads_canonical_checkpoint_strict():
    m = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, out_activation="softplus", olr_channel_idx=2)
    state = torch.load(TWOHEAD_CKPT, map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=True)


@pytest.mark.skipif(not DIFFUSION_CKPT.exists(), reason="canonical diffusion checkpoint not on scratch")
def test_diffusion_unet_loads_canonical_checkpoint_strict():
    m = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    state = torch.load(DIFFUSION_CKPT, map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=True)
