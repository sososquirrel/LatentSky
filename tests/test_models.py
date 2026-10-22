"""Architecture tests + strict state_dict compatibility against the canonical
trained checkpoints (docs/FRAME.md Definitions), proving the recovery from
_archive/ is faithful to what was actually trained."""

from pathlib import Path

import pytest
import torch

from latentsky.models import UNet2D, UNet2DTwoHead, VelocityUNet

RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
ONEHEAD_CKPT = RUNS / "unet_models/one_head/10var/patchsplit_onehead_lead1h/checkpoint.pt"
TWOHEAD_CKPT = RUNS / "unet_models/two_head/10var/patchsplit_twohead_olr_fss_lead1h/checkpoint.pt"
FLOW_CKPT = RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt"


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


def test_velocity_unet_forward_shape():
    m = VelocityUNet(cond_ch=10, depth=2, base_ch=8, time_emb_dim=16)
    v = m(torch.randn(2, 1, 16, 16), torch.randn(2, 10, 16, 16), torch.tensor([3, 900]))
    assert v.shape == (2, 1, 16, 16)


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


@pytest.mark.skipif(not FLOW_CKPT.exists(), reason="canonical flow checkpoint not on scratch")
def test_velocity_unet_loads_canonical_flow_checkpoint_strict():
    m = VelocityUNet(cond_ch=10, base_ch=64, depth=5, time_emb_dim=128)
    state = torch.load(FLOW_CKPT, map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=True)
