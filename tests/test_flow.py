import torch

from latentsky.models import ConditionalDiffusionUNet
from latentsky.models.flow import RectifiedFlow


def test_training_pairs_shapes_and_interpolation_limits():
    rf = RectifiedFlow(sigma=0.0)  # deterministic source for exactness
    zf = torch.rand(4, 1, 16, 16)
    zt = torch.rand(4, 1, 16, 16)
    x_t, t, v = rf.training_pairs(zf, zt)
    assert x_t.shape == zf.shape and t.shape == (4,) and v.shape == zf.shape
    assert torch.allclose(v, zt - zf)
    # x_t is on the straight line between source and target
    tb = t.view(-1, 1, 1, 1)
    assert torch.allclose(x_t, (1 - tb) * zf + tb * zt, atol=1e-6)


def test_sample_with_perfect_velocity_reaches_truth():
    zf = torch.rand(2, 1, 16, 16)
    zt = torch.rand(2, 1, 16, 16)
    rf = RectifiedFlow(sigma=0.0)

    class Oracle(torch.nn.Module):
        def forward(self, x, cond, t):
            return zt - zf  # the exact constant velocity of the linear path

    out = rf.sample(Oracle(), cond=torch.zeros(2, 10, 16, 16), z_forecast=zf, n_steps=8)
    assert torch.allclose(out, zt, atol=1e-5)


def test_sample_runs_with_real_backbone_and_noise():
    torch.manual_seed(0)
    rf = RectifiedFlow(sigma=0.2)
    net = ConditionalDiffusionUNet(cond_ch=3, base_ch=8, depth=2, time_emb_dim=16)
    net.eval()
    out = rf.sample(net, torch.randn(1, 3, 16, 16), torch.rand(1, 1, 16, 16), n_steps=4)
    assert out.shape == (1, 1, 16, 16)
    # ensemble: different generators give different draws
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    o1 = rf.sample(net, torch.zeros(1, 3, 16, 16), torch.zeros(1, 1, 16, 16), 4, g1)
    o2 = rf.sample(net, torch.zeros(1, 3, 16, 16), torch.zeros(1, 1, 16, 16), 4, g2)
    assert not torch.allclose(o1, o2)


def test_sample_path_ends_at_sample():
    torch.manual_seed(0)
    net = ConditionalDiffusionUNet(cond_ch=10, base_ch=8, depth=2, time_emb_dim=16).eval()
    zf = torch.rand(1, 1, 16, 16)
    cond = torch.rand(1, 10, 16, 16)
    rf = RectifiedFlow(sigma=0.1)
    x1 = rf.sample(net, cond, zf, n_steps=4, generator=torch.Generator().manual_seed(3))
    path = rf.sample_path(net, cond, zf, n_steps=4, snap_at=(0.0, 0.5, 1.0),
                          generator=torch.Generator().manual_seed(3))
    assert set(path) == {0.0, 0.5, 1.0}
    assert torch.allclose(path[1.0], x1)


def test_flow_config_reads_experiment():
    from latentsky.models.flow import flow_config
    c3 = flow_config(3)
    assert c3["sigma"] == 0.1876 and c3["n_steps"] == 32 and c3["target_offset"] == 13
    assert flow_config(6)["sigma"] == 0.204
