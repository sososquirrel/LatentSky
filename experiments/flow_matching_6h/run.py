"""Train the conditional rectified flow at lead 6 h, then evaluate single draws
against the SDEdit ceiling. See HYPOTHESIS.md."""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead
from latentsky.models.flow import RectifiedFlow
from latentsky.training import (
    build_filtered_patch_loader, blocks_for_piece, compute_patch_grid,
    load_piece_arrays, make_patch_split,
)

HERE = Path(__file__).parent
RUNS = Path("/glade/derecho/scratch/sabramian/LatentSky_runs")
INPUT_VARS = [
    "precipw", "t2m", "olrtoa", "mse_500hPa", "mse_850hPa",
    "uzonal_500hPa", "uzonal_850hPa", "umeridional_500hPa",
    "umeridional_850hPa", "rain_log1p",
]
SCALE = 2.0

sys.path.insert(0, str(HERE.parent / "tstart_protocol"))
from run import band_power, mosaic, tiles  # noqa: E402


def z_of(rain_mm: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(rain_mm, min=0.0)) / SCALE


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    device = torch.device("cuda")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    delta = cfg["lead_steps_15"] + 1

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / f"unet_models/two_head/10var/{cfg['warm_start_run']}/checkpoint.pt",
        map_location=device, weights_only=True))
    twohead = twohead.to(device).eval()
    for p in twohead.parameters():
        p.requires_grad_(False)

    net = ConditionalDiffusionUNet(
        cond_ch=10, base_ch=cfg["base_ch"], depth=cfg["depth"],
        time_emb_dim=cfg["time_emb_dim"]).to(device)
    rf = RectifiedFlow(sigma=cfg["sigma"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg["lr_step"], gamma=cfg["lr_gamma"])

    probe = np.load(f"{cfg['pieces_dir']}/{INPUT_VARS[0]}_piece00.npy", mmap_mode="r")
    _, H, W = probe.shape
    nrows, ncols = compute_patch_grid(H, W, cfg["patch_size"])
    train_b, val_b, _ = make_patch_split(
        cfg["n_total_pieces"], nrows, ncols, cfg["patch_split_seed"],
        cfg["patch_split_train_frac"], cfg["patch_split_val_frac"])
    print(f"split: {len(train_b)} train / {len(val_b)} val blocks", flush=True)

    def batch_terms(x, y):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            det, _ = twohead(x)
        zf, zt = z_of(det), z_of(y)
        x_t, t, v = rf.training_pairs(zf, zt)
        v_pred = net(x_t, x, t * 999.0)
        return F.mse_loss(v_pred, v)

    best_val, patience_ctr = float("inf"), 0
    hist = {"train": [], "val": []}
    ckpt = HERE / "checkpoint.pt"
    for epoch in range(cfg["num_epochs"]):
        order = list(range(cfg["n_total_pieces"]))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        tr_sum = tr_n = va_sum = va_n = 0
        for pid in order:
            tb, vb = blocks_for_piece(train_b, pid), blocks_for_piece(val_b, pid)
            if not tb and not vb:
                continue
            X, Y = load_piece_arrays(cfg["pieces_dir"], pid, INPUT_VARS, "rain_total", delta)
            if tb:
                net.train()
                loader = build_filtered_patch_loader(X, Y, cfg["patch_size"],
                                                     cfg["batch_size"], True, tb)
                for x, y in loader or []:
                    opt.zero_grad(set_to_none=True)
                    loss = batch_terms(x, y)
                    loss.backward(); opt.step()
                    tr_sum += float(loss.item()); tr_n += 1
            if vb:
                net.eval()
                loader = build_filtered_patch_loader(X, Y, cfg["patch_size"],
                                                     cfg["batch_size"], False, vb)
                with torch.no_grad():
                    for x, y in loader or []:
                        va_sum += float(batch_terms(x, y).item()); va_n += 1
            del X, Y
        sched.step()
        tr, va = tr_sum / max(tr_n, 1), va_sum / max(va_n, 1)
        hist["train"].append(tr); hist["val"].append(va)
        print(f"epoch {epoch+1}: train={tr:.5f} val={va:.5f} "
              f"{datetime.now().isoformat(timespec='seconds')}", flush=True)
        if va < best_val:
            best_val, patience_ctr = va, 0
            torch.save(net.state_dict(), ckpt)
            print("  ** saved", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print("  early stop", flush=True)
                break

    json.dump({"config": cfg, "loss_history": hist, "best_val": best_val,
               "finished_at": datetime.now().isoformat(timespec="seconds")},
              open(HERE / "manifest.json", "w"), indent=2)

    # ---- evaluation against the SDEdit ceiling ----
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.eval()
    bands, wets, truth_bands = [], [], []
    with torch.no_grad():
        for fi, t in enumerate(cfg["eval_frames"]):
            chans = [np.asarray(np.load(f"{cfg['pieces_dir']}/{v}_piece{cfg['eval_piece']:02d}.npy",
                                        mmap_mode="r")[t], dtype=np.float32) for v in INPUT_VARS]
            x = np.stack(chans, 0)[:, :256, :1408]
            y = np.asarray(np.load(f"{cfg['pieces_dir']}/rain_total_piece{cfg['eval_piece']:02d}.npy",
                                   mmap_mode="r")[t + delta], dtype=np.float32)[:256, :1408]
            cond = tiles(x).to(device)
            det, _ = twohead(cond)
            g = torch.Generator(device=device).manual_seed(100 + fi)
            z = rf.sample(net, cond, z_of(det), n_steps=cfg["eval_n_steps"], generator=g)
            mm = mosaic(np.clip(np.expm1(SCALE * np.clip(z.cpu().numpy()[:, 0], 0, 2.5)), 0, None))
            bands.append(band_power(mm)); wets.append(float((mm > 0.1).mean()))
            truth_bands.append(band_power(y))
            print(f"eval frame {t}: ratio={bands[-1]/truth_bands[-1]:.3f}", flush=True)
    ratio = float(np.mean(bands) / np.mean(truth_bands))
    wet = float(np.mean(wets))
    twet = float(np.mean([( np.asarray(np.load(f"{cfg['pieces_dir']}/rain_total_piece{cfg['eval_piece']:02d}.npy", mmap_mode='r')[t + delta], dtype=np.float32)[:256,:1408] > 0.1).mean() for t in cfg["eval_frames"]]))
    print(f"FLOW 6h: band_ratio={ratio:.3f} wet={wet:.4f} truth_wet={twet:.4f} "
          f"(SDEdit ceiling: 0.80-0.85, wet -20%)", flush=True)


if __name__ == "__main__":
    main()
