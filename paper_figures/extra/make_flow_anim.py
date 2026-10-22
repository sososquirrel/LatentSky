"""Build the flow-matching training/inference animation page.

Renders, on the standard +3 h example block:
  TRAINING  — the crossfade x_t = (1-t)x0 + t*x1 at 21 values of t, beside
              the target velocity v = x1 - x0 (the same picture at every t).
  INFERENCE — the 32 Heun steps from a fresh sprinkle x0 to the final draw,
              beside the velocity the network reports at each step.

Frames are 128x128 PNGs embedded as data URIs in a self-contained page
written to the scratchpad. CPU, ~2 min (64 network evaluations).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path("/glade/u/home/sabramian/LatentSky")
plt.style.use(str(REPO / "docs" / "latentsky.mplstyle"))
from latentsky.layout import EXAMPLE_BLOCK as BLOCK  # noqa: E402
from latentsky.layout import EXAMPLE_FRAME as FRAME  # noqa: E402
from latentsky.layout import EXAMPLE_PIECE as PIECE  # noqa: E402
from latentsky.layout import INPUT_VARS, P, PIECES, RUNS, rain_from_z  # noqa: E402
from latentsky.layout import LOG_RAIN_SCALE as SCALE  # noqa: E402
from latentsky.models.flow import flow_config  # noqa: E402

SIGMA = flow_config(3)["sigma"]

from latentsky.models import ConditionalDiffusionUNet, UNet2DTwoHead  # noqa: E402
from latentsky.models.flow import T_EMB_RANGE, RectifiedFlow  # noqa: E402

OUT = Path("/glade/derecho/scratch/sabramian/tmp/claude-47666/"
           "-glade-u-home-sabramian-LatentSky/"
           "8968f095-fafb-4503-ad0d-2766a3e12c79/scratchpad")
NSTEPS = 32
SEED = 11


# ---------------------------------------------------------------- rendering
def png64(arr01, cmap):
    """arr01 in [0,1] (values < 0 fall to the cmap's 'under' color)."""
    buf = io.BytesIO()
    plt.imsave(buf, arr01, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower",
               format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    torch.set_num_threads(8)
    r, c = BLOCK
    sl = np.s_[r * P:(r + 1) * P, c * P:(c + 1) * P]
    x = np.stack([np.asarray(np.load(PIECES / f"{v}_piece{PIECE:02d}.npy",
                                     mmap_mode="r")[FRAME],
                             dtype=np.float32)[sl] for v in INPUT_VARS], 0)
    truth = np.asarray(np.load(PIECES / f"rain_total_piece{PIECE:02d}.npy",
                               mmap_mode="r")[FRAME + 13],
                       dtype=np.float32)[sl]
    cond = torch.from_numpy(x[None])

    twohead = UNet2DTwoHead(in_ch=10, base_ch=64, depth=5, olr_channel_idx=2)
    twohead.load_state_dict(torch.load(
        RUNS / "unet_models/two_head/10var/"
               "patchsplit_twohead_olr_fss_lead3h/checkpoint.pt",
        map_location="cpu", weights_only=True))
    twohead.eval()
    net = ConditionalDiffusionUNet(cond_ch=10, base_ch=64, depth=5,
                                   time_emb_dim=128)
    net.load_state_dict(torch.load(
        RUNS / "flow_models/flow_matching_lead3h.checkpoint.pt",
        map_location="cpu", weights_only=True))
    net.eval()
    rf = RectifiedFlow(sigma=SIGMA)

    with torch.no_grad():
        det_t, _ = twohead(cond)
        zf = torch.log1p(torch.clamp(det_t, min=0.0)) / SCALE
        z1 = torch.from_numpy(np.log1p(truth)[None, None] / SCALE)
        g = torch.Generator().manual_seed(SEED)
        x0 = rf.source(zf, generator=g)
        v_target = (z1 - x0)

        # --- inference: Heun, recording every state and velocity ----------
        xs, vs = [], []
        xc = x0.clone()
        dt = 1.0 / NSTEPS
        for i in range(NSTEPS):
            t0 = torch.full((xc.shape[0],), i * dt)
            v0 = net(xc, cond, t0 * (T_EMB_RANGE - 1))
            xs.append(xc.numpy()[0, 0].copy())
            vs.append(v0.numpy()[0, 0].copy())
            xp = xc + dt * v0
            if i + 1 < NSTEPS:
                t1 = torch.full((xc.shape[0],), (i + 1) * dt)
                v1 = net(xp, cond, t1 * (T_EMB_RANGE - 1))
                xc = xc + dt * 0.5 * (v0 + v1)
            else:
                xc = xp
        xs.append(xc.numpy()[0, 0].copy())          # the final draw, t = 1

    zf0 = zf.numpy()[0, 0]
    x0n = x0.numpy()[0, 0]
    z1n = z1.numpy()[0, 0]
    vt = v_target.numpy()[0, 0]


    # shared scales
    vmax = max(float(truth.max()), 1.0)
    lo, hi = np.log10(0.1), np.log10(vmax)
    vlim = float(np.abs(vt).max()) * 0.85

    rcmap = plt.get_cmap("Blues").copy()
    rcmap.set_under("#ffffff")
    vcmap = plt.get_cmap("RdBu_r").copy()

    def rain(zfield):
        f = rain_from_z(zfield)
        return png64((np.log10(np.clip(f, 1e-9, None)) - lo) / (hi - lo),
                     rcmap)

    def vel(vfield):
        return png64(np.clip((vfield + vlim) / (2 * vlim), 0, 1), vcmap)

    # --- training frames: the crossfade ---------------------------------
    ts = np.linspace(0, 1, 21)
    train_frames = [rain((1 - t) * x0n + t * z1n) for t in ts]
    target_png = vel(vt)

    # --- inference frames ------------------------------------------------
    infer_x = [rain(s) for s in xs]
    infer_v = [vel(s) for s in vs]

    ctx = {
        "forecast": rain(zf0),
        "x0": rain(x0n),
        "truth": rain(z1n),
    }

    peak_draw = float(rain_from_z(xs[-1]).max())
    peak_fc = float(rain_from_z(zf0).max())
    print(f"peaks: forecast {peak_fc:.1f} mm, draw {peak_draw:.1f} mm, "
          f"truth {truth.max():.1f} mm")

    html = build_html(ctx, train_frames, target_png, infer_x, infer_v,
                      ts, vmax, vlim, peak_fc, peak_draw, float(truth.max()))
    (OUT / "flow_anim.html").write_text(html)
    kb = len(html) / 1024
    print(f"written flow_anim.html ({kb:.0f} KB)")


def build_html(ctx, train, target, ix, iv, ts, vmax, vlim, pfc, pdr, ptr):
    import json
    return TEMPLATE.replace("__TRAIN__", json.dumps(train)) \
                   .replace("__TARGET__", json.dumps(target)) \
                   .replace("__IX__", json.dumps(ix)) \
                   .replace("__IV__", json.dumps(iv)) \
                   .replace("__TS__", json.dumps([round(float(t), 3) for t in ts])) \
                   .replace("__FORECAST__", json.dumps(ctx["forecast"])) \
                   .replace("__X0__", json.dumps(ctx["x0"])) \
                   .replace("__TRUTH__", json.dumps(ctx["truth"])) \
                   .replace("__VMAX__", f"{vmax:.1f}") \
                   .replace("__VLIM__", f"{vlim:.2f}") \
                   .replace("__PFC__", f"{pfc:.1f}") \
                   .replace("__PDR__", f"{pdr:.1f}") \
                   .replace("__PTR__", f"{ptr:.1f}")


TEMPLATE = r"""<title>Crossfade and Integrate</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,540;9..144,640&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f7f8f5; --ink:#1d2833; --muted:#5b6a76; --line:#d9dfd9;
  --panel:#ffffff; --truth:#2166ac; --flow:#c55a11; --sage:#6b7f6a;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#141a20; --ink:#e8ecef; --muted:#9aa8b2; --line:#2c3843;
    --panel:#1b232b; --truth:#6ba3d6; --flow:#e08a4a; --sage:#93a892;
  }
}
:root[data-theme="dark"]{
  --paper:#141a20; --ink:#e8ecef; --muted:#9aa8b2; --line:#2c3843;
  --panel:#1b232b; --truth:#6ba3d6; --flow:#e08a4a; --sage:#93a892;
}
body{background:var(--paper); color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.6;}
main{max-width:76ch; margin:0 auto; padding:3rem 1.2rem 5rem;}
h1{font-family:Fraunces,Georgia,serif; font-weight:640; font-size:2.4rem;
  line-height:1.12; margin:0 0 .5rem; text-wrap:balance;}
h2{font-family:Fraunces,Georgia,serif; font-weight:540; font-size:1.3rem;
  margin:0 0 .2rem;}
p{margin:0 0 1rem;}
.lede{font-size:1.1rem; color:var(--muted); margin-bottom:2.2rem;}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:.72rem;
  letter-spacing:.14em; text-transform:uppercase; color:var(--sage);
  margin-bottom:.8rem;}
.stage{background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:1.1rem 1.1rem 1rem; margin:1.1rem 0 2rem;}
.stage.train{border-left:3px solid var(--truth);}
.stage.infer{border-left:3px solid var(--flow);}
.row{display:grid; grid-template-columns:1fr 1fr; gap:1.1rem;
  align-items:start;}
.row3{display:grid; grid-template-columns:repeat(3,1fr); gap:1.1rem;}
figure{margin:0;}
img{width:100%; height:auto; display:block; border:1px solid var(--line);
  border-radius:3px; image-rendering:pixelated; background:#fff;}
figcaption{font-family:"IBM Plex Mono",monospace; font-size:.72rem;
  color:var(--muted); margin-top:.4rem; line-height:1.45;}
.controls{display:flex; align-items:center; gap:.7rem; margin-top:.9rem;
  font-family:"IBM Plex Mono",monospace; font-size:.78rem;
  flex-wrap:wrap;}
button{font:inherit; color:var(--ink); background:none;
  border:1px solid var(--line); border-radius:4px; padding:.3rem .75rem;
  cursor:pointer;}
button:hover{border-color:var(--flow);}
button:focus-visible{outline:2px solid var(--flow); outline-offset:2px;}
input[type=range]{flex:1; min-width:150px; accent-color:var(--flow);}
.readout{color:var(--muted); font-variant-numeric:tabular-nums;
  min-width:11ch; text-align:right;}
.note{border-left:3px solid var(--sage); padding:.1rem 0 .1rem .9rem;
  color:var(--muted); font-size:.95rem; margin:1.3rem 0;}
.tag{font-family:"IBM Plex Mono",monospace; font-size:.7rem;
  letter-spacing:.08em; text-transform:uppercase;}
.tag.t{color:var(--truth);} .tag.f{color:var(--flow);}
.scale{font-family:"IBM Plex Mono",monospace; font-size:.7rem;
  color:var(--muted); margin-top:.7rem;}
@media (max-width:640px){ .row,.row3{grid-template-columns:1fr;} }
</style>
<main>
<div class="eyebrow">LatentSky companion · the +3 h example block</div>
<h1>Crossfade and Integrate</h1>
<p class="lede">Flow matching has two lives: during <em>training</em> it
slides a noisy forecast into the truth and asks the network how fast each
pixel is moving; during <em>inference</em> it takes that answer and rides it
from a fresh sprinkle of noise to one sharp, plausible rain field. Both
lives, animated on the same block.</p>

<h2>The cast</h2>
<div class="row3" style="margin-top:.8rem">
  <figure><img src="__FORECAST__" alt="the deterministic forecast">
    <figcaption>the U-Net forecast — the conditional mean, peak __PFC__ mm</figcaption></figure>
  <figure><img src="__X0__" alt="the noisy source">
    <figcaption>x₀ = forecast + σ·ε, σ = 0.188 (the measured error)</figcaption></figure>
  <figure><img src="__TRUTH__" alt="the truth">
    <figcaption>x₁ — what actually happened, peak __PTR__ mm</figcaption></figure>
</div>

<div class="stage train">
  <span class="tag t">Training</span>
  <h2>Slide x₀ into the truth, freeze, ask</h2>
  <p style="font-size:.97rem">Pick a random moment <em>t</em> of the
  crossfade x<sub>t</sub> = (1−t)·x₀ + t·x₁, show the network only that
  frozen frame (plus the ten weather channels), and ask it for the velocity
  of every pixel. Because the crossfade is a straight line travelled at
  constant speed, the correct answer is the same picture at every t — which
  is why one network can serve the whole path.</p>
  <div class="row">
    <figure><img id="trainX" src="" alt="the crossfade state x_t">
      <figcaption>x<sub>t</sub> — the frozen frame the network sees</figcaption></figure>
    <figure><img src="__TARGET__" alt="the target velocity">
      <figcaption>v = x₁ − x₀ — the answer, identical at every t</figcaption></figure>
  </div>
  <div class="controls">
    <button id="trainPlay">▶ play</button>
    <input type="range" id="trainSlide" min="0" max="20" value="0">
    <span class="readout" id="trainOut">t = 0.00</span>
  </div>
</div>

<div class="stage infer">
  <span class="tag f">Inference</span>
  <h2>Ride the answer, 32 steps</h2>
  <p style="font-size:.97rem">Now the truth is unknown. Start from a fresh
  sprinkle x₀, ask the network for the velocity at the current state, step
  forward by v̂·Δt with Δt = 1/32, and repeat. The state on the left sharpens
  as it goes; the velocity on the right — the network's answer — barely
  changes, which is what makes the transport nearly straight and 32 steps
  plenty.</p>
  <div class="row">
    <figure><img id="inferX" src="" alt="the state during integration">
      <figcaption>x<sub>t</sub> — the field being transported</figcaption></figure>
    <figure><img id="inferV" src="" alt="the predicted velocity">
      <figcaption>v̂(x<sub>t</sub>, t, c) — the network's answer at this step</figcaption></figure>
  </div>
  <div class="controls">
    <button id="inferPlay">▶ play</button>
    <input type="range" id="inferSlide" min="0" max="32" value="0">
    <span class="readout" id="inferOut">step 0 · t = 0.00</span>
  </div>
  <div class="scale">rain: log scale, 0.1 → __VMAX__ mm · velocity:
  symmetric ±__VLIM__ (z units), red = rain appearing, blue = rain
  leaving</div>
</div>

<div class="note">Two things to notice. The velocity panel never sharpens ---
it is a conditional mean, and stays one; sharpness is produced by
<em>integrating</em> it from a random start, not by any single network
answer. And the final draw peaks at __PDR__ mm against the forecast's
__PFC__ mm, with the truth at __PTR__ mm: the ride restores the intensity
the average had flattened.</div>
</main>
<script>
(() => {
const TRAIN = __TRAIN__, TARGET = __TARGET__, IX = __IX__, IV = __IV__,
      TS = __TS__;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

function rig(imgs, slide, out, btn, n, fmt, speed){
  let i = 0, playing = false, timer = null;
  const show = k => { i = k; imgs(k); slide.value = k; out.textContent = fmt(k); };
  const tick = () => { show((i + 1) % (n + 1));
    if(i === n){ stop(); return; } };
  const start = () => { playing = true; btn.textContent = '❚❚ pause';
    if(i === n) show(0);
    timer = setInterval(tick, speed); };
  const stop = () => { playing = false; btn.textContent = '▶ play';
    clearInterval(timer); };
  btn.onclick = () => playing ? stop() : start();
  slide.oninput = () => { stop(); show(+slide.value); };
  show(0);
  return {start, stop};
}

const tX = document.getElementById('trainX');
rig(k => { tX.src = TRAIN[k]; },
    document.getElementById('trainSlide'),
    document.getElementById('trainOut'),
    document.getElementById('trainPlay'),
    TRAIN.length - 1,
    k => 't = ' + TS[k].toFixed(2), 260);

const iX = document.getElementById('inferX'), iV = document.getElementById('inferV');
rig(k => { iX.src = IX[k]; iV.src = IV[Math.min(k, IV.length - 1)]; },
    document.getElementById('inferSlide'),
    document.getElementById('inferOut'),
    document.getElementById('inferPlay'),
    IX.length - 1,
    k => 'step ' + k + ' · t = ' + (k / 32).toFixed(2), 170);
})();
</script>
"""


if __name__ == "__main__":
    main()
