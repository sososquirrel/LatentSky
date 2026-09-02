"""Inject the real-example PNGs into model_reference.html as data URIs,
between the BEGIN_REAL_EXAMPLE / END_REAL_EXAMPLE markers (idempotent)."""
import base64
import re
from pathlib import Path

REPO = Path("/glade/u/home/sabramian/LatentSky")
ASSETS = REPO / "docs" / "assets_model_reference"
HTML = REPO / "docs" / "model_reference.html"

info = eval((ASSETS / "example_info.txt").read_text())

def uri(name):
    b = (ASSETS / name).read_bytes()
    print(f"{name}: {len(b)/1024:.0f} KB")
    return "data:image/png;base64," + base64.b64encode(b).decode()

block = f"""<!-- BEGIN_REAL_EXAMPLE -->
<p>The same chain on a real held-out case: the rainiest test block of the patch split
(piece {info['piece']}, frame {info['frame']}, block {info['block']} — a 128-px tile,
≈3600 km meridionally), lead 3 h, canonical checkpoints. First the deterministic side:
the rain field now (the persistence forecast), the two-head prediction three hours ahead,
and what actually happened.</p>
<figure style="margin:1.2rem 0">
<img src="{uri('example_context.png')}" alt="Three rain maps on a shared logarithmic mm scale: the current rain field used as the persistence forecast, the two-head forecast for three hours ahead, and the true field three hours ahead."
     style="width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px">
<figcaption style="font-size:.85rem;color:var(--muted);max-width:72ch">
Context (shared log color scale, mm). The two-head forecast tracks the observed
systems but is smoother and weaker at the peaks — {info['det_max']:.1f} mm against
{info['truth_max']:.1f} mm in truth: the conditional-mean signature the decomposition
quantifies.
</figcaption>
</figure>
<p>Then the diffusion refinement of that middle panel — the actual states of the
reverse chain, from the warm start at t = 100 down to the finished draw:</p>
<figure style="margin:1.2rem 0">
<img src="{uri('example_denoise.png')}" alt="Six rain maps showing the reverse diffusion chain: the noisy state at timestep 100, progressively denoised states at timesteps 60, 30 and 10, the finished draw at timestep 0, and the mean of three such draws."
     style="width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px">
<figcaption style="font-size:.85rem;color:var(--muted);max-width:72ch">
The denoising trajectory, anchored at the two-head forecast (first panel). Top row: the
actual chain state x_t — the t = 100 state is that same forecast mixed with
√(1−ᾱ₁₀₀) ≈ 17 % Gaussian noise (the log mm scale turns faint positive noise on dry
pixels into speckle; the band structure underneath is the forecast's). Bottom row: the
model's running estimate of the rain, x̂₀(t) — at t = 100 it is still nearly the
forecast, and it sharpens step by step into the final draw, which restores small-scale
texture and peak intensity the deterministic forecast smoothed away
(max {info['final_max']:.1f} mm against truth's {info['truth_max']:.1f} — sharp, but
stochastic). Averaging three draws (ensemble mean) trades some sharpness back for lower
variance.
</figcaption>
</figure>
<p>And the finished products side by side — what each forecast hands you for the same
hour:</p>
<figure style="margin:1.2rem 0">
<img src="{uri('example_compare.png')}" alt="Four rain maps on a shared logarithmic mm scale: the current rain field, the two-head forecast three hours ahead, the three-draw diffusion ensemble mean three hours ahead, and the true field three hours ahead."
     style="width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px">
<figcaption style="font-size:.85rem;color:var(--muted);max-width:72ch">
Persistence, the two forecasts, and truth. The two-head field is the smoothest; the
diffusion mean restores finer structure while inheriting the two-head's placement; the
remaining differences against truth — displaced cells, birth and decay the models
miss — are what the T(L) analysis measures scale by scale.
</figcaption>
</figure>
<!-- END_REAL_EXAMPLE -->"""

html = HTML.read_text()
new = re.sub(r"<!-- BEGIN_REAL_EXAMPLE -->.*?<!-- END_REAL_EXAMPLE -->", lambda m: block,
             html, flags=re.S)
assert new != html or "BEGIN_REAL_EXAMPLE" in new
HTML.write_text(new)
print("embedded; html size", len(new) / 1e6, "MB")
