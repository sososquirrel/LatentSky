"""Figure: schematic of latentsky.models.ConditionalDiffusionUNet (the
denoising network behind the diffusion_lead* runs), drawn in the same
Ronneberger-style U-Net diagram as fig17_architecture_schematic.py (the
plain UNet2D / 9var mse+uv reference), so the two are directly comparable
side by side.

Reuses fig17's bar/arrow drawing conventions exactly, but reflects the
three real architectural differences of the conditional diffusion model:

  1. Input is the noisy target x_t (1 channel) concatenated with the same
     met conditioning fields (cond_ch=10) -> 11 input channels, not 9/10.
  2. Timestep t is embedded (Sinusoidal(128) -> Linear-SiLU-Linear -> 512-d)
     and injected via FiLM (h = h*(1+gamma)+beta) at EVERY encoder level,
     the bottleneck, AND every decoder level -- not just the bottleneck
     (as in the two-head model's OLR-FiLM) and not absent (as in plain
     UNet2D). Drawn as a purple "temb rail" across the top with an arrow
     down into each level's second conv bar.
  3. Output is a single-channel predicted noise field with NO activation
     (noise is unconstrained), not "Softplus (>=0)".

Exact channel/spatial progression and parameter count computed the same
way ConditionalDiffusionUNet.__init__ builds its encoder/decoder (base_ch=64,
depth=5, cond_ch=10, time_emb_dim=128 -- from configs/diffusion_lead*.yaml)
-- not hand-typed.
"""
from __future__ import annotations
import math
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"
PAPER_DIR = REPO / "figures/paper"

COND_CH = 10          # configs/diffusion_lead*.yaml input_vars count
IN_CH = COND_CH + 1   # + noisy x_t
OUT_CH = 1
BASE_CH = 64
DEPTH = 5
TIME_EMB_DIM = 128
PATCH = 64            # schematic patch size -- matches fig17's convention
                        # (a generic illustrative size, not literal training
                        # patch_size=128, kept identical so the two figures'
                        # bottleneck depth/scale read the same)

# exact count from instantiating latentsky.models.ConditionalDiffusionUNet(
# cond_ch=10, base_ch=64, depth=5, time_emb_dim=128) -- see module docstring
N_PARAMS = 130_879_937

CONV_COLOR = "#E8974A"
CONV_EDGE = "#B36A1E"
POOL_COLOR = "#B33F3F"
DECONV_COLOR = "#3F7D3F"
SKIP_COLOR = "0.55"
CONV_ARROW = "#3B6FA8"
FILM_COLOR = "#8E44AD"   # purple -- timestep-conditioning FiLM, new vs fig17

def bar_h(spatial):
    return 1.1 + 1.55 * math.log2(spatial)

BAR_W = 0.16

fig, ax = plt.subplots(figsize=(21, 11.5))
ax.set_xlim(-1, 28)
ax.set_ylim(-1, 11.5)
ax.axis("off")
ax.set_aspect("equal")


def vbar(x, base_y, h, color, edge, hollow_frac=0.0):
    if hollow_frac > 0:
        hh = h * hollow_frac
        ax.add_patch(Rectangle((x, base_y), BAR_W, hh, facecolor="white",
                               edgecolor=edge, lw=1.3, zorder=3))
        ax.add_patch(Rectangle((x, base_y + hh), BAR_W, h - hh, facecolor=color,
                               edgecolor=edge, lw=1.3, zorder=3))
    else:
        ax.add_patch(Rectangle((x, base_y), BAR_W, h, facecolor=color,
                               edgecolor=edge, lw=1.3, zorder=3))


def small_arrow(x0, y, x1, color=CONV_ARROW):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
               arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                               mutation_scale=11), zorder=4)


def big_vert_arrow(x, y0, y1, color):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
               arrowprops=dict(arrowstyle="-|>", color=color, lw=3.2,
                               mutation_scale=16), zorder=4)


def skip_arrow(x0, y, x1):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
               arrowprops=dict(arrowstyle="-|>", color=SKIP_COLOR, lw=2.2,
                               mutation_scale=13), zorder=1)


def film_arrow(x, y_top, y_bot):
    """Downward purple arrow from the temb rail into a level's 2nd conv
    bar -- represents FiLM(t) modulation applied right after that conv."""
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
               arrowprops=dict(arrowstyle="-|>", color=FILM_COLOR, lw=1.6,
                               mutation_scale=10, linestyle=(0, (3, 2))),
               zorder=6)


def label_above(x, h, base_y, text, fontsize=8.5):
    ax.text(x + BAR_W / 2, base_y + h + 0.15, text, ha="center", va="bottom",
           fontsize=fontsize, fontweight="bold")


def label_spatial(x, base_y, text):
    ax.text(x, base_y - 0.28, text, ha="center", va="top", fontsize=8.5, color="0.3")


# ---- geometry: encoder levels 0..DEPTH-1, bottleneck, decoder levels DEPTH-1..0 ----
enc_ch = [BASE_CH * (2 ** d) for d in range(DEPTH)]
enc_sp = [PATCH // (2 ** d) for d in range(DEPTH)]
bott_ch = enc_ch[-1] * 2
bott_sp = PATCH // (2 ** DEPTH)

x = 0.0
enc_bar_x = []
enc_base_y = []
film_targets = []  # (x_center, y_top) of every level's FiLM-modulated bar, for the temb rail
y_top = 8.0

h0 = bar_h(enc_sp[0])
in_base_y = y_top - h0
vbar(x, in_base_y, h0, "0.4", "0.2")
label_above(x, h0, in_base_y, f"{IN_CH}")
ax.text(x - 0.35, in_base_y + h0 / 2, f"{PATCH}×{PATCH}", ha="right", va="center",
       fontsize=9, rotation=90)
x += BAR_W

cur_base_y = in_base_y
for d in range(DEPTH):
    h = bar_h(enc_sp[d])
    base_y = y_top - h
    x += 0.5
    small_arrow(x - 0.3, base_y + h / 2, x)
    vbar(x, base_y, h, CONV_COLOR, CONV_EDGE)
    label_above(x, h, base_y, f"{enc_ch[d]}")
    x += BAR_W
    x += 0.5
    small_arrow(x - 0.3, base_y + h / 2, x)
    vbar(x, base_y, h, CONV_COLOR, CONV_EDGE)
    label_above(x, h, base_y, f"{enc_ch[d]}")
    enc_bar_x.append(x)
    enc_base_y.append(base_y)
    film_targets.append((x + BAR_W / 2, base_y + h))
    label_spatial(x + BAR_W / 2, base_y, f"{enc_sp[d]}×{enc_sp[d]}")
    x += BAR_W

    drop = 1.35
    x += 0.55
    big_vert_arrow(x - 0.3, base_y + h / 2 - 0.05, base_y + h / 2 - 0.05 - drop, POOL_COLOR)
    y_top = y_top - drop

# Bottleneck
hb = bar_h(bott_sp)
base_yb = y_top - hb
x += 0.5
small_arrow(x - 0.3, base_yb + hb / 2, x)
vbar(x, base_yb, hb, CONV_COLOR, CONV_EDGE)
label_above(x, hb, base_yb, f"{bott_ch}")
x += BAR_W
x += 0.5
small_arrow(x - 0.3, base_yb + hb / 2, x)
vbar(x, base_yb, hb, CONV_COLOR, CONV_EDGE)
label_above(x, hb, base_yb, f"{bott_ch}")
label_spatial(x + BAR_W / 2, base_yb, f"{bott_sp}×{bott_sp}")
film_targets.append((x + BAR_W / 2, base_yb + hb))
bott_out_x = x
bott_out_y = base_yb
bott_h = hb
x += BAR_W

# ---- decoder ----
dec_ch = list(reversed(enc_ch))
dec_sp = list(reversed(enc_sp))
prev_x, prev_base_y, prev_h = bott_out_x, bott_out_y, bott_h
y_top_dec = y_top
for i in range(DEPTH):
    level = DEPTH - 1 - i
    h = bar_h(dec_sp[i])
    rise = 1.35
    y_top_dec = y_top_dec + rise
    base_y = y_top_dec - h

    x += 0.55
    big_vert_arrow(x - 0.3, prev_base_y + prev_h / 2 - 0.05,
                   prev_base_y + prev_h / 2 - 0.05 + rise, DECONV_COLOR)

    concat_ch = dec_ch[i] * 2
    vbar(x, base_y, h, CONV_COLOR, CONV_EDGE, hollow_frac=0.5)
    label_above(x, h, base_y, f"{concat_ch}")
    skip_arrow(enc_bar_x[level] + BAR_W, enc_base_y[level] + bar_h(enc_sp[level]) / 2, x)
    x += BAR_W

    x += 0.5
    small_arrow(x - 0.3, base_y + h / 2, x)
    vbar(x, base_y, h, CONV_COLOR, CONV_EDGE)
    label_above(x, h, base_y, f"{dec_ch[i]}")
    x += BAR_W
    x += 0.5
    small_arrow(x - 0.3, base_y + h / 2, x)
    vbar(x, base_y, h, CONV_COLOR, CONV_EDGE)
    label_above(x, h, base_y, f"{dec_ch[i]}")
    label_spatial(x + BAR_W / 2, base_y, f"{dec_sp[i]}×{dec_sp[i]}")
    film_targets.append((x + BAR_W / 2, base_y + h))
    x += BAR_W

    prev_x, prev_base_y, prev_h = x - BAR_W, base_y, h

# Output 1x1 conv -- single-channel noise, no activation
h_out = prev_h
x += 0.5
small_arrow(x - 0.3, prev_base_y + prev_h / 2, x, color="k")
vbar(x, prev_base_y, h_out, "0.4", "0.2")
label_above(x, h_out, prev_base_y, f"{OUT_CH}")
ax.text(x + BAR_W + 0.15, prev_base_y + h_out / 2, "noise ε\n(no activation)",
       ha="left", va="center", fontsize=8.5, color="0.3")

# ---- timestep-conditioning rail: t -> Sinusoidal -> MLP -> FiLM(t) at every level ----
rail_y = 9.15
x_rail_start = -0.6
x_rail_end = max(fx for fx, _ in film_targets) + 0.3
ax.plot([x_rail_start, x_rail_end], [rail_y, rail_y], color=FILM_COLOR, lw=1.6,
        linestyle=(0, (3, 2)), zorder=5)
ax.text(x_rail_start, rail_y + 0.18,
        f"t ~ Uniform(0,999) → Sinusoidal({TIME_EMB_DIM}) → Linear-SiLU-Linear "
        f"({TIME_EMB_DIM*4}-d) → FiLM γ,β at every level below",
        ha="left", va="bottom", fontsize=9.5, color=FILM_COLOR, fontweight="bold")
for fx, fy in film_targets:
    film_arrow(fx, rail_y, fy + 0.05)

# ---- legend ----
legend_elems = [
    Line2D([0], [0], color=CONV_ARROW, lw=2.2, label="Conv 3×3, GroupNorm(8), ReLU"),
    Line2D([0], [0], color=POOL_COLOR, lw=3.2, label="MaxPool 2×2"),
    Line2D([0], [0], color=DECONV_COLOR, lw=3.2, label="ConvTranspose 2×2 (upsample)"),
    Line2D([0], [0], color=SKIP_COLOR, lw=2.2, label="Skip connection (concatenate)"),
    Line2D([0], [0], color=FILM_COLOR, lw=1.6, linestyle=(0, (3, 2)),
           label="FiLM(t): h·(1+γ)+β from timestep embedding"),
]
ax.legend(handles=legend_elems, loc="lower right", bbox_to_anchor=(1.0, 0.0),
         fontsize=10, frameon=False)

ax.text(
    11, 11.05,
    "LatentSky ConditionalDiffusionUNet — 10var met conditioning + noisy target xₜ\n"
    f"base_ch={BASE_CH}, depth={DEPTH}, {N_PARAMS/1e6:.1f}M parameters   |   "
    "hollow bar = concatenated skip connection, before its own conv   |   "
    "purple = timestep FiLM, applied at every encoder/bottleneck/decoder level",
    ha="center", va="top", fontsize=12,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "fig17_architecture_schematic_diffusion.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}", flush=True)

PAPER_DIR.mkdir(parents=True, exist_ok=True)
out_paper = PAPER_DIR / "fig17_architecture_schematic_diffusion.png"
fig.savefig(out_paper, dpi=200, bbox_inches="tight")
print(f"wrote {out_paper}", flush=True)
