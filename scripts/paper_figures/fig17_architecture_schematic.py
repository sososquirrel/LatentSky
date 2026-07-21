"""Figure: schematic of the U-Net architecture used for the 9var (mse+uv)
model (latentsky.models.UNet2D, base_ch=64, depth=5), drawn in the classic
Ronneberger-style U-Net diagram: vertical bars sized by spatial resolution,
colored arrows for each operation (conv, pool, deconv, skip concat).

Exact channel/spatial progression computed the same way UNet2D.__init__
builds its encoder/decoder -- not hand-typed.
"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/paper_figures"

IN_CH = 9
OUT_CH = 1
BASE_CH = 64
DEPTH = 5
PATCH = 64
N_PARAMS = 124_377_665

CONV_COLOR = "#E8974A"   # bar fill (feature map)
CONV_EDGE = "#B36A1E"
POOL_COLOR = "#B33F3F"   # red down arrow
DECONV_COLOR = "#3F7D3F"  # green up arrow
SKIP_COLOR = "0.55"      # gray horizontal arrow
CONV_ARROW = "#3B6FA8"   # blue small arrow

# spatial size -> bar height (compressed scale so 2x2 stage stays visible)
def bar_h(spatial):
    import math
    return 1.1 + 1.55 * math.log2(spatial)

BAR_W = 0.16
GAP = 0.05

fig, ax = plt.subplots(figsize=(21, 9.5))
ax.set_xlim(-1, 28)
ax.set_ylim(-1, 9.5)
ax.axis("off")
ax.set_aspect("equal")


def vbar(x, base_y, h, color, edge, hollow_frac=0.0):
    """Vertical bar; if hollow_frac>0, the bottom portion is drawn unfilled
    (represents the skip-connected portion before its own conv, as in the
    reference figure)."""
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
enc_bar_x = []   # x of the 2nd (output) conv bar at each encoder level, for skip start
enc_base_y = []  # base y of encoder stage (aligned so bottoms step down)
y_top = 8.0      # top alignment for input/stage 0

# Input bar
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
    base_y = y_top - h  # top-aligned within this row; rows step down visually via pool arrows below
    # two conv bars per DoubleConv
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
    label_spatial(x + BAR_W / 2, base_y, f"{enc_sp[d]}×{enc_sp[d]}")
    x += BAR_W

    if d < DEPTH - 1:
        # pool arrow down + shift right, drop to next row
        nxt_h = bar_h(enc_sp[d + 1])
        drop = (y_top - h / 2) - (y_top - nxt_h / 2) if False else 1.35
        x += 0.55
        big_vert_arrow(x - 0.3, base_y + h / 2 - 0.05, base_y + h / 2 - 0.05 - drop, POOL_COLOR)
        ax.text(x - 0.3, base_y + h / 2 - 0.05 - drop / 2, "", fontsize=1)
        y_top = y_top - drop
    else:
        # last encoder level -> bottleneck
        x += 0.55
        drop = 1.35
        big_vert_arrow(x - 0.3, base_y + h / 2 - 0.05, base_y + h / 2 - 0.05 - drop, POOL_COLOR)
        y_top = y_top - drop

# Bottleneck (DoubleConv 1024 -> 2048 -> 2048)
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
bott_out_x = x
bott_out_y = base_yb
bott_h = hb
x += BAR_W

# ---- decoder: mirror back up, deconv (green) + concat (gray, hollow bar) + 2 convs ----
dec_ch = list(reversed(enc_ch))       # index i -> level DEPTH-1-i
dec_sp = list(reversed(enc_sp))
prev_x, prev_base_y, prev_h = bott_out_x, bott_out_y, bott_h
y_top_dec = y_top  # will climb back up
for i in range(DEPTH):
    level = DEPTH - 1 - i
    h = bar_h(dec_sp[i])
    rise = 1.35
    y_top_dec = y_top_dec + rise
    base_y = y_top_dec - h

    x += 0.55
    big_vert_arrow(x - 0.3, prev_base_y + prev_h / 2 - 0.05,
                   prev_base_y + prev_h / 2 - 0.05 + rise, DECONV_COLOR)

    # concat bar (hollow bottom = skip-copied portion), channel count = 2x dec_ch[i]
    concat_ch = dec_ch[i] * 2
    vbar(x, base_y, h, CONV_COLOR, CONV_EDGE, hollow_frac=0.5)
    label_above(x, h, base_y, f"{concat_ch}")
    # skip connection arrow from matching encoder bar
    skip_arrow(enc_bar_x[level] + BAR_W, enc_base_y[level] + bar_h(enc_sp[level]) / 2,
              x)
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
    x += BAR_W

    prev_x, prev_base_y, prev_h = x - BAR_W, base_y, h

# Output 1x1 conv
h_out = prev_h
x += 0.5
small_arrow(x - 0.3, prev_base_y + prev_h / 2, x, color="k")
vbar(x, prev_base_y, h_out, "0.4", "0.2")
label_above(x, h_out, prev_base_y, f"{OUT_CH}")
ax.text(x + BAR_W + 0.15, prev_base_y + h_out / 2, "Softplus\n(≥0)", ha="left",
       va="center", fontsize=8.5, color="0.3")

# ---- legend ----
legend_elems = [
    Line2D([0], [0], color=CONV_ARROW, lw=2.2, label="Conv 3×3, GroupNorm(8), ReLU"),
    Line2D([0], [0], color=POOL_COLOR, lw=3.2, label="MaxPool 2×2"),
    Line2D([0], [0], color=DECONV_COLOR, lw=3.2, label="ConvTranspose 2×2 (upsample)"),
    Line2D([0], [0], color=SKIP_COLOR, lw=2.2, label="Skip connection (concatenate)"),
]
ax.legend(handles=legend_elems, loc="lower right", bbox_to_anchor=(1.0, 0.0),
         fontsize=10, frameon=False)

ax.text(
    11, 9.15,
    "LatentSky UNet2D — 9var (mse+uv) input set\n"
    f"base_ch={BASE_CH}, depth={DEPTH}, {N_PARAMS/1e6:.1f}M parameters   |   "
    "hollow bar = concatenated skip connection, before its own conv",
    ha="center", va="top", fontsize=12,
)

out = OUT_DIR / "fig17_architecture_schematic.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}", flush=True)
