"""Fig 1b — the shared U-Net trunk with each model's own edges.

Black: trunk (DoubleConv boxes at five resolutions, 2048-ch bottleneck,
skips). Blue: two-head edges (rain + wet/dry heads, OLR-FiLM). Orange: flow
velocity-net edges (x_t appended to input, FiLM(t), single v-hat head).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.style.use("/glade/u/home/sabramian/LatentSky/docs/latentsky.mplstyle")

BLUE, ORANGE, BLACK = "#1f6fb4", "#c55a11", "#222222"
enc_ch = [64, 128, 256, 512, 1024]
res = [128, 64, 32, 16, 8]

fig, ax = plt.subplots(figsize=(6.89, 3.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")

W, H = 7.5, 5.2


def box(x, y, label, color=BLACK, fc="white"):
    ax.add_patch(Rectangle((x, y), W, H, ec=color, fc=fc, lw=1.4, zorder=3))
    ax.text(x + W / 2, y + H / 2, label, ha="center", va="center",
            fontsize=8, zorder=4)
    return x, y


def arrow(p0, p1, color=BLACK, ls="-", lw=1.3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 color=color, ls=ls, lw=lw, zorder=2))


# encoder (left, descending) and decoder (right, ascending)
xe, xd = 14, 64
ys = [46, 36.5, 27, 17.5, 8]
for i, (ch, r, y) in enumerate(zip(enc_ch, res, ys)):
    box(xe + 3 * i, y, f"{ch} ch\n{r}²")
    box(xd - 3 * i, y, f"{ch} ch\n{r}²")
    if i < 4:  # pool / up arrows
        arrow((xe + 3 * i + W / 2 + 1, y - 0.3),
              (xe + 3 * (i + 1) + W / 2 - 1, ys[i + 1] + H + 0.3))
        arrow((xd - 3 * (i + 1) + W / 2 - 1, ys[i + 1] + H + 0.3),
              (xd - 3 * i + W / 2 + 1, y - 0.3))
    # skip connections
    arrow((xe + 3 * i + W, y + H / 2), (xd - 3 * i, y + H / 2),
          color="#999999", ls=(0, (4, 3)), lw=1.0)

# bottleneck
bx, by = (xe + 12 + xd - 12) / 2 + 1.5, 1.0
box(bx, by, "2048 ch\n4²")
arrow((xe + 12 + W / 2 + 1, ys[4] - 0.3), (bx + 1, by + H - 0.5))
arrow((bx + W - 1, by + H - 0.5), (xd - 12 + W / 2 - 1, ys[4] - 0.3))

# input (black) + the flow's extra channel (orange)
ax.text(xe + W / 2, 54.5, "10 input channels @ 128²", ha="center",
        fontsize=8.5)
ax.text(xe + W + 8, 54.5, "(+ flow state $x_t$: 11 ch)", ha="left",
        fontsize=8.5, color=ORANGE)
arrow((xe + W / 2, 53.3), (xe + W / 2, ys[0] + H + 0.4))

# two-head edges (blue)
hx = xd + W + 4
ax.add_patch(Rectangle((hx, 44), 21, 4.6, ec=BLUE, fc="#eaf2fb", lw=1.3))
ax.text(hx + 10.5, 46.3, "rain head (softplus, mm)", ha="center", fontsize=8,
        color=BLUE)
ax.add_patch(Rectangle((hx, 38), 21, 4.6, ec=BLUE, fc="#eaf2fb", lw=1.3))
ax.text(hx + 10.5, 40.3, "wet/dry head (logits)", ha="center", fontsize=8,
        color=BLUE)
arrow((xd + W, ys[0] + H / 2 + 1.2), (hx, 46.3), color=BLUE)
arrow((xd + W, ys[0] + H / 2 - 1.2), (hx, 40.3), color=BLUE)

# flow head (orange)
ax.add_patch(Rectangle((hx, 31), 21, 4.6, ec=ORANGE, fc="#fdf0e6", lw=1.3))
ax.text(hx + 10.5, 33.3, r"velocity head $\hat{v}$", ha="center", fontsize=8,
        color=ORANGE)
arrow((xd + W, ys[0] + H / 2 - 2.4), (hx, 33.3), color=ORANGE)

# FiLM conditioning
ax.text(bx + W / 2, by - 0.5, "OLR-FiLM (two-head)   ·   FiLM(t) (flow)",
        ha="center", va="top", fontsize=8)
ax.annotate("", xy=(bx + W / 2, by), xytext=(bx + W / 2, by - 0.4))

ax.text(2, 30, "encoder\n(pool ½)", fontsize=8, ha="left", color="#555555")
ax.text(hx + 10.5, 26, "decoder (up ×2)", fontsize=8, ha="center",
        color="#555555")
ax.plot([], [], color="#999999", ls=(0, (4, 3)), label="skip, concat")
ax.plot([], [], color=BLUE, label="two-head edges")
ax.plot([], [], color=ORANGE, label="flow edges")
ax.legend(loc="lower left", fontsize=8, frameon=False,
          bbox_to_anchor=(0.0, -0.02))

fig.savefig("/glade/u/home/sabramian/LatentSky/paper/figures/fig1b_architecture.pdf",
            bbox_inches="tight")
print("written")
