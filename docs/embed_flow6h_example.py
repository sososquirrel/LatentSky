"""Insert the flow-matching 6h example figures (flow6h_compare.png,
flow6h_errors.png) into docs/model_reference.html, right after the
"adopted successor" verdict paragraph and before section 5 -- idempotent
via BEGIN_FLOW6H_EXAMPLE / END_FLOW6H_EXAMPLE markers."""
import base64
import re
from pathlib import Path

REPO = Path("/glade/u/home/sabramian/LatentSky")
ASSETS = REPO / "docs" / "assets_model_reference"
HTML = REPO / "docs" / "model_reference.html"


def uri(name):
    b = (ASSETS / name).read_bytes()
    print(f"{name}: {len(b) / 1024:.0f} KB")
    return "data:image/png;base64," + base64.b64encode(b).decode()


block = f"""<!-- BEGIN_FLOW6H_EXAMPLE -->
<p>The same standard example block (piece 8, frame 64, block (1,4)), at the flow's
hardest trained lead, 6 h: persistence, the two-head forecast, three independent flow
draws, and truth.</p>
<figure style="margin:1.2rem 0">
<img src="{uri('flow6h_compare.png')}" alt="Six rain maps on a shared logarithmic mm scale: rain now (persistence), the two-head forecast at plus six hours, three independent flow-matching draws at plus six hours, and the true field at plus six hours."
     style="width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px">
<figcaption style="font-size:.85rem;color:var(--muted);max-width:72ch">
Two-head is the smoothest of the six panels and undershoots the peak (6.6 mm against
truth's 12.1 mm) — the conditional-mean signature again, six hours out. Each flow draw
restores sharper structure and, unlike the deterministic forecast, does not
systematically undershoot the peak (draw maxima 15.4–19.2 mm) — sharp but stochastic,
not weak. Wet-area fraction stays close to truth across the board (truth 13.6 %,
two-head 13.4 %, draws 13.0–14.0 %), consistent with the +1.5 % aggregate verdict above.
</figcaption>
</figure>
<figure style="margin:1.2rem 0">
<img src="{uri('flow6h_errors.png')}" alt="Three error maps on a shared symmetric-log diverging scale: truth minus two-head, truth minus flow draw one, and flow draw one minus two-head, all at plus six hours."
     style="width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px">
<figcaption style="font-size:.85rem;color:var(--muted);max-width:72ch">
truth − two-head (left) is spatially coherent — a systematic smoothing bias, structured
error. truth − flow draw 1 (middle) is speckled instead — errors from being
approximately right but locally displaced, not a systematic bias. flow draw 1 − two-head
(right) shows where the flow added texture the deterministic forecast lacked.
</figcaption>
</figure>
<!-- END_FLOW6H_EXAMPLE -->

"""

html = HTML.read_text()
if "BEGIN_FLOW6H_EXAMPLE" in html:
    html = re.sub(r"<!-- BEGIN_FLOW6H_EXAMPLE -->.*?<!-- END_FLOW6H_EXAMPLE -->\n\n",
                   lambda m: block, html, flags=re.S)
else:
    anchor = "the other five leads are\nqueued pending cluster availability.</p>\n"
    assert anchor in html, "insertion anchor not found"
    html = html.replace(anchor, anchor + "\n" + block, 1)

HTML.write_text(html)
print("embedded; html size", len(html) / 1e6, "MB")
