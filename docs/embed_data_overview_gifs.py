"""Swap the static domain-map and split PNGs in docs/model_reference.html
for the animated GIFs from make_data_overview_gifs.py -- located by the
unique figcaption text that follows each <img>, since these two figures
predate the BEGIN/END marker convention used for the real-example section."""
import base64
import re
from pathlib import Path

REPO = Path("/glade/u/home/sabramian/LatentSky")
ASSETS = REPO / "docs" / "assets_model_reference"
HTML = REPO / "docs" / "model_reference.html"

IMG_RE = re.compile(r'<img src="data:image/(?:png|gif);base64,[^"]*"')

TARGETS = [
    ("data_domain.gif", "The full domain (precipw shown, piece 8"),
    ("data_split.gif", "The patch split, seed 1919810"),
]


def gif_uri(name):
    b = (ASSETS / name).read_bytes()
    print(f"{name}: {len(b) / 1024:.0f} KB")
    return "data:image/gif;base64," + base64.b64encode(b).decode()


html = HTML.read_text()
for gif_name, caption_marker in TARGETS:
    cap_pos = html.index(caption_marker)
    img_matches = list(IMG_RE.finditer(html, 0, cap_pos))
    assert img_matches, f"no <img> found before caption {caption_marker!r}"
    m = img_matches[-1]
    new_tag = f'<img src="{gif_uri(gif_name)}"'
    html = html[:m.start()] + new_tag + html[m.end():]

HTML.write_text(html)
print("embedded; html size", len(html) / 1e6, "MB")
