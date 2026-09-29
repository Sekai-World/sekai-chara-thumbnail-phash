"""Geometry shared by gallery building and querying.

Gallery images are the raw card art (thumbnail/chara). Queries are card
crops cut out of a screenshot, i.e. art with the in-game frame and overlays
(attribute icon, rarity stars, level bar, master rank badge, ...) drawn on
top. A Layout describes where the art sits inside a card crop and which parts
of the card are covered by overlays; both sides mask those parts identically
so they never contribute to similarity.

Overlays are placed relative to the card, so masks are given in card
coordinates (`card_masks`) and converted to art coordinates on demand. That
way re-calibrating `card_to_art` never requires touching the masks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Bump when preprocessing changes in a way that invalidates stored gallery vectors.
PREPROCESS_VERSION = 2

Box = tuple[float, float, float, float]  # relative (left, top, right, bottom)

LAYOUT_DIR = Path(__file__).parent / "layouts"


@dataclass(frozen=True)
class Layout:
    name: str
    card_to_art: Box  # art square inside the card crop
    card_masks: tuple[Box, ...]  # overlay regions, card-relative
    fill: tuple[int, int, int] = (128, 128, 128)
    description: str = ""
    detector: dict | None = None  # screenshot card localisation parameters (see detect.py)

    @classmethod
    def load(cls, name_or_path: str | Path = "default") -> "Layout":
        p = Path(name_or_path)
        if not p.is_file():
            p = LAYOUT_DIR / f"{name_or_path}.json"
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls(
            name=d["name"],
            card_to_art=tuple(d["card_to_art"]),
            card_masks=tuple(tuple(m) for m in d.get("card_masks", [])),
            fill=tuple(d.get("fill", (128, 128, 128))),
            description=d.get("description", ""),
            detector=d.get("detector"),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **changes) -> "Layout":
        return Layout.from_dict({**self.to_dict(), **changes})

    @property
    def masks(self) -> tuple[Box, ...]:
        """Overlay regions in art coordinates (clipped to the art)."""
        al, at, ar, ab = self.card_to_art
        aw, ah = ar - al, ab - at
        out = []
        for l, t, r, b in self.card_masks:
            box = (
                min(max((l - al) / aw, 0.0), 1.0),
                min(max((t - at) / ah, 0.0), 1.0),
                min(max((r - al) / aw, 0.0), 1.0),
                min(max((b - at) / ah, 0.0), 1.0),
            )
            if box[2] > box[0] and box[3] > box[1]:
                out.append(box)
        return tuple(out)

    def fingerprint(self) -> str:
        """Identifies what affects gallery content (not description / detector)."""
        d = self.to_dict()
        d.pop("description")
        d.pop("detector")
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def load_image(src, background=(128, 128, 128)) -> Image.Image:
    """Open an image as RGB, compositing any transparency over `background`."""
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img.load()
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (*background, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _abs_box(size: tuple[int, int], box: Box) -> tuple[float, float, float, float]:
    w, h = size
    return (box[0] * w, box[1] * h, box[2] * w, box[3] * h)


def art_from_card(card: Image.Image, layout: Layout) -> Image.Image:
    """Cut the art region out of a card crop."""
    l, t, r, b = _abs_box(card.size, layout.card_to_art)
    return card.crop((round(l), round(t), round(r), round(b)))


def apply_masks(art: Image.Image, layout: Layout) -> Image.Image:
    """Paint overlay regions with a flat colour (input for the embedder)."""
    if not layout.masks:
        return art
    out = art.copy()
    draw = ImageDraw.Draw(out)
    for box in layout.masks:
        l, t, r, b = _abs_box(out.size, box)
        # Cover every pixel the box touches; thin strips must not vanish or invert.
        x0, y0, x1, y1 = int(np.floor(l)), int(np.floor(t)), int(np.ceil(r)) - 1, int(np.ceil(b)) - 1
        if x1 >= x0 and y1 >= y0:
            draw.rectangle((x0, y0, x1, y1), fill=tuple(layout.fill))
    return out


def keep_mask(layout: Layout, size: int, dilate: int = 1) -> np.ndarray:
    """Boolean (size, size) array: True for template pixels outside every overlay."""
    keep = np.ones((size, size), dtype=bool)
    for l, t, r, b in layout.masks:
        x0 = max(0, int(np.floor(l * size)) - dilate)
        y0 = max(0, int(np.floor(t * size)) - dilate)
        x1 = min(size, int(np.ceil(r * size)) + dilate)
        y1 = min(size, int(np.ceil(b * size)) + dilate)
        keep[y0:y1, x0:x1] = False
    return keep


def make_template(art: Image.Image, size: int) -> np.ndarray:
    """Small colour thumbnail of the art used for pixel-level re-ranking."""
    return np.asarray(art.resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def art_template_variants(
    card: Image.Image,
    layout: Layout,
    size: int,
    shifts: tuple[float, ...] = (-0.03, 0.0, 0.03),
    scales: tuple[float, ...] = (0.97, 1.0, 1.03),
) -> np.ndarray:
    """Templates of the art region under small shift / scale perturbations.

    Screenshot crops are never perfectly aligned; searching a few offsets at
    query time makes the pixel comparison tolerant to that. Returns
    (len(shifts)**2 * len(scales), size, size, 3) uint8.
    """
    W, H = card.size
    l, t, r, b = _abs_box(card.size, layout.card_to_art)
    cx, cy, bw, bh = (l + r) / 2, (t + b) / 2, r - l, b - t
    # Supersample once so the per-variant resamples stay cheap.
    scale_up = max(1.0, 4 * size / max(bw, 1))
    src = card if scale_up == 1.0 else card.resize((round(W * scale_up), round(H * scale_up)), Image.Resampling.BILINEAR)
    k = src.size[0] / W
    out = []
    for s in scales:
        for dy in shifts:
            for dx in shifts:
                w2, h2 = bw * s, bh * s
                x0 = cx + dx * bw - w2 / 2
                y0 = cy + dy * bh - h2 / 2
                # Keep the box inside the image (Pillow requires it).
                x0 = min(max(x0, 0.0), W - w2) if w2 <= W else 0.0
                y0 = min(max(y0, 0.0), H - h2) if h2 <= H else 0.0
                sw, sh = src.size
                box = (
                    max(0.0, x0 * k),
                    max(0.0, y0 * k),
                    min(float(sw), (x0 + w2) * k),  # float error can land just past the edge
                    min(float(sh), (y0 + h2) * k),
                )
                out.append(np.asarray(src.resize((size, size), Image.Resampling.BOX, box=box), dtype=np.uint8))
    return np.stack(out)
