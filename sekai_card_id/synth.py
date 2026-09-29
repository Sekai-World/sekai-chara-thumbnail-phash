"""Render screenshot-like card crops from raw art, for evaluation without real screenshots.

This is only an approximation of the in-game look: a flat frame, blob
"icons" over the masked regions, resampling, JPEG, colour jitter and crop
misalignment. It measures robustness to those distortions; it does not
replace an evaluation on real, labelled screenshots.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw, ImageEnhance

from .preprocess import Layout

FRAME_COLORS = [(210, 210, 215), (120, 200, 230), (240, 200, 90), (230, 150, 200), (180, 160, 240)]
ICON_COLORS = [(255, 110, 150), (100, 200, 255), (130, 230, 130), (255, 200, 60), (170, 120, 255), (255, 255, 255)]


def render_card(art: Image.Image, layout: Layout, rng: random.Random, card_px: int = 156) -> Image.Image:
    """Art + frame + overlays at the layout's geometry (clean, full resolution)."""
    card = Image.new("RGB", (card_px, card_px), rng.choice(FRAME_COLORS))
    l, t, r, b = (round(v * card_px) for v in layout.card_to_art)
    card.paste(art.resize((r - l, b - t), Image.Resampling.BICUBIC), (l, t))
    draw = ImageDraw.Draw(card)
    draw.rectangle((l, t, r - 1, b - 1), outline=(255, 255, 255), width=1)
    aw, ah = r - l, b - t
    for ml, mt, mr, mb in layout.masks:
        x0, y0, x1, y1 = l + ml * aw, t + mt * ah, l + mr * aw, t + mb * ah
        w, h = x1 - x0, y1 - y0
        if w <= 2 or h <= 2:
            continue
        if w / h > 2:  # a band: row of stars + a text plate
            n = rng.randint(1, 5)
            d = min(h * 0.8, w / 8)
            for i in range(n):
                cx, cy = x0 + d * (0.7 + 1.1 * i), y0 + h / 2
                draw.regular_polygon((cx, cy, d / 2), 5, fill=(255, 220, 80), outline=(120, 80, 20))
            draw.rectangle((x1 - w * 0.35, y0 + h * 0.2, x1 - w * 0.03, y1 - h * 0.2), fill=(40, 40, 60))
        else:  # an icon
            pad = 0.1 * min(w, h)
            draw.ellipse((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=rng.choice(ICON_COLORS), outline=(255, 255, 255), width=2)
    return card


def degrade(card: Image.Image, rng: random.Random, jitter: float = 0.03) -> Image.Image:
    """Simulate how a card ends up in a user's screenshot crop."""
    size = rng.randint(72, 260)
    img = card.resize((size, size), rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.92, 1.08))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))
    # Imperfect card localisation: shift and rescale the crop box.
    pad = round(size * 0.1)
    canvas = Image.new("RGB", (size + 2 * pad, size + 2 * pad), (235, 235, 240))
    canvas.paste(img, (pad, pad))
    s = size * rng.uniform(1 - jitter, 1 + jitter)
    x0 = pad + (size - s) / 2 + rng.uniform(-jitter, jitter) * size
    y0 = pad + (size - s) / 2 + rng.uniform(-jitter, jitter) * size
    img = canvas.crop((round(x0), round(y0), round(x0 + s), round(y0 + s)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(50, 95))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def synth_query(art: Image.Image, layout: Layout, rng: random.Random, jitter: float = 0.03) -> Image.Image:
    return degrade(render_card(art, layout, rng), rng, jitter)
