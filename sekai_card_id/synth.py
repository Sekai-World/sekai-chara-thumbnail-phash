"""Render screenshot-like cards from raw art, for evaluation and tests without real screenshots.

This is only an approximation of the in-game look: a flat frame, blob
"icons" and stars over the masked regions, a level bar (when the layout has
a level_bar detector), resampling, JPEG, colour jitter and crop
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
    cfg = layout.detector or {}
    card_w = card_px
    card_h = round(card_px * cfg.get("card_aspect", 1.0))
    card = Image.new("RGB", (card_w, card_h), rng.choice(FRAME_COLORS))
    l, t, r, b = layout.card_to_art
    box = (round(l * card_w), round(t * card_h), round(r * card_w), round(b * card_h))
    card.paste(art.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BICUBIC), box[:2])
    draw = ImageDraw.Draw(card)
    for ml, mt, mr, mb in layout.card_masks:
        x0, y0, x1, y1 = ml * card_w, mt * card_h, mr * card_w, mb * card_h
        w, h = x1 - x0, y1 - y0
        if w <= 2 or h <= 2:
            continue
        if w / h > 2:  # a band: row of stars along its top
            d = min(h * 0.4, w / 8)
            for i in range(rng.randint(1, 4)):
                draw.regular_polygon((x0 + d * (0.7 + 1.1 * i), y0 + d * 0.6, d / 2), 5, fill=(255, 220, 80), outline=(120, 80, 20))
        else:  # an icon
            pad = 0.1 * min(w, h)
            draw.rectangle((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=rng.choice(ICON_COLORS))
    if cfg.get("type") == "level_bar":
        bar_w = card_w / cfg.get("card_width_per_bar", 1.0)
        bar_h = bar_w * 0.17
        by1 = card_h - cfg.get("bottom_margin", 0.0) * card_w
        draw.rectangle((0, by1 - bar_h, bar_w - 1, by1 - 1), fill=tuple(cfg["bar_color"]))
        # "Lv.60" glyph blobs
        for i in range(rng.randint(3, 5)):
            gx = bar_w * (0.06 + 0.09 * i)
            draw.rectangle((gx, by1 - bar_h * 0.8, gx + bar_w * 0.05, by1 - bar_h * 0.2), fill=(250, 250, 250))
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
    return _jpeg(img, rng.randint(50, 95))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def synth_query(art: Image.Image, layout: Layout, rng: random.Random, jitter: float = 0.03) -> Image.Image:
    return degrade(render_card(art, layout, rng), rng, jitter)


def render_screenshot(
    arts: list[Image.Image],
    layout: Layout,
    rng: random.Random,
    cols: int = 5,
    card_px: int = 130,
    gap: int = 24,
    size: tuple[int, int] = (1000, 700),
    origin: tuple[int, int] = (60, 40),
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """Lay cards out in a grid on a busy background. Returns (image, true card boxes)."""
    W, H = size
    bg = Image.linear_gradient("L").resize((W, H)).convert("RGB")
    bg = Image.blend(bg, Image.new("RGB", (W, H), (150, 165, 200)), 0.7)
    boxes = []
    for i, art in enumerate(arts):
        r, c = divmod(i, cols)
        card = render_card(art, layout, rng, card_px)
        x, y = origin[0] + c * (card_px + gap), origin[1] + r * (card.size[1] + gap)
        if x + card.size[0] > W:
            continue
        bg.paste(card, (x, y))  # cards past the bottom edge are pasted clipped
        boxes.append((x, y, x + card.size[0], y + card.size[1]))
    return _jpeg(bg, 85), boxes
