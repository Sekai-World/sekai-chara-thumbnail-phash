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

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from .detect import load_level_text_template
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
        # "Lv." from the detector's template, then digit blobs
        text_h = round(bar_h * 0.6)
        tpl = load_level_text_template(cfg.get("level_text_template", "level_text.png"))
        glyph = Image.fromarray(np.round(tpl * 255).astype(np.uint8)).resize((2 * text_h, text_h), Image.Resampling.BILINEAR)
        gx, gy = round(bar_w * 0.06), round(by1 - bar_h * 0.8)
        card.paste((250, 250, 250), (gx, gy, gx + glyph.width, gy + text_h), glyph)
        for i in range(rng.randint(1, 2)):
            dx = gx + glyph.width + bar_w * (0.02 + 0.09 * i)
            draw.rectangle((dx, by1 - bar_h * 0.8, dx + bar_w * 0.05, by1 - bar_h * 0.2), fill=(250, 250, 250))
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


def occlude_top(
    img: Image.Image, y_clip: int, card_boxes: list[tuple[int, int, int, int]], fade: int = 12
) -> Image.Image:
    """Simulate the scroll view hiding everything above `y_clip`.

    Inside each card's columns, rows above the clip line are replaced by
    background interpolated from the pixels beside the cards on the same row,
    with a linear fade into the card over `fade` rows (as the game does).
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32).copy()
    H, W = arr.shape[:2]
    margin = 3  # keep antialiased frame pixels out of the background sample
    covered = np.zeros(W, dtype=bool)
    sample = np.ones(W, dtype=bool)
    for x0, _, x1, _ in card_boxes:
        covered[max(0, x0) : min(W, x1)] = True
        sample[max(0, x0 - margin) : min(W, x1 + margin)] = False
    xs = np.arange(W)
    free = xs[sample]
    rows = min(H, y_clip + fade)
    bg = np.stack(
        [np.stack([np.interp(xs, free, arr[y, free, c]) for c in range(3)], axis=1) for y in range(rows)]
    )
    # The real background is a blurred picture: smooth the estimate vertically too.
    k = 9
    padded = np.concatenate([bg[:1].repeat(k // 2, 0), bg, bg[-1:].repeat(k // 2, 0)])
    bg = np.stack([padded[i : i + k].mean(axis=0) for i in range(rows)])
    for y in range(rows):
        alpha = 1.0 if y < y_clip else 1.0 - (y - y_clip + 1) / (fade + 1)
        arr[y, covered] = alpha * bg[y, covered] + (1 - alpha) * arr[y, covered]
    return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))
