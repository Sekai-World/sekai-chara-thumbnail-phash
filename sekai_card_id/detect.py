"""Find cards in a card-list screenshot. No model, no training.

Every fully visible card in the list has a dark "Lv.xx" bar of a fixed colour
spanning its full width near the bottom. We find those bars, then derive each
card box from its bar using the layout's detector geometry, and regularise
the result with the grid (all cards share one size).

Cards whose bar is scrolled out of view are not reported; they show up in
the next screenshot, and duplicates across screenshots are merged by card id.

A card at the top of the scroll view can have its bar visible while its upper
part is already scrolled under the panel edge. Clipped rows show the panel
background, which continues smoothly from the gutters beside the card, so we
measure `clip_top` by walking down the card while its rows look like that
background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .preprocess import Layout


@dataclass
class CardBox:
    x0: float
    y0: float
    x1: float
    y1: float
    row: int = 0
    col: int = 0
    clip_top: float = 0.0  # fraction of the card height hidden at the top

    def crop(self, img: Image.Image) -> Image.Image:
        return img.crop((round(self.x0), round(self.y0), round(self.x1), round(self.y1)))

    def as_list(self) -> list[int]:
        return [round(self.x0), round(self.y0), round(self.x1), round(self.y1)]


@dataclass
class Bar:
    x0: int
    x1: int  # exclusive
    y0: int
    y1: int  # exclusive
    area: int = 0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _row_runs(row: np.ndarray, max_gap: int, min_len: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def find_bars(mask: np.ndarray, min_width: int, max_gap: int, max_vgap: int = 1) -> list[Bar]:
    """Group horizontal runs of `mask` over nearby rows into rectangles.

    Rows crossing the white "Lv.60" text (and a master rank badge) are split
    into pieces; pieces shorter than `min_width` are ignored. A bar may skip up
    to `max_vgap` rows, so its clean top and bottom rows still join when every
    row in between is fragmented. `area` is the mask fill of the final box.
    """
    active: list[Bar] = []
    done: list[Bar] = []
    for y in range(mask.shape[0]):
        still = []
        for bar in active:
            (still if bar.y1 >= y - max_vgap else done).append(bar)
        active = still
        for a, b in _row_runs(mask[y], max_gap, min_width):
            for bar in active:
                overlap = min(b, bar.x1) - max(a, bar.x0)
                if overlap > 0.5 * min(b - a, bar.width):
                    bar.x0, bar.x1, bar.y1 = min(bar.x0, a), max(bar.x1, b), y + 1
                    break
            else:
                active.append(Bar(a, b, y, y + 1))
    bars = done + active
    for bar in bars:
        bar.area = int(mask[bar.y0 : bar.y1, bar.x0 : bar.x1].sum())
    return bars


def _looks_like_level_bar(arr: np.ndarray, mask: np.ndarray, bar: Bar) -> bool:
    """Structure checks that a same-coloured patch (e.g. a dark banner) fails.

    A level bar is a strip: the rows just above (card art) and just below
    (rarity frame) are not bar-coloured. And it carries the white "Lv.xx"
    text in its left part.
    """
    H = mask.shape[0]
    h = bar.height
    pad = max(2, round(0.15 * h))
    xs = slice(bar.x0, bar.x1)
    above = mask[max(0, bar.y0 - pad) : bar.y0, xs]
    below = mask[bar.y1 : min(H, bar.y1 + pad), xs]
    if above.size and above.mean() > 0.5:
        return False
    if below.size and below.mean() > 0.5:
        return False
    text = arr[bar.y0 : bar.y1, bar.x0 + round(0.05 * bar.width) : bar.x0 + round(0.6 * bar.width)]
    white = (text >= 190).all(axis=2).mean() if text.size else 0.0
    return 0.05 <= white <= 0.6


def _cluster(values: list[float], tol: float) -> list[float]:
    """1-D clustering: sorted values closer than `tol` share a centre (median)."""
    centres: list[list[float]] = []
    for v in sorted(values):
        if centres and v - centres[-1][-1] <= tol:
            centres[-1].append(v)
        else:
            centres.append([v])
    return [float(np.median(c)) for c in centres]


def measure_top_clip(arr: np.ndarray, box: CardBox, cfg: dict) -> float:
    """Fraction of `box` (from its top) that shows panel background instead of card.

    A row counts as background when it is close to the median colour of the
    gutters on either side of the card and has little texture. The first
    card row is the rarity frame, which is neither, so unclipped cards measure 0.
    """
    H, W = arr.shape[:2]
    x0, x1 = round(box.x0), round(box.x1)
    h = box.y1 - box.y0
    g = max(3, round(0.1 * (x1 - x0)))
    max_diff = cfg.get("clip_bg_diff", 22)
    max_std = cfg.get("clip_bg_std", 20)
    top = round(box.y0)
    y = max(top, 0)
    visible_run = 0
    first_visible = None
    while y < min(round(box.y0 + 0.9 * h), H):
        row = arr[y, max(x0 + 2, 0) : min(x1 - 2, W)]
        gutter = np.concatenate([arr[y, max(0, x0 - g) : max(0, x0 - 2)], arr[y, min(W, x1 + 2) : min(W, x1 + g)]])
        if len(gutter) == 0 or len(row) == 0:
            break
        diff = np.abs(row - np.median(gutter, axis=0)).mean()
        std = row.std(axis=0).mean()
        if diff < max_diff and std < max_std:
            visible_run, first_visible = 0, None
        else:
            first_visible = y if first_visible is None else first_visible
            visible_run += 1
            if visible_run >= 3:
                break
        y += 1
    start = first_visible if first_visible is not None else y
    return float(min(max(start - box.y0, 0.0) / h, 1.0))


def _on_grid(values: list[float], card_size: float) -> list[bool]:
    """Which cluster centres sit on the lattice spanned by the well-supported ones.

    A partly faded or overlapped bar can survive the shape filters and open
    a spurious column/row between real ones; real cards sit a whole number of
    pitches apart.
    """
    centres = _cluster(values, 0.3 * card_size)
    counts = [sum(abs(v - c) <= 0.3 * card_size for v in values) for c in centres]
    strong = [c for c, n in zip(centres, counts) if n >= max(2, 0.5 * max(counts))]
    gaps = [b - a for a, b in zip(strong, strong[1:]) if b - a >= 0.9 * card_size]
    if not gaps:
        return [True] * len(values)
    pitch = min(gaps)  # neighbouring strong columns may skip an empty one
    pitch = float(np.median([g / round(g / pitch) for g in gaps]))
    anchor = strong[int(np.argmax([n for c, n in zip(centres, counts) if c in strong]))]
    ok_centres = [abs((c - anchor) / pitch - round((c - anchor) / pitch)) <= 0.08 for c in centres]
    return [ok_centres[int(np.argmin([abs(v - c) for c in centres]))] for v in values]


def detect_cards(img: Image.Image, layout: Layout) -> list[CardBox]:
    cfg = layout.detector
    if not cfg or cfg.get("type") != "level_bar":
        raise ValueError(f"layout {layout.name!r} has no level_bar detector configured")
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    H, W = arr.shape[:2]
    color = np.asarray(cfg["bar_color"], dtype=np.int16)
    mask = (np.abs(arr - color) <= cfg.get("bar_tolerance", 18)).all(axis=2)

    short = min(H, W)
    min_width = max(8, round(0.04 * short))
    max_gap = max(2, round(0.003 * short))
    max_vgap = max(2, round(0.02 * short))
    lo, hi = cfg.get("bar_aspect", (0.12, 0.23))
    card_scale = cfg.get("card_width_per_bar", 1.0)
    card_aspect = cfg.get("card_aspect", 1.0)
    bars = [
        b
        for b in find_bars(mask, min_width, max_gap, max_vgap)
        if b.width >= 0.05 * short
        # The list is landscape with several cards per row: a card is never
        # wider than ~30% of the screen or taller than it.
        and b.width * card_scale <= cfg.get("max_card_width", 0.3) * W
        and b.width * card_scale * card_aspect <= H
        and lo <= b.height / b.width <= hi
        and b.area >= 0.4 * b.width * b.height
        and _looks_like_level_bar(arr, mask, b)
    ]
    if not bars:
        return []

    # Every card in the grid has the same size. The largest bars are intact;
    # narrower ones were clipped by an overlay (e.g. master rank badge).
    widths = np.array([b.width for b in bars])
    ref = float(np.median(widths[widths >= 0.9 * widths.max()]))
    bars = [b for b in bars if 0.6 * ref <= b.width <= 1.1 * ref]

    card_w = ref * cfg.get("card_width_per_bar", 1.0)
    card_h = card_w * cfg.get("card_aspect", 1.0)
    margin = card_w * cfg.get("bottom_margin", 0.0)
    on_cols = _on_grid([b.x0 for b in bars], card_w)
    on_rows = _on_grid([b.y1 for b in bars], card_h)
    bars = [b for b, c, r in zip(bars, on_cols, on_rows) if c and r]
    rows = _cluster([b.y1 for b in bars], 0.3 * card_h)
    cols = _cluster([b.x0 for b in bars], 0.3 * card_w)

    boxes = []
    for b in bars:
        r = int(np.argmin([abs(b.y1 - v) for v in rows]))
        c = int(np.argmin([abs(b.x0 - v) for v in cols]))
        x0 = cols[c]
        y1 = rows[r] + margin
        box = CardBox(x0, y1 - card_h, x0 + card_w, y1, r, c)
        # The top may run off the image (scrolled away); the sides may not.
        if box.x0 >= -1 and box.x1 <= W + 1 and box.y1 <= H + 1:
            boxes.append(box)
    arr_f = arr.astype(np.float32)
    for box in boxes:
        box.clip_top = measure_top_clip(arr_f, box, cfg)
    boxes.sort(key=lambda bx: (bx.row, bx.col))
    return boxes
