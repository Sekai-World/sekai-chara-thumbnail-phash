"""Find cards in a card-list screenshot. No model, no training.

Every fully visible card in the list has a dark "Lv.xx" bar of a fixed colour
spanning its full width near the bottom. We find those bars, then derive each
card box from its bar using the layout's detector geometry, and regularise
the result with the grid (all cards share one size).

Cards whose bar is scrolled out of view are not reported; they show up in
the next screenshot, and duplicates across screenshots are merged by card id.
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


def _cluster(values: list[float], tol: float) -> list[float]:
    """1-D clustering: sorted values closer than `tol` share a centre (median)."""
    centres: list[list[float]] = []
    for v in sorted(values):
        if centres and v - centres[-1][-1] <= tol:
            centres[-1].append(v)
        else:
            centres.append([v])
    return [float(np.median(c)) for c in centres]


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
    bars = [
        b
        for b in find_bars(mask, min_width, max_gap, max_vgap)
        if b.width >= 0.05 * short
        and lo <= b.height / b.width <= hi
        and b.area >= 0.4 * b.width * b.height
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
    rows = _cluster([b.y1 for b in bars], 0.3 * card_h)
    cols = _cluster([b.x0 for b in bars], 0.3 * card_w)

    boxes = []
    for b in bars:
        r = int(np.argmin([abs(b.y1 - v) for v in rows]))
        c = int(np.argmin([abs(b.x0 - v) for v in cols]))
        x0 = cols[c]
        y1 = rows[r] + margin
        box = CardBox(x0, y1 - card_h, x0 + card_w, y1, r, c)
        if box.x0 >= -1 and box.y0 >= -1 and box.x1 <= W + 1 and box.y1 <= H + 1:
            boxes.append(box)
    boxes.sort(key=lambda bx: (bx.row, bx.col))
    return boxes
