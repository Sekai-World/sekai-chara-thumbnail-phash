"""Screenshots -> the set of cards they show."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .detect import CardBox, detect_cards
from .preprocess import Layout
from .search import Match, Matcher

DEFAULT_MAX_TOP_CLIP = 0.35
CLIP_MARGIN = 0.02  # also ignore the fade just below the measured clip line


@dataclass
class Sighting:
    shot: int
    box: CardBox
    matches: list[Match]
    skipped: str | None = None

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None

    def where(self) -> dict:
        return {"shot": self.shot, "row": self.box.row, "col": self.box.col, "box": self.box.as_list()}


def scan(matcher: Matcher, shots: list[Image.Image], detect_layout: Layout | None = None, top: int = 3) -> list[Sighting]:
    """Detect and identify every card in every screenshot.

    Cards whose top is scrolled under the panel edge are matched on their
    visible part only; past `max_top_clip` too little art is left and the
    card is skipped (it is fully visible in an adjacent screenshot).
    """
    detect_layout = detect_layout or matcher.layout
    max_clip = (detect_layout.detector or {}).get("max_top_clip", DEFAULT_MAX_TOP_CLIP)
    sightings = []
    for i, img in enumerate(shots):
        boxes = detect_cards(img, detect_layout)
        todo = []
        for b in boxes:
            if b.clip_top > max_clip:
                sightings.append(Sighting(i, b, [], skipped="clipped"))
            else:
                todo.append(b)
        if not todo:
            continue
        occlusions = [[(0.0, 0.0, 1.0, b.clip_top + CLIP_MARGIN)] if b.clip_top > 0.005 else None for b in todo]
        results = matcher.match([b.crop(img) for b in todo], top=top, occlusions=occlusions)
        sightings += [Sighting(i, b, r) for b, r in zip(todo, results)]
    sightings.sort(key=lambda s: (s.shot, s.box.row, s.box.col))
    return sightings


def _num(x: float) -> float | None:
    return round(x, 4) if math.isfinite(x) else None


def _candidates(s: Sighting) -> list[dict]:
    return [
        {"card_id": m.entry.card_id, "state": m.entry.state, "score": _num(m.score), "margin": _num(m.margin)}
        for m in s.matches
    ]


def collect(sightings: list[Sighting]) -> dict:
    """Merge sightings into one entry per card id.

    A player owns each card at most once, so the same card in two
    screenshots (overlapping scroll positions) is one card. Sightings the
    matcher rejected, and extra sightings of one card id within the *same*
    screenshot (one of them must be wrong), go to `uncertain` with their
    top candidates so a UI can ask the user to confirm.
    """
    by_card: dict[int, list[Sighting]] = {}
    uncertain, skipped = [], []
    for s in sightings:
        if s.skipped:
            skipped.append({**s.where(), "reason": s.skipped, "clip_top": round(s.box.clip_top, 3)})
        elif not s.best:
            continue
        elif s.best.accepted is False:
            uncertain.append({**s.where(), "reason": "low_confidence", "candidates": _candidates(s)})
        else:
            by_card.setdefault(s.best.entry.card_id, []).append(s)

    cards = []
    for card_id, group in by_card.items():
        group.sort(key=lambda s: -s.best.score)
        kept, seen_shots = [], set()
        for s in group:
            if s.shot in seen_shots:
                uncertain.append({**s.where(), "reason": "duplicate_in_screenshot", "candidates": _candidates(s)})
            else:
                seen_shots.add(s.shot)
                kept.append(s)
        top = kept[0].best
        cards.append(
            {
                "card_id": card_id,
                "state": top.entry.state,
                "score": _num(top.score),
                "margin": _num(top.margin),
                "accepted": top.accepted,  # None: no reject thresholds fitted for this gallery
                "seen": [s.where() for s in kept],
            }
        )
    cards.sort(key=lambda c: (c["seen"][0]["shot"], c["seen"][0]["row"], c["seen"][0]["col"]))
    uncertain.sort(key=lambda u: (u["shot"], u["row"], u["col"]))
    return {"cards": cards, "uncertain": uncertain, "skipped": skipped, "detections": len(sightings)}


def write_debug(debug_dir: Path, shots: list[Image.Image], names: list[str], sightings: list[Sighting]) -> None:
    """Annotated screenshots, every card crop, and a labels.csv pre-filled with predictions.

    Correct the wrong rows of labels.csv and it becomes input for
    `eval --labels` and `calibrate`. Box colours: green accepted, orange
    rejected, red no thresholds, grey skipped.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = debug_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    annotated = [img.copy() for img in shots]
    draws = [ImageDraw.Draw(a) for a in annotated]
    rows = []
    for s in sightings:
        name = f"{Path(names[s.shot]).stem}_r{s.box.row}c{s.box.col}.png"
        s.box.crop(shots[s.shot]).save(crops_dir / name)
        b = s.best
        x0, y0, x1, y1 = s.box.as_list()
        color = (160, 160, 160) if s.skipped else {True: (0, 200, 0), False: (255, 140, 0), None: (255, 0, 0)}[b.accepted if b else None]
        draws[s.shot].rectangle((x0, y0, x1, y1), outline=color, width=3)
        if s.box.clip_top > 0:
            cy = y0 + s.box.clip_top * (y1 - y0)
            draws[s.shot].line((x0, cy, x1, cy), fill=color, width=2)
        if b:
            draws[s.shot].rectangle((x0, y1 - 32, x0 + 84, y1), fill=(0, 0, 0))
            label = f"#{b.entry.card_id}{'+' if b.entry.state == 'after_training' else ''}\n{b.score:.2f}"
            draws[s.shot].text((x0 + 3, y1 - 30), label, fill=(255, 255, 0))
        if s.skipped or s.box.clip_top > 0.005:
            continue  # partial crops would mislead `eval --labels` and `calibrate`
        rows.append(
            {
                "path": f"crops/{name}",
                "card_id": b.entry.card_id if b else "",
                "state": b.entry.state if b else "",
                "score": f"{b.score:.4f}" if b else "",
                "accepted": "" if not b or b.accepted is None else int(b.accepted),
            }
        )
    for a, n in zip(annotated, names):
        a.save(debug_dir / f"{Path(n).stem}_annotated.png")
    with open(debug_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "card_id", "state", "score", "accepted"])
        w.writeheader()
        w.writerows(rows)
