"""Screenshots -> the set of cards they show."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .detect import CardBox, detect_cards
from .preprocess import Layout
from .search import Match, Matcher


@dataclass
class Sighting:
    shot: int
    box: CardBox
    matches: list[Match]

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None


def scan(matcher: Matcher, shots: list[Image.Image], detect_layout: Layout | None = None, top: int = 3) -> list[Sighting]:
    detect_layout = detect_layout or matcher.layout
    sightings = []
    for i, img in enumerate(shots):
        boxes = detect_cards(img, detect_layout)
        if not boxes:
            continue
        results = matcher.match([b.crop(img) for b in boxes], top=top)
        sightings += [Sighting(i, b, r) for b, r in zip(boxes, results)]
    return sightings


def collect(sightings: list[Sighting]) -> dict:
    """Merge sightings into one entry per card id.

    A player owns each card at most once, so the same card in two
    screenshots (overlapping scroll positions) is one card. Two sightings in
    the *same* screenshot resolving to one card id means one of them is wrong;
    those are reported as conflicts.
    """
    by_card: dict[int, list[Sighting]] = {}
    for s in sightings:
        if s.best:
            by_card.setdefault(s.best.entry.card_id, []).append(s)
    cards, conflicts = [], []
    for card_id, group in by_card.items():
        group.sort(key=lambda s: -s.best.score)
        shots = [s.shot for s in group]
        if len(shots) != len(set(shots)):
            conflicts.append(card_id)
        top = group[0].best
        cards.append(
            {
                "card_id": card_id,
                "state": top.entry.state,
                "score": round(top.score, 4),
                "margin": round(top.score - group[0].matches[1].score, 4) if len(group[0].matches) > 1 else None,
                "seen": [{"shot": s.shot, "row": s.box.row, "col": s.box.col, "box": s.box.as_list()} for s in group],
            }
        )
    cards.sort(key=lambda c: (c["seen"][0]["shot"], c["seen"][0]["row"], c["seen"][0]["col"]))
    return {"cards": cards, "conflicts": conflicts, "detections": len(sightings)}


def write_debug(debug_dir: Path, shots: list[Image.Image], names: list[str], sightings: list[Sighting]) -> None:
    """Annotated screenshots, every card crop, and a labels.csv pre-filled with predictions.

    Correct the wrong rows of labels.csv and it becomes input for
    `eval --labels` and `calibrate`.
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
        draws[s.shot].rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)
        if b:
            draws[s.shot].rectangle((x0, y0, x0 + 78, y0 + 30), fill=(0, 0, 0))
            draws[s.shot].text((x0 + 3, y0 + 2), f"#{b.entry.card_id}{'+' if b.entry.state == 'after_training' else ''}\n{b.score:.2f}", fill=(255, 255, 0))
        rows.append(
            {
                "path": f"crops/{name}",
                "card_id": b.entry.card_id if b else "",
                "state": b.entry.state if b else "",
                "score": f"{b.score:.4f}" if b else "",
            }
        )
    for a, n in zip(annotated, names):
        a.save(debug_dir / f"{Path(n).stem}_annotated.png")
    with open(debug_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "card_id", "state", "score"])
        w.writeheader()
        w.writerows(rows)
