"""Card localisation on real in-game card list screenshots (2026-09 UI)."""

from pathlib import Path

import pytest
from PIL import Image

from sekai_card_id.detect import detect_cards
from sekai_card_id.preprocess import Layout
from sekai_card_id.synth import occlude_top

SHOTS = Path(__file__).parent / "fixtures" / "screenshots"
# (file, first card box) measured by hand; both shots show 4 full rows of 10
# plus a 5th row cut off by the bottom of the scroll view above its level bar.
CASES = [
    ("card_list_2000x923_a.webp", (243, 145, 373, 275)),
    ("card_list_2000x923_b.webp", (243, 160, 373, 290)),
]


def load(name):
    return Image.open(SHOTS / name).convert("RGB")


@pytest.mark.parametrize("name,first", CASES)
def test_finds_every_full_card(name, first):
    boxes = detect_cards(load(name), Layout.load("default"))
    assert len(boxes) == 40
    assert sorted({(b.row, b.col) for b in boxes}) == [(r, c) for r in range(4) for c in range(10)]
    assert max(abs(u - v) for u, v in zip(boxes[0].as_list(), first)) <= 2
    # regular 154px pitch
    for b in boxes:
        assert abs(b.x0 - first[0] - 154 * b.col) <= 2
        assert abs(b.y0 - first[1] - 154 * b.row) <= 2
    assert all(b.clip_top < 0.01 for b in boxes)


@pytest.mark.parametrize("name,first", CASES)
@pytest.mark.parametrize("hidden", [25, 60, 100])
def test_measures_top_row_scrolled_under_panel(name, first, hidden):
    img = load(name)
    layout = Layout.load("default")
    boxes = detect_cards(img, layout)
    occluded = occlude_top(img, first[1] + hidden, [b.as_list() for b in boxes])
    after = detect_cards(occluded, layout)
    h = first[3] - first[1]
    # Every detection still sits exactly on the card grid: no spurious boxes
    # from half-faded bars.
    for b in after:
        assert abs((b.x0 - first[0]) / 154 - round((b.x0 - first[0]) / 154)) < 0.02
        assert abs((b.y0 - first[1]) / 154 - round((b.y0 - first[1]) / 154)) < 0.02
    top_row = [b for b in after if abs(b.y0 - first[1]) < 5]
    rest = [b for b in after if abs(b.y0 - first[1]) >= 5]
    assert len(rest) == 30 and all(b.clip_top < 0.01 for b in rest)
    if hidden / h < 0.7:
        # bar still fully visible: every top card found, with its hidden share
        assert len(top_row) == 10
        assert all(hidden / h - 0.01 <= b.clip_top <= hidden / h + 0.06 for b in top_row)
    else:
        # the fade reaches the bar: whatever is still found is mostly hidden
        assert all(b.clip_top > layout.detector["max_top_clip"] for b in top_row)
