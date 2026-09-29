import random

import pytest

from conftest import fake_art
from sekai_card_id.calibrate import fit_card_to_art
from sekai_card_id.detect import detect_cards
from sekai_card_id.embedders import TinyEmbedder
from sekai_card_id.gallery import build_gallery
from sekai_card_id.masterdata import expand_entries
from sekai_card_id.preprocess import Layout, load_image
from sekai_card_id.scan import collect, scan
from sekai_card_id.search import Matcher
from sekai_card_id.synth import render_card, render_screenshot


def test_masks_follow_card_to_art():
    layout = Layout.from_dict(
        {"name": "t", "card_to_art": [0.1, 0.1, 0.9, 0.9], "card_masks": [[0, 0, 0.3, 0.3], [0, 0.9, 1, 1]]}
    )
    (icon,) = layout.masks  # the second mask lies entirely outside the art
    assert icon == pytest.approx((0, 0, 0.25, 0.25))


@pytest.mark.parametrize("card_px", [96, 130, 180])
def test_detects_grid_and_skips_clipped_row(card_px):
    layout = Layout.load("default")
    rng = random.Random(card_px)
    gap = card_px // 5
    cols = 5
    width = 60 + cols * (card_px + gap)
    # 3 full rows + a 4th row cut off by the bottom edge above its level bar
    height = 40 + 3 * (card_px + gap) + card_px // 2
    img, truth = render_screenshot(
        [fake_art(i) for i in range(4 * cols)], layout, rng, cols, card_px, gap, size=(width, height)
    )
    boxes = detect_cards(img, layout)
    full = [t for t in truth if t[3] <= height]
    assert len(boxes) == len(full) == 3 * cols
    for b, t in zip(boxes, full):
        assert max(abs(u - v) for u, v in zip(b.as_list(), t)) <= 2


def test_scan_recognises_screenshot_cards(card_assets):
    cards, asset_dir = card_assets
    layout = Layout.load("default")
    g, _ = build_gallery(expand_entries(cards), asset_dir, TinyEmbedder(), layout, log=lambda *_: None)
    matcher = Matcher(g, TinyEmbedder())
    rng = random.Random(3)
    picked = rng.sample(g.entries, 30)
    arts = [load_image(asset_dir / f"{e.key}.png") for e in picked]
    shots = [
        render_screenshot(arts[:15], layout, rng, cols=5, size=(900, 560))[0],
        render_screenshot(arts[10:], layout, rng, cols=5, size=(900, 700))[0],  # overlaps the first
    ]
    result = collect(scan(matcher, shots))
    found = {c["card_id"] for c in result["cards"]}
    expected = {e.card_id for e in picked}
    assert result["detections"] == 35
    assert len(found & expected) >= 0.9 * len(expected)


def test_calibrate_recovers_art_box():
    true = Layout.load("default")
    rng = random.Random(0)
    pairs = []
    for i in range(6):
        art = fake_art(100 + i)
        pairs.append((render_card(art, true, rng, 150), art))
    l, t, r, b = true.card_to_art
    start = true.replace(card_to_art=[l + 0.03, t - 0.02, r + 0.01, b - 0.04])
    fitted, report = fit_card_to_art(start, pairs, log=lambda *_: None)
    assert report["mean_ncc_after"] > report["mean_ncc_before"]
    assert max(abs(u - v) for u, v in zip(fitted.card_to_art, true.card_to_art)) <= 0.012
