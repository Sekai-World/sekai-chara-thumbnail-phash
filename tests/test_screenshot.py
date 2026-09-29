import random

from PIL import Image

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
from sekai_card_id.synth import occlude_top, render_card, render_screenshot


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


def _gallery(card_assets):
    cards, asset_dir = card_assets
    layout = Layout.load("default")
    g, _ = build_gallery(expand_entries(cards), asset_dir, TinyEmbedder(), layout, log=lambda *_: None)
    return g, asset_dir, layout


@pytest.mark.parametrize("hidden,expect_skip", [(0.2, False), (0.6, True)])
def test_scan_handles_top_row_under_panel(card_assets, hidden, expect_skip):
    g, asset_dir, layout = _gallery(card_assets)
    matcher = Matcher(g, TinyEmbedder())
    rng = random.Random(5)
    picked = rng.sample(g.entries, 15)
    arts = [load_image(asset_dir / f"{e.key}.png") for e in picked]
    img, truth = render_screenshot(arts, layout, rng, cols=5, size=(900, 560))
    top = truth[0][1]
    img = occlude_top(img, round(top + hidden * (truth[0][3] - top)), truth)
    sightings = scan(matcher, [img])
    row0 = [s for s in sightings if s.box.row == 0]
    assert len(row0) == 5 and all(abs(s.box.clip_top - hidden) < 0.08 for s in row0)
    if expect_skip:
        assert all(s.skipped == "clipped" for s in row0)
    else:
        assert not any(s.skipped for s in row0)
        hits = sum(s.best.entry.card_id == e.card_id for s, e in zip(row0, picked[:5]))
        assert hits >= 4


def test_margin_ignores_other_state_of_same_card(card_assets):
    g, asset_dir, layout = _gallery(card_assets)
    matcher = Matcher(g, TinyEmbedder())
    trained = next(e for e in g.entries if e.state == "after_training")
    q = render_card(load_image(asset_dir / f"{trained.key}.png"), layout, random.Random(0), 150)
    (res,) = matcher.match([q], top=20)
    best = res[0]
    rival = max(m.score for m in res if m.entry.card_id != best.entry.card_id)
    assert best.margin == pytest.approx(best.score - rival)


def test_fit_reject_turns_away_unknown_cards(card_assets):
    from sekai_card_id.evaluate import fit_reject, synthetic_queries

    g, asset_dir, layout = _gallery(card_assets)
    matcher = Matcher(g, TinyEmbedder())
    queries = synthetic_queries(g.entries, asset_dir, layout, n=45, seed=2)
    fit = fit_reject(matcher, queries, target_error=0.02)
    assert fit["known_accepted_wrong"] + fit["unknown_accepted"] <= 2 * 0.02 + 1e-9
    assert fit["known_accepted_correct"] >= 0.8

    matcher.reject = fit
    cards = [q for q, _ in queries]
    known = matcher.match(cards, top=1)
    unknown = matcher.match(cards, top=1, exclude_card_ids=[{e.card_id} for _, e in queries])
    assert sum(r[0].accepted for r in unknown) <= 1
    assert all(r[0].entry.card_id == e.card_id for r, (_, e) in zip(known, queries) if r[0].accepted)


def test_collect_routes_rejected_and_duplicate_sightings():
    from sekai_card_id.detect import CardBox
    from sekai_card_id.masterdata import CardEntry
    from sekai_card_id.scan import Sighting
    from sekai_card_id.search import Match

    def m(card_id, score, accepted):
        e = CardEntry(f"k{card_id}", card_id, "normal", "b", 1, "rarity_4", "cool", 0)
        return Match(e, score, score, score, margin=0.1, accepted=accepted)

    box = lambda r, c: CardBox(0, 0, 10, 10, r, c)  # noqa: E731
    sightings = [
        Sighting(0, box(0, 0), [m(7, 0.9, True)]),
        Sighting(0, box(0, 1), [m(7, 0.8, True)]),  # same card twice in one shot
        Sighting(1, box(0, 0), [m(7, 0.85, True)]),  # same card in the next shot: fine
        Sighting(1, box(0, 1), [m(9, 0.4, False)]),  # rejected
        Sighting(1, box(0, 2), [], skipped="clipped"),
    ]
    out = collect(sightings)
    assert [c["card_id"] for c in out["cards"]] == [7]
    assert [s["shot"] for s in out["cards"][0]["seen"]] == [0, 1]
    assert sorted(u["reason"] for u in out["uncertain"]) == ["duplicate_in_screenshot", "low_confidence"]
    assert out["skipped"][0]["reason"] == "clipped"


def test_ignores_banner_in_bar_colour():
    """A wide dark band with white text (common in UI banners) is not a level bar."""
    from PIL import ImageDraw

    layout = Layout.load("default")
    img = Image.new("RGB", (800, 400), (230, 235, 245))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 120, 799, 260), fill=tuple(layout.detector["bar_color"]))
    for i in range(8):  # big white glyph blocks, as in banner titles
        d.rectangle((60 + 85 * i, 150, 110 + 85 * i, 230), fill=(255, 255, 255))
    assert detect_cards(img, layout) == []
