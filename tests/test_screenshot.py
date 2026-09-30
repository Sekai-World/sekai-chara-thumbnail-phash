import random

import pytest
from PIL import Image

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


def _bar_rect(box, layout):
    """Where render_card draws the level bar of a card at `box`."""
    x0, _, x1, y1 = box
    w = x1 - x0
    by1 = y1 - layout.detector["bottom_margin"] * w
    return x0, by1 - 0.17 * w, x1, by1


def test_find_bars_joins_pieces_of_one_bar():
    """Seen in a real shot: the first row only covers the bar's right end, the second
    row's left run starts a separate piece, the text splits the rows below, and the
    last row joins the left piece only. The pieces are still one bar."""
    import numpy as np

    from sekai_card_id.detect import find_bars

    mask = np.zeros((30, 200), bool)
    mask[2, 70:150] = True
    mask[3, 0:65] = mask[3, 72:150] = True
    mask[4:23, 75:150] = True
    mask[4:23, 0:20] = True  # left of the text: too short to count on its own
    mask[23, 0:100] = True
    (bar,) = find_bars(mask, min_width=40, max_gap=2, max_vgap=22)
    assert (bar.x0, bar.x1, bar.y0, bar.y1) == (0, 150, 2, 24)


def test_detects_bars_clipped_by_master_rank_badge():
    """A master rank badge over the right end of the bar leaves a squat bar that still counts."""
    from PIL import ImageDraw

    layout = Layout.load("default")
    rng = random.Random(7)
    img, truth = render_screenshot([fake_art(i) for i in range(15)], layout, rng, cols=5, size=(900, 560))
    d = ImageDraw.Draw(img)
    for box in truth[::2]:
        x0, y0, x1, y1 = _bar_rect(box, layout)
        d.rectangle((x0 + 0.68 * (x1 - x0), y0 - 4, x1, y1 + 2), fill=(80, 190, 200))
    boxes = detect_cards(img, layout)
    assert len(boxes) == 15
    for b, t in zip(boxes, truth):
        assert max(abs(u - v) for u, v in zip(b.as_list(), t)) <= 2


def test_stray_wide_bar_does_not_set_card_size():
    """One bar-like patch wider than the cards (e.g. in the header) must not scale every box."""
    layout = Layout.load("default")
    rng = random.Random(8)
    img, truth = render_screenshot(
        [fake_art(i) for i in range(15)], layout, rng, cols=5, size=(900, 600), origin=(60, 120)
    )
    wide = render_card(fake_art(99), layout, rng, 210)  # bar 1.6x the width of the real ones
    img.paste(wide.crop((0, 150, 210, 210)), (600, 0))
    boxes = detect_cards(img, layout)
    assert len(boxes) == 15
    for b, t in zip(boxes, truth):
        assert max(abs(u - v) for u, v in zip(b.as_list(), t)) <= 2


def test_ignores_bars_without_level_text():
    """Sorted by talent (or on other screens) the bar shows other text than "Lv.": not the level view."""
    from PIL import ImageDraw

    layout = Layout.load("default")
    rng = random.Random(9)
    img, truth = render_screenshot([fake_art(i) for i in range(15)], layout, rng, cols=5, size=(900, 560))
    d = ImageDraw.Draw(img)
    for box in truth:
        x0, y0, x1, y1 = _bar_rect(box, layout)
        d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=tuple(layout.detector["bar_color"]))
        for i in range(5):  # five digit blobs, e.g. "35801"
            gx = x0 + (x1 - x0) * (0.06 + 0.09 * i)
            d.rectangle((gx, y0 + 0.2 * (y1 - y0), gx + 0.05 * (x1 - x0), y1 - 0.2 * (y1 - y0)), fill=(250, 250, 250))
    assert detect_cards(img, layout) == []


def test_scan_accepts_only_the_ten_column_card_list(card_assets):
    g, asset_dir, layout = _gallery(card_assets)
    matcher = Matcher(g, TinyEmbedder())
    rng = random.Random(4)
    arts = [load_image(asset_dir / f"{e.key}.png") for e in g.entries]
    ten = {"cols": 10, "card_px": 96, "gap": 16, "size": (1300, 560)}
    other_screen = render_screenshot(arts[:15], layout, rng, cols=5, size=(900, 560))[0]
    full = render_screenshot((arts * 2)[:40], layout, rng, **ten)[0]
    last_page = render_screenshot(arts[:14], layout, rng, **ten)[0]  # one full row + 4
    res = scan(matcher, [other_screen, full, last_page])
    assert res.rejected == {0: "not_card_list"}
    assert res.detected == [15, 40, 14]
    assert {s.shot for s in res.sightings} == {1, 2}


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
    result = collect(scan(matcher, shots, grid_cols=5).sightings)
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
    sightings = scan(matcher, [img], grid_cols=5).sightings
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

    box = lambda r, c: CardBox(0, 0, 10, 10, r, c)
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
