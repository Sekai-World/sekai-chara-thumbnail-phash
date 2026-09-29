import json
import random

import numpy as np
import pytest
from PIL import Image

from conftest import fake_art, fake_cards
from sekai_card_id.assets import sync_assets
from sekai_card_id.embedders import TinyEmbedder
from sekai_card_id.evaluate import evaluate, synthetic_queries
from sekai_card_id.gallery import Gallery, build_gallery
from sekai_card_id.masterdata import card_states, expand_entries
from sekai_card_id.preprocess import Layout, art_template_variants, load_image
from sekai_card_id.search import Matcher
from sekai_card_id.synth import synth_query


def test_card_states():
    assert card_states({"specialTrainingCosts": []}) == ["normal"]
    assert card_states({"specialTrainingCosts": [{}]}) == ["normal", "after_training"]
    assert card_states({"initialSpecialTrainingStatus": "done"}) == ["normal", "after_training"]
    entries = expand_entries(fake_cards(5))
    assert [e.key for e in entries if e.card_id == 3] == ["res004_no003_normal", "res004_no003_after_training"]


def test_sync_downloads_then_caches(tmp_path):
    src = tmp_path / "cdn"
    src.mkdir()
    entries = expand_entries(fake_cards(3))
    present = entries[:-1]
    for e in present:
        fake_art(0).save(src / f"{e.asset_bundle}_{e.state}.png")
    url = src.as_uri() + "/{bundle}_{state}.{ext}"
    dest = tmp_path / "assets"

    r1 = sync_assets(entries, dest, url, exts=("webp", "png"), workers=2)
    assert sorted(r1.downloaded) == sorted(e.key for e in present)
    assert r1.missing == [entries[-1].key] and not r1.failed
    assert json.loads((dest / "_sync_report.json").read_text())["missing"] == [entries[-1].key]

    r2 = sync_assets(entries, dest, url, workers=2)
    assert sorted(r2.cached) == sorted(e.key for e in present) and not r2.downloaded


def test_load_image_composites_alpha():
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    assert load_image(img).getpixel((0, 0)) == (128, 128, 128)


def test_build_is_incremental(card_assets, tmp_path):
    cards, asset_dir = card_assets
    entries = expand_entries(cards)
    layout = Layout.load("default")
    g1, s1 = build_gallery(entries, asset_dir, TinyEmbedder(), layout, log=lambda *_: None)
    assert s1["embedded"] == len(entries) and s1["reused"] == 0
    g1.save(tmp_path / "g")
    loaded = Gallery.load(tmp_path / "g")
    np.testing.assert_allclose(loaded.embeddings, g1.embeddings, atol=1e-3)

    # A new card and a re-issued asset: only those two get embedded.
    fake_art(999).save(asset_dir / f"{entries[0].key}.png")
    new_cards = cards + fake_cards(31)[-1:]
    new_entries = expand_entries(new_cards)
    new_key = new_entries[-1].key
    fake_art(1000).save(asset_dir / f"{new_key}.png")
    g2, s2 = build_gallery(new_entries, asset_dir, TinyEmbedder(), layout, previous=loaded, log=lambda *_: None)
    assert s2["embedded"] == 2 and s2["reused"] == len(entries) - 1
    assert g2.entries[-1].key == new_key

    # Changing the layout invalidates every stored vector.
    other = layout.replace(card_masks=[])
    _, s3 = build_gallery(new_entries, asset_dir, TinyEmbedder(), other, previous=g2, log=lambda *_: None)
    assert s3["embedded"] == len(new_entries)


def test_build_skips_missing_assets(card_assets):
    cards, asset_dir = card_assets
    entries = expand_entries(cards)
    (asset_dir / f"{entries[0].key}.png").unlink()
    g, s = build_gallery(entries, asset_dir, TinyEmbedder(), Layout.load(), log=lambda *_: None)
    assert s["missing_assets"] == 1 and g.meta["count"] == len(entries) - 1


def test_matcher_rejects_other_embedder(card_assets):
    cards, asset_dir = card_assets
    g, _ = build_gallery(expand_entries(cards), asset_dir, TinyEmbedder(), Layout.load(), log=lambda *_: None)
    other = TinyEmbedder()
    other.spec = {**other.spec, "version": 2}
    with pytest.raises(ValueError):
        Matcher(g, other)


def test_template_variants_shape():
    card = fake_art(1, 150)
    v = art_template_variants(card, Layout.load(), 32)
    assert v.shape == (27, 32, 32, 3) and v.dtype == np.uint8


def test_synthetic_queries_are_recognised(card_assets):
    cards, asset_dir = card_assets
    entries = expand_entries(cards)
    layout = Layout.load()
    g, _ = build_gallery(entries, asset_dir, TinyEmbedder(), layout, log=lambda *_: None)
    matcher = Matcher(g, TinyEmbedder())
    queries = synthetic_queries(g.entries, asset_dir, layout, n=40, seed=1)
    report = evaluate(matcher, queries)
    by_w = {r["rerank_weight"]: r for r in report["runs"]}
    assert by_w[0.5]["top1"] >= 0.95
    assert by_w[0.5]["top5"] == 1.0


def test_filters_restrict_candidates(card_assets):
    cards, asset_dir = card_assets
    layout = Layout.load()
    g, _ = build_gallery(expand_entries(cards), asset_dir, TinyEmbedder(), layout, log=lambda *_: None)
    matcher = Matcher(g, TinyEmbedder())
    q = synth_query(load_image(asset_dir / f"{g.entries[0].key}.png"), layout, random.Random(0))
    (res,) = matcher.match([q], top=50, filters={"rarity": ["rarity_4"]})
    assert res and all(m.entry.rarity == "rarity_4" for m in res)
