import random

import pytest
from PIL import Image, ImageDraw, ImageFilter

from sekai_card_id.masterdata import expand_entries


def fake_art(seed: int, size: int = 128) -> Image.Image:
    """Procedural stand-in for card art: gradient background + a few shapes."""
    rng = random.Random(seed)
    img = Image.linear_gradient("L").resize((size, size)).rotate(rng.uniform(0, 360))
    c0 = tuple(rng.randrange(256) for _ in range(3))
    c1 = tuple(rng.randrange(256) for _ in range(3))
    img = Image.merge("RGB", [p.point(lambda v, a=a, b=b: a + (b - a) * v // 255) for p, a, b in zip([img] * 3, c0, c1)])
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(4, 8)):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(size * 0.08, size * 0.3)
        color = tuple(rng.randrange(256) for _ in range(3))
        if rng.random() < 0.5:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        else:
            draw.regular_polygon((x, y, r), rng.randint(3, 6), rotation=rng.uniform(0, 90), fill=color)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def fake_cards(n: int) -> list[dict]:
    cards = []
    for i in range(1, n + 1):
        rarity = ["rarity_1", "rarity_2", "rarity_3", "rarity_4", "rarity_birthday"][i % 5]
        trainable = rarity in ("rarity_3", "rarity_4")
        cards.append(
            {
                "id": i,
                "characterId": 1 + i % 26,
                "cardRarityType": rarity,
                "attr": "cool",
                "assetbundleName": f"res{1 + i % 26:03d}_no{i:03d}",
                "releaseAt": 1600000000000 + i,
                "specialTrainingCosts": [{"x": 1}] if trainable else [],
            }
        )
    return cards


@pytest.fixture
def card_assets(tmp_path):
    """(cards, asset_dir) with procedural art for every entry."""
    cards = fake_cards(30)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    for i, e in enumerate(expand_entries(cards)):
        fake_art(i).save(asset_dir / f"{e.key}.png")
    return cards, asset_dir
