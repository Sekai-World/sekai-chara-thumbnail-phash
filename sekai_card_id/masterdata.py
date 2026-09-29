"""Card master data -> gallery entries (one per card art state)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .net import fetch_bytes

DEFAULT_CARDS_URL = (
    "https://raw.githubusercontent.com/Sekai-World/sekai-master-db-diff/main/cards.json"
)

STATES = ("normal", "after_training")


@dataclass(frozen=True)
class CardEntry:
    """One gallery item: a single thumbnail art of a card."""

    key: str  # f"{asset_bundle}_{state}", also the asset file stem
    card_id: int
    state: str  # "normal" | "after_training"
    asset_bundle: str
    character_id: int
    rarity: str
    attr: str
    release_at: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CardEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def load_cards(source: str | Path) -> list[dict]:
    """Load cards.json from a URL or a local path."""
    s = str(source)
    if s.startswith(("http://", "https://", "file://")):
        data = fetch_bytes(s)
        if data is None:
            raise FileNotFoundError(s)
        return json.loads(data)
    return json.loads(Path(s).read_text(encoding="utf-8"))


def card_states(card: dict) -> list[str]:
    """Art states a card can appear in.

    Cards with special training (rarity 3/4) have a second art. Some cards ship
    already trained (initialSpecialTrainingStatus == "done"); we still list
    both states and let the asset sync drop whichever art does not exist.
    """
    states = ["normal"]
    if card.get("specialTrainingCosts") or card.get("initialSpecialTrainingStatus") == "done":
        states.append("after_training")
    return states


def expand_entries(cards: list[dict]) -> list[CardEntry]:
    entries = []
    for card in sorted(cards, key=lambda c: c["id"]):
        for state in card_states(card):
            bundle = card["assetbundleName"]
            entries.append(
                CardEntry(
                    key=f"{bundle}_{state}",
                    card_id=card["id"],
                    state=state,
                    asset_bundle=bundle,
                    character_id=card["characterId"],
                    rarity=card["cardRarityType"],
                    attr=card.get("attr", ""),
                    release_at=card.get("releaseAt", 0),
                )
            )
    return entries
