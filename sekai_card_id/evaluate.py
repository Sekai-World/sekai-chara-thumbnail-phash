"""Accuracy of a gallery + matcher on synthetic or real labelled card crops."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import find_asset
from .masterdata import CardEntry
from .preprocess import load_image
from .search import Matcher
from .synth import synth_query


def synthetic_queries(
    entries: list[CardEntry], asset_dir: Path, layout, n: int, seed: int = 0, jitter: float = 0.03
) -> list[tuple[Image.Image, CardEntry]]:
    rng = random.Random(seed)
    pool = [e for e in entries if find_asset(asset_dir, e.key)]
    picked = rng.sample(pool, min(n, len(pool)))
    return [(synth_query(load_image(find_asset(asset_dir, e.key), layout.fill), layout, rng, jitter), e) for e in picked]


def labelled_queries(csv_path: Path, entries: list[CardEntry]) -> list[tuple[Image.Image, CardEntry]]:
    """CSV columns: path,card_id[,state]. Paths are relative to the CSV file.

    Each image is a crop of one card (frame included). Without `state`, only
    card-level accuracy is meaningful.
    """
    by_key = {(e.card_id, e.state): e for e in entries}
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            state = (row.get("state") or "normal").strip()
            e = by_key.get((int(row["card_id"]), state))
            if e is None:
                raise ValueError(f"{row}: card/state not in gallery")
            out.append((load_image(csv_path.parent / row["path"]), e))
    return out


def evaluate(matcher: Matcher, queries, rerank_weights=(0.0, None), batch_size: int = 64) -> dict:
    """Top-1/top-5 accuracy for each re-rank weight (None = matcher default).

    Embeddings are computed once and shared by all weights.
    """
    cards = [q for q, _ in queries]
    truth = [e for _, e in queries]
    t0 = time.time()
    vecs = []
    for i in range(0, len(cards), batch_size):
        vecs.append(matcher.embed_cards(cards[i : i + batch_size]))
    vecs = np.concatenate(vecs) if vecs else np.zeros((0, matcher.gallery.embeddings.shape[1]), np.float32)
    embed_s = time.time() - t0

    report = {"queries": len(queries), "embed_ms_per_query": round(1000 * embed_s / max(1, len(queries)), 1), "runs": []}
    for w in rerank_weights:
        weight = matcher.rerank_weight if w is None else w
        t0 = time.time()
        results = matcher.match(cards, top=5, query_vecs=vecs, rerank_weight=weight)
        rank_s = time.time() - t0
        top1 = top5 = card1 = same_char_err = 0
        errors = []
        for res, e in zip(results, truth):
            keys = [m.entry.key for m in res]
            top1 += bool(keys) and keys[0] == e.key
            top5 += e.key in keys
            card1 += bool(res) and res[0].entry.card_id == e.card_id
            if res and res[0].entry.key != e.key:
                same_char_err += res[0].entry.character_id == e.character_id
                errors.append({"truth": e.key, "got": res[0].entry.key, "score": round(res[0].score, 4)})
        n = max(1, len(truth))
        report["runs"].append(
            {
                "rerank_weight": weight,
                "top1": round(top1 / n, 4),
                "top5": round(top5 / n, 4),
                "top1_card_ignoring_state": round(card1 / n, 4),
                "errors": len(errors),
                "errors_same_character": same_char_err,
                "rank_ms_per_query": round(1000 * rank_s / n, 1),
                "error_samples": errors[:20],
            }
        )
    return report
