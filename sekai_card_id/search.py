"""Query: card crop -> ranked gallery entries.

1. Embed the masked art region and take the top-k gallery rows by cosine.
2. Re-rank those candidates by normalised cross-correlation of small colour
   templates, maximised over a few shift/scale perturbations of the crop.
   Embeddings are good at "which art is this" under distortion; the pixel
   check separates near-identical arts (same character, similar pose) and
   absorbs imperfect crop alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .embedders import Embedder
from .gallery import Gallery, fingerprint
from .masterdata import CardEntry
from .preprocess import apply_masks, art_from_card, art_template_variants, keep_mask


@dataclass
class Match:
    entry: CardEntry
    score: float
    cos: float
    ncc: float | None


def _zero_mean_l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.mean(axis=-1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


class Matcher:
    def __init__(
        self,
        gallery: Gallery,
        embedder: Embedder,
        top_k: int = 20,
        rerank_weight: float = 0.5,
        shifts: tuple[float, ...] = (-0.03, 0.0, 0.03),
        scales: tuple[float, ...] = (0.97, 1.0, 1.03),
    ):
        self.gallery = gallery
        self.layout = gallery.layout
        self.template_size = gallery.meta["template_size"]
        fp = fingerprint(embedder, self.layout, self.template_size)
        if fp != gallery.meta["fingerprint"]:
            raise ValueError(
                "embedder does not match the gallery it is querying\n"
                f"  gallery: {gallery.meta['fingerprint']}\n  query:   {fp}"
            )
        self.embedder = embedder
        self.top_k = top_k
        self.rerank_weight = rerank_weight
        self.shifts = shifts
        self.scales = scales
        self.keep = keep_mask(self.layout, self.template_size)
        self.tpl_vecs = _zero_mean_l2(gallery.templates[:, self.keep].reshape(len(gallery.entries), -1))

    def _allowed(self, filters: dict | None) -> np.ndarray | None:
        if not filters:
            return None
        ok = np.ones(len(self.gallery.entries), dtype=bool)
        for field, values in filters.items():
            values = set(values) if isinstance(values, (list, tuple, set)) else {values}
            ok &= np.array([getattr(e, field) in values for e in self.gallery.entries])
        return ok

    def embed_cards(self, cards: list[Image.Image]) -> np.ndarray:
        arts = [apply_masks(art_from_card(c, self.layout), self.layout) for c in cards]
        return self.embedder.embed(arts)

    def match(
        self,
        cards: list[Image.Image],
        top: int = 5,
        filters: dict | None = None,
        query_vecs: np.ndarray | None = None,
        rerank_weight: float | None = None,
    ) -> list[list[Match]]:
        """`cards` are RGB crops of single cards (frame included) from a screenshot.

        `query_vecs` lets callers reuse embeddings (e.g. to compare re-rank
        weights without re-running the model).
        """
        w = self.rerank_weight if rerank_weight is None else rerank_weight
        q = self.embed_cards(cards) if query_vecs is None else query_vecs
        cos_all = q @ self.gallery.embeddings.T
        allowed = self._allowed(filters)
        if allowed is not None:
            cos_all[:, ~allowed] = -np.inf
        n_valid = len(self.gallery.entries) if allowed is None else int(allowed.sum())
        k = min(self.top_k, n_valid)
        results = []
        for card, cos in zip(cards, cos_all):
            if k == 0:
                results.append([])
                continue
            cand = np.argpartition(-cos, k - 1)[:k]
            ncc = None
            score = cos[cand]
            if w > 0:
                variants = art_template_variants(card, self.layout, self.template_size, self.shifts, self.scales)
                v = _zero_mean_l2(variants[:, self.keep].reshape(len(variants), -1))
                ncc = (v @ self.tpl_vecs[cand].T).max(axis=0)
                score = (1 - w) * cos[cand] + w * ncc
            order = np.argsort(-score)[:top]
            results.append(
                [
                    Match(
                        entry=self.gallery.entries[cand[i]],
                        score=float(score[i]),
                        cos=float(cos[cand[i]]),
                        ncc=None if ncc is None else float(ncc[i]),
                    )
                    for i in order
                ]
            )
        return results
