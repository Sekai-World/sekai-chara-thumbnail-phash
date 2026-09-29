"""Query: card crop -> ranked gallery entries.

1. Embed the masked art region and take the top-k gallery rows by cosine.
2. Re-rank those candidates by normalised cross-correlation of small colour
   templates, maximised over a few shift/scale perturbations of the crop.
   Embeddings are good at "which art is this" under distortion; the pixel
   check separates near-identical arts (same character, similar pose) and
   absorbs imperfect crop alignment.
3. Accept or reject the best match. `margin` is the lead of a match over
   the best candidate of a *different* card (the other art state of the
   same card is not a rival). A query whose card is missing from the
   gallery (e.g. released after the last gallery build) still gets a best
   match, but with a low score and/or margin; thresholds for both are fitted
   by `eval --fit-reject` and stored next to the gallery.

Queries can carry extra card-relative occlusion boxes (e.g. the part of a
card scrolled under the panel edge); those pixels are ignored on the query
side for both the embedding and the pixel comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .embedders import Embedder
from .gallery import Gallery, fingerprint
from .masterdata import CardEntry
from .preprocess import Box, apply_masks, art_from_card, art_template_variants, keep_mask


@dataclass
class Match:
    entry: CardEntry
    score: float
    cos: float
    ncc: float | None
    margin: float = 0.0  # score lead over the best candidate of another card
    accepted: bool | None = None  # None when no reject thresholds are configured


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
        reject: dict | None = None,
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
        self.card_ids = np.array([e.card_id for e in gallery.entries])
        self.reject = reject if reject is not None else gallery.reject

    def is_accepted(self, m: Match) -> bool | None:
        if not self.reject:
            return None
        return m.score >= self.reject["min_score"] and m.margin >= self.reject["min_margin"]

    def _allowed(self, filters: dict | None) -> np.ndarray | None:
        if not filters:
            return None
        ok = np.ones(len(self.gallery.entries), dtype=bool)
        for field, values in filters.items():
            values = set(values) if isinstance(values, (list, tuple, set)) else {values}
            ok &= np.array([getattr(e, field) in values for e in self.gallery.entries])
        return ok

    def _query_layout(self, occlusion: list[Box] | None):
        return self.layout if not occlusion else self.layout.replace(card_masks=list(self.layout.card_masks) + list(occlusion))

    def embed_cards(self, cards: list[Image.Image], occlusions: list[list[Box] | None] | None = None) -> np.ndarray:
        occlusions = occlusions or [None] * len(cards)
        arts = []
        for c, occ in zip(cards, occlusions):
            lay = self._query_layout(occ)
            arts.append(apply_masks(art_from_card(c, self.layout), lay))
        return self.embedder.embed(arts)

    def match(
        self,
        cards: list[Image.Image],
        top: int = 5,
        filters: dict | None = None,
        query_vecs: np.ndarray | None = None,
        rerank_weight: float | None = None,
        occlusions: list[list[Box] | None] | None = None,
        exclude_card_ids: list[set[int]] | None = None,
    ) -> list[list[Match]]:
        """`cards` are RGB crops of single cards (frame included) from a screenshot.

        `occlusions[i]` lists card-relative boxes hidden in card i.
        `exclude_card_ids[i]` removes cards from the candidates of query i
        (used to simulate cards missing from the gallery).
        `query_vecs` lets callers reuse embeddings (e.g. to compare re-rank
        weights without re-running the model).
        """
        w = self.rerank_weight if rerank_weight is None else rerank_weight
        occlusions = occlusions or [None] * len(cards)
        q = self.embed_cards(cards, occlusions) if query_vecs is None else query_vecs
        cos_all = q @ self.gallery.embeddings.T
        allowed = self._allowed(filters)
        if allowed is not None:
            cos_all[:, ~allowed] = -np.inf
        results = []
        for i, (card, cos, occ) in enumerate(zip(cards, cos_all, occlusions)):
            if exclude_card_ids and exclude_card_ids[i]:
                cos = cos.copy()
                cos[np.isin(self.card_ids, list(exclude_card_ids[i]))] = -np.inf
            k = min(self.top_k, int(np.isfinite(cos).sum()))
            if k == 0:
                results.append([])
                continue
            cand = np.argpartition(-cos, k - 1)[:k]
            ncc = None
            score = cos[cand]
            if w > 0:
                variants = art_template_variants(card, self.layout, self.template_size, self.shifts, self.scales)
                if occ:
                    keep = keep_mask(self._query_layout(occ), self.template_size)
                    g = _zero_mean_l2(self.gallery.templates[cand][:, keep].reshape(k, -1))
                else:
                    keep, g = self.keep, self.tpl_vecs[cand]
                v = _zero_mean_l2(variants[:, keep].reshape(len(variants), -1))
                ncc = (v @ g.T).max(axis=0)
                score = (1 - w) * cos[cand] + w * ncc
            order = np.argsort(-score)
            ids = self.card_ids[cand]
            matches = []
            for j in order[:top]:
                rivals = score[ids != ids[j]]
                margin = float(score[j] - rivals.max()) if rivals.size else float("inf")
                m = Match(
                    entry=self.gallery.entries[cand[j]],
                    score=float(score[j]),
                    cos=float(cos[cand[j]]),
                    ncc=None if ncc is None else float(ncc[j]),
                    margin=margin,
                )
                m.accepted = self.is_accepted(m)
                matches.append(m)
            results.append(matches)
        return results
