"""End-to-end recognition of the labelled real screenshots.

Needs a gallery built from real card art (`sekai-card-id update`), so it only
runs when SEKAI_GALLERY points at one, e.g.
    SEKAI_GALLERY=data/gallery pytest tests/test_real_recognition.py
"""

import os
from pathlib import Path

import pytest

GALLERY = os.environ.get("SEKAI_GALLERY")
LABELS = Path(__file__).parent / "fixtures" / "screenshots" / "labels.csv"

pytestmark = pytest.mark.skipif(not GALLERY, reason="set SEKAI_GALLERY to a gallery built from real card art")


def test_labelled_screenshots_are_recognised():
    from sekai_card_id.embedders import make_embedder
    from sekai_card_id.evaluate import evaluate, labelled_queries
    from sekai_card_id.gallery import Gallery
    from sekai_card_id.search import Matcher

    gallery = Gallery.load(Path(GALLERY))
    matcher = Matcher(gallery, make_embedder(gallery.meta["embedder"], weights=os.environ.get("SEKAI_WEIGHTS")))
    queries = labelled_queries(LABELS, gallery.entries, gallery.layout)
    assert len(queries) == 80
    report = evaluate(matcher, queries, rerank_weights=(None,))
    (run,) = report["runs"]
    assert run["top1"] == 1.0  # card and training state
    if matcher.reject:
        assert run["accepted"] == 1.0 and run["accepted_precision"] == 1.0
