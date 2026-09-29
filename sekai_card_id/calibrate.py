"""Fit `card_to_art` from a few labelled card crops.

Given crops whose card is known, search for the art box inside the crop that
makes the crop's visible (unmasked) pixels best match the official art, by
coordinate descent on the mean normalised cross-correlation over all
samples. Overlay masks are card-relative, so they move with the candidate
box automatically.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocess import Layout, keep_mask

SIZE = 48


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))


class _Sample:
    def __init__(self, crop: Image.Image, art: Image.Image):
        # Pad so candidate boxes may extend past the crop edge.
        self.pad = 0.15
        w, h = crop.size
        pw, ph = round(w * self.pad), round(h * self.pad)
        canvas = Image.new("RGB", (w + 2 * pw, h + 2 * ph), (128, 128, 128))
        canvas.paste(crop, (pw, ph))
        self.canvas, self.w, self.h, self.pw, self.ph = canvas, w, h, pw, ph
        self.art = np.asarray(art.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BOX), np.float32)

    def score(self, box, keep: np.ndarray) -> float:
        l, t, r, b = box
        px = (self.pw + l * self.w, self.ph + t * self.h, self.pw + r * self.w, self.ph + b * self.h)
        region = np.asarray(self.canvas.resize((SIZE, SIZE), Image.Resampling.BOX, box=px), np.float32)
        return _ncc(region[keep], self.art[keep])


def _objective(layout: Layout, box, samples) -> float:
    keep = keep_mask(layout.replace(card_to_art=list(box)), SIZE)
    return float(np.mean([s.score(box, keep) for s in samples]))


def fit_card_to_art(layout: Layout, pairs: list[tuple[Image.Image, Image.Image]], log=print) -> tuple[Layout, dict]:
    """`pairs` = [(card crop, official art)]. Returns the refined layout and a report."""
    samples = [_Sample(c, a) for c, a in pairs]
    box = np.array(layout.card_to_art, dtype=np.float64)
    best = _objective(layout, box, samples)
    start = best
    for step in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        improved = True
        while improved:
            improved = False
            # Move each edge, shift the whole box, or scale it about its centre.
            moves = []
            for i in range(4):
                for sgn in (-1, 1):
                    d = np.zeros(4)
                    d[i] = sgn * step
                    moves.append(d)
            for sgn in (-1, 1):
                moves += [np.array([sgn * step, 0, sgn * step, 0]), np.array([0, sgn * step, 0, sgn * step])]
                moves.append(np.array([-sgn * step, -sgn * step, sgn * step, sgn * step]))
            for d in moves:
                cand = box + d
                if cand[2] - cand[0] < 0.5 or cand[3] - cand[1] < 0.5:
                    continue
                val = _objective(layout, cand, samples)
                if val > best + 1e-5:
                    box, best, improved = cand, val, True
        log(f"  step {step}: mean NCC {best:.4f} box {np.round(box, 4).tolist()}")
    fitted = layout.replace(card_to_art=[round(float(v), 4) for v in box])
    keep = keep_mask(fitted, SIZE)
    report = {
        "samples": len(samples),
        "mean_ncc_before": round(start, 4),
        "mean_ncc_after": round(best, 4),
        "per_sample_after": [round(s.score(box, keep), 4) for s in samples],
        "card_to_art": fitted.card_to_art,
    }
    return fitted, report
