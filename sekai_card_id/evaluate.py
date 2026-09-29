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


def labelled_queries(csv_path: Path, entries: list[CardEntry], layout=None) -> list[tuple[Image.Image, CardEntry]]:
    """Labelled card crops. Paths are relative to the CSV file. Two formats:

    - ``path,card_id[,state]``: each image is a crop of one card (frame
      included), e.g. the labels.csv written by ``scan --debug-dir``;
    - ``screenshot,row,col,card_id[,state]``: cards are located in the
      screenshot with the layout's detector (needs `layout`), so a handful of
      screenshots plus a small CSV make a test set.

    Without `state`, only card-level accuracy is meaningful. Rows without a
    card_id are skipped.
    """
    from .detect import detect_cards

    by_key = {(e.card_id, e.state): e for e in entries}
    shots: dict[str, dict] = {}
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row.get("card_id") or "").strip():
                continue  # unlabelled
            state = (row.get("state") or "normal").strip()
            e = by_key.get((int(row["card_id"]), state))
            if e is None:
                raise ValueError(f"{row}: card/state not in gallery")
            if row.get("screenshot"):
                name = row["screenshot"]
                if name not in shots:
                    if layout is None:
                        raise ValueError("screenshot labels need a layout to detect cards")
                    img = load_image(csv_path.parent / name)
                    shots[name] = {"img": img, "boxes": {(b.row, b.col): b for b in detect_cards(img, layout)}}
                box = shots[name]["boxes"].get((int(row["row"]), int(row["col"])))
                if box is None:
                    raise ValueError(f"{row}: no card detected at that position")
                out.append((box.crop(shots[name]["img"]), e))
            else:
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
        top1 = top5 = card1 = same_char_err = acc = acc_ok = 0
        errors = []
        for res, e in zip(results, truth):
            if res and res[0].accepted:
                acc += 1
                acc_ok += res[0].entry.card_id == e.card_id
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
                **(
                    {"accepted": round(acc / n, 4), "accepted_precision": round(acc_ok / max(1, acc), 4)}
                    if matcher.reject
                    else {}
                ),
                "error_samples": errors[:20],
            }
        )
    return report


def fit_reject(matcher: Matcher, queries, target_error: float = 0.01, batch_size: int = 64) -> dict:
    """Pick (min_score, min_margin) for accepting a best match.

    Every query is matched twice: normally ("known"), and with its true card
    removed from the gallery ("unknown", i.e. a card we do not have yet).
    An error is accepting an unknown query or accepting a wrong card; we
    maximise correctly accepted known queries while keeping errors at or
    below `target_error` of all trials.
    """
    cards = [q for q, _ in queries]
    truth = [e for _, e in queries]
    vecs = np.concatenate([matcher.embed_cards(cards[i : i + batch_size]) for i in range(0, len(cards), batch_size)])
    known = matcher.match(cards, top=1, query_vecs=vecs)
    unknown = matcher.match(cards, top=1, query_vecs=vecs, exclude_card_ids=[{e.card_id} for e in truth])

    def feats(results):
        rows = [(r[0].score, min(r[0].margin, 1.0)) if r else (-1.0, 0.0) for r in results]
        return np.array(rows, dtype=np.float64).reshape(-1, 2)

    kf, uf = feats(known), feats(unknown)
    correct = np.array([bool(r) and r[0].entry.card_id == e.card_id for r, e in zip(known, truth)])
    n = len(truth)
    score_grid = np.unique(np.concatenate([[-np.inf], kf[:, 0], uf[:, 0]]))
    margin_grid = np.unique(np.concatenate([[0.0], np.quantile(np.concatenate([kf[:, 1], uf[:, 1]]), np.linspace(0, 1, 41))]))
    best = None
    for m in margin_grid:
        k_ok_m = kf[:, 1] >= m
        u_ok_m = uf[:, 1] >= m
        for s in score_grid:
            k_acc = k_ok_m & (kf[:, 0] >= s)
            errors = int((k_acc & ~correct).sum() + (u_ok_m & (uf[:, 0] >= s)).sum())
            if errors > target_error * 2 * n:
                continue
            good = int((k_acc & correct).sum())
            if best is None or good > best[0] or (good == best[0] and errors < best[1]):
                best = (good, errors, float(s), float(m))
            break  # higher s only accepts fewer: the first feasible s is best for this m
    if best is None:
        raise ValueError("no thresholds meet the target error")
    good, errors, s, m = best
    # Any score threshold up to the lowest correctly accepted score keeps the
    # same correct acceptances. Raise it to the middle of the gap below that
    # score: a second, independent guard (often the margin alone separated
    # the fitting data) with headroom on both sides.
    acc_ok = (kf[:, 1] >= m) & (kf[:, 0] >= s) & correct
    if acc_ok.any():
        c_min = kf[acc_ok, 0].min()
        err_scores = np.concatenate([kf[~correct, 0], uf[:, 0]])
        below = err_scores[err_scores < c_min]
        if below.size:
            s = max(s, float((below.max() + c_min) / 2))
    k_acc = (kf[:, 0] >= s) & (kf[:, 1] >= m)
    u_acc = (uf[:, 0] >= s) & (uf[:, 1] >= m)
    return {
        "min_score": round(s, 4) if np.isfinite(s) else -1.0,
        "min_margin": round(m, 4),
        "target_error": target_error,
        "queries": n,
        "known_top1": round(float(correct.mean()), 4),
        "known_accepted": round(float(k_acc.mean()), 4),
        "known_accepted_correct": round(float((k_acc & correct).mean()), 4),
        "known_accepted_wrong": round(float((k_acc & ~correct).mean()), 4),
        "unknown_accepted": round(float(u_acc.mean()), 4),
    }
