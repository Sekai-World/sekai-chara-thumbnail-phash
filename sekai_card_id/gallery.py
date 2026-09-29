"""Gallery = one embedding + one small colour template per card art.

Stored as a directory:
  meta.json        fingerprint of model + layout + preprocessing, counts
  entries.json     CardEntry dicts (+ asset sha256) in row order
  embeddings.npy   float16 (N, D), L2-normalised
  templates.npy    uint8 (N, S, S, 3) masked art thumbnails for re-ranking
  reject.json      optional accept thresholds (from `eval --fit-reject`)

Building is incremental: rows whose asset bytes and fingerprint are unchanged
are copied from the previous gallery, so adding new cards only embeds the new
cards.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import find_asset
from .embedders import Embedder
from .masterdata import CardEntry
from .preprocess import PREPROCESS_VERSION, Layout, apply_masks, load_image, make_template

DEFAULT_TEMPLATE_SIZE = 32
REJECT_FILE = "reject.json"


@dataclass
class Gallery:
    meta: dict
    entries: list[CardEntry]
    hashes: list[str]
    embeddings: np.ndarray  # float32 (N, D)
    templates: np.ndarray  # uint8 (N, S, S, 3)
    reject: dict | None = None  # accept thresholds fitted for this gallery (reject.json)

    @property
    def layout(self) -> Layout:
        return Layout.from_dict(self.meta["layout"])

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{**e.to_dict(), "sha256": h} for e, h in zip(self.entries, self.hashes)]
        # Write data first and meta last, so a crash never leaves a meta.json
        # pointing at mismatched arrays.
        np.save(out_dir / "embeddings.npy", self.embeddings.astype(np.float16))
        np.save(out_dir / "templates.npy", self.templates)
        (out_dir / "entries.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        (out_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "Gallery":
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        rows = json.loads((out_dir / "entries.json").read_text(encoding="utf-8"))
        emb = np.load(out_dir / "embeddings.npy").astype(np.float32)
        tpl = np.load(out_dir / "templates.npy")
        if not (len(rows) == len(emb) == len(tpl) == meta["count"]):
            raise ValueError(f"{out_dir}: gallery files are inconsistent")
        reject = None
        reject_path = out_dir / REJECT_FILE
        if reject_path.is_file():
            data = json.loads(reject_path.read_text(encoding="utf-8"))
            # Thresholds are only meaningful for the model/layout they were fitted on.
            if data.get("fingerprint") == meta["fingerprint"]:
                reject = data
        return cls(meta, [CardEntry.from_dict(r) for r in rows], [r["sha256"] for r in rows], emb, tpl, reject)

    def save_reject(self, out_dir: Path, thresholds: dict) -> None:
        data = {"fingerprint": self.meta["fingerprint"], **thresholds}
        (out_dir / REJECT_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        self.reject = data


def fingerprint(embedder: Embedder, layout: Layout, template_size: int) -> str:
    return f"{embedder.fingerprint()}|layout={layout.fingerprint()}|pre={PREPROCESS_VERSION}|tpl={template_size}"


def build_gallery(
    entries: list[CardEntry],
    asset_dir: Path,
    embedder: Embedder,
    layout: Layout,
    previous: Gallery | None = None,
    template_size: int = DEFAULT_TEMPLATE_SIZE,
    batch_size: int = 64,
    log=print,
) -> tuple[Gallery, dict]:
    fp = fingerprint(embedder, layout, template_size)
    reuse: dict[str, tuple[str, int]] = {}
    if previous is not None and previous.meta.get("fingerprint") == fp:
        reuse = {e.key: (h, i) for i, (e, h) in enumerate(zip(previous.entries, previous.hashes))}
    elif previous is not None:
        log("model/layout changed since the previous gallery: re-embedding everything")

    kept: list[CardEntry] = []
    hashes: list[str] = []
    vec_rows: list[np.ndarray | None] = []
    tpl_rows: list[np.ndarray | None] = []
    todo: list[tuple[int, Path]] = []
    missing = []
    for e in entries:
        path = find_asset(asset_dir, e.key)
        if path is None:
            missing.append(e.key)
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        idx = len(kept)
        kept.append(e)
        hashes.append(h)
        prev = reuse.get(e.key)
        if prev and prev[0] == h:
            vec_rows.append(previous.embeddings[prev[1]])
            tpl_rows.append(previous.templates[prev[1]])
        else:
            vec_rows.append(None)
            tpl_rows.append(None)
            todo.append((idx, path))

    t0 = time.time()
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        arts = [apply_masks(load_image(p, layout.fill), layout) for _, p in chunk]
        vecs = embedder.embed(arts)
        for (idx, _), art, v in zip(chunk, arts, vecs):
            vec_rows[idx] = v
            tpl_rows[idx] = make_template(art, template_size)
        log(f"  embedded {min(start + batch_size, len(todo))}/{len(todo)}")

    dim = len(vec_rows[0]) if vec_rows else 0
    gallery = Gallery(
        meta={
            "fingerprint": fp,
            "embedder": embedder.spec,
            "layout": layout.to_dict(),
            "preprocess_version": PREPROCESS_VERSION,
            "template_size": template_size,
            "count": len(kept),
            "dim": dim,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        entries=kept,
        hashes=hashes,
        embeddings=np.stack(vec_rows).astype(np.float32) if kept else np.zeros((0, dim), np.float32),
        templates=np.stack(tpl_rows) if kept else np.zeros((0, template_size, template_size, 3), np.uint8),
    )
    stats = {
        "total": len(kept),
        "embedded": len(todo),
        "reused": len(kept) - len(todo),
        "missing_assets": len(missing),
        "seconds": round(time.time() - t0, 1),
    }
    return gallery, stats
