from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assets import DEFAULT_ASSET_URL, DEFAULT_EXTS, sync_assets
from .embedders import DEFAULT_MODEL, make_embedder
from .gallery import DEFAULT_TEMPLATE_SIZE, Gallery, build_gallery
from .masterdata import DEFAULT_CARDS_URL, expand_entries, load_cards
from .preprocess import Layout, load_image


def _cards_source(args) -> str:
    if args.cards:
        return args.cards
    local = Path(args.asset_dir) / "cards.json"
    return str(local) if local.is_file() else DEFAULT_CARDS_URL


def cmd_sync(args) -> int:
    cards = load_cards(args.cards or DEFAULT_CARDS_URL)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "cards.json").write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")
    entries = expand_entries(cards)
    print(f"{len(cards)} cards -> {len(entries)} art entries")
    report = sync_assets(entries, asset_dir, args.asset_url, tuple(args.ext.split(",")), args.workers)
    print(f"sync: {report.summary()}")
    for key, err in list(report.failed.items())[:10]:
        print(f"  failed {key}: {err}", file=sys.stderr)
    return 1 if report.failed else 0


def cmd_build(args) -> int:
    entries = expand_entries(load_cards(_cards_source(args)))
    gallery_dir = Path(args.gallery)
    previous = None
    if not args.full and (gallery_dir / "meta.json").is_file():
        previous = Gallery.load(gallery_dir)
    embedder = make_embedder(args.model, weights=args.weights, img_size=args.img_size, pool=args.pool, device=args.device)
    layout = Layout.load(args.layout)
    gallery, stats = build_gallery(
        entries,
        Path(args.asset_dir),
        embedder,
        layout,
        previous=previous,
        template_size=args.template_size,
        batch_size=args.batch_size,
    )
    gallery.save(gallery_dir)
    print(f"gallery: {json.dumps(stats)} -> {gallery_dir}")
    return 0


def cmd_update(args) -> int:
    sync_rc = cmd_sync(args)
    build_rc = cmd_build(args)
    return sync_rc or build_rc


def _load_matcher(args):
    from .search import Matcher

    gallery = Gallery.load(Path(args.gallery))
    embedder = make_embedder(gallery.meta["embedder"], weights=args.weights, device=args.device)
    return Matcher(gallery, embedder, top_k=args.top_k, rerank_weight=args.rerank_weight)


def _parse_filters(items: list[str]) -> dict:
    filters: dict[str, list] = {}
    for item in items or []:
        field, _, value = item.partition("=")
        vals = [int(v) if v.isdigit() else v for v in value.split(",")]
        filters.setdefault(field, []).extend(vals)
    return filters


def cmd_match(args) -> int:
    matcher = _load_matcher(args)
    images = [load_image(p) for p in args.images]
    results = matcher.match(images, top=args.top, filters=_parse_filters(args.filter))
    out = [
        {
            "image": str(p),
            "matches": [
                {
                    "card_id": m.entry.card_id,
                    "state": m.entry.state,
                    "key": m.entry.key,
                    "score": round(m.score, 4),
                    "cos": round(m.cos, 4),
                    "ncc": None if m.ncc is None else round(m.ncc, 4),
                }
                for m in res
            ],
        }
        for p, res in zip(args.images, results)
    ]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate, labelled_queries, synthetic_queries

    matcher = _load_matcher(args)
    entries = matcher.gallery.entries
    if args.labels:
        queries = labelled_queries(Path(args.labels), entries)
        source = f"labels:{args.labels}"
    else:
        queries = synthetic_queries(entries, Path(args.asset_dir), matcher.layout, args.samples, args.seed, args.jitter)
        source = f"synthetic(n={len(queries)}, seed={args.seed}, jitter={args.jitter})"
    report = {"source": source, "gallery": matcher.gallery.meta["fingerprint"], **evaluate(matcher, queries)}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    for run in report["runs"]:
        print(
            f"rerank_weight={run['rerank_weight']}: top1={run['top1']:.3f} top5={run['top5']:.3f} "
            f"card-top1={run['top1_card_ignoring_state']:.3f} errors={run['errors']} "
            f"(same character: {run['errors_same_character']})"
        )
    print(f"embed {report['embed_ms_per_query']} ms/query, {report['queries']} queries ({source})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sekai-card-id", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--cards", help=f"cards.json URL or path (default: <asset-dir>/cards.json, else {DEFAULT_CARDS_URL})")
        sp.add_argument("--asset-dir", default="data/assets")

    def sync_args(sp):
        sp.add_argument("--asset-url", default=DEFAULT_ASSET_URL, help="template with {bundle} {state} {ext}")
        sp.add_argument("--ext", default=",".join(DEFAULT_EXTS), help="extensions to try, in order")
        sp.add_argument("--workers", type=int, default=8)

    def model_args(sp):
        sp.add_argument("--gallery", default="data/gallery")
        sp.add_argument("--model", default=DEFAULT_MODEL, help='timm model name, or "tiny" (no download)')
        sp.add_argument("--weights", help="local weights file instead of downloading")
        sp.add_argument("--img-size", type=int, default=224)
        sp.add_argument("--pool", default="cls+avg", choices=["cls", "avg", "cls+avg"])
        sp.add_argument("--layout", default="default", help="layout name or JSON path")
        sp.add_argument("--template-size", type=int, default=DEFAULT_TEMPLATE_SIZE)
        sp.add_argument("--batch-size", type=int, default=64)
        sp.add_argument("--device")
        sp.add_argument("--full", action="store_true", help="ignore the previous gallery and re-embed everything")

    def query_args(sp):
        sp.add_argument("--gallery", default="data/gallery")
        sp.add_argument("--weights", help="local weights file, if the gallery was built with one")
        sp.add_argument("--device")
        sp.add_argument("--top-k", type=int, default=20, help="embedding candidates passed to re-ranking")
        sp.add_argument("--rerank-weight", type=float, default=0.5)

    sp = sub.add_parser("sync", help="download master data and card art")
    data_args(sp)
    sync_args(sp)
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("build", help="build / incrementally update the gallery")
    data_args(sp)
    model_args(sp)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("update", help="sync + build (what CI runs)")
    data_args(sp)
    sync_args(sp)
    model_args(sp)
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("match", help="identify card crops")
    query_args(sp)
    sp.add_argument("images", nargs="+")
    sp.add_argument("--top", type=int, default=5)
    sp.add_argument("--filter", action="append", help="e.g. rarity=rarity_4 or character_id=1,2")
    sp.set_defaults(func=cmd_match)

    sp = sub.add_parser("eval", help="accuracy on synthetic or labelled crops")
    query_args(sp)
    sp.add_argument("--asset-dir", default="data/assets")
    sp.add_argument("--labels", help="CSV path,card_id[,state] of real card crops")
    sp.add_argument("--samples", type=int, default=500)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--jitter", type=float, default=0.03, help="synthetic crop misalignment (fraction of card size)")
    sp.add_argument("--out", help="write the full JSON report here")
    sp.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)
