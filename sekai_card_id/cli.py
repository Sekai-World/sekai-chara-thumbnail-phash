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
    reject = None
    if args.min_score is not None or args.min_margin is not None:
        base = gallery.reject or {"min_score": -1.0, "min_margin": 0.0}
        reject = {
            "min_score": base["min_score"] if args.min_score is None else args.min_score,
            "min_margin": base["min_margin"] if args.min_margin is None else args.min_margin,
        }
    return Matcher(gallery, embedder, top_k=args.top_k, rerank_weight=args.rerank_weight, reject=reject)


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
                    "margin": round(m.margin, 4) if m.margin != float("inf") else None,
                    "accepted": m.accepted,
                }
                for m in res
            ],
        }
        for p, res in zip(args.images, results)
    ]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate, fit_reject, labelled_queries, synthetic_queries

    matcher = _load_matcher(args)
    entries = matcher.gallery.entries
    if args.labels:
        queries = labelled_queries(Path(args.labels), entries, matcher.layout)
        source = f"labels:{args.labels}"
    else:
        queries = synthetic_queries(entries, Path(args.asset_dir), matcher.layout, args.samples, args.seed, args.jitter)
        source = f"synthetic(n={len(queries)}, seed={args.seed}, jitter={args.jitter})"
    fitted = None
    if args.fit_reject:
        fitted = fit_reject(matcher, queries, target_error=args.target_error)
        matcher.gallery.save_reject(Path(args.gallery), {**fitted, "fitted_on": source})
        matcher.reject = matcher.gallery.reject
    report = {"source": source, "gallery": matcher.gallery.meta["fingerprint"], **evaluate(matcher, queries)}
    if fitted:
        report["reject_fit"] = fitted
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
    if fitted:
        print(
            f"reject thresholds -> {args.gallery}/reject.json: min_score={fitted['min_score']} "
            f"min_margin={fitted['min_margin']}; known accepted {fitted['known_accepted']:.3f} "
            f"(wrong {fitted['known_accepted_wrong']:.3f}), unknown accepted {fitted['unknown_accepted']:.3f}"
        )
    return 0


def cmd_detect(args) -> int:
    from .detect import detect_cards

    layout = Layout.load(args.layout)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for path in args.images:
        img = load_image(path)
        boxes = detect_cards(img, layout)
        report.append(
            {
                "image": path,
                "cards": [{"row": b.row, "col": b.col, "box": b.as_list(), "clip_top": round(b.clip_top, 3)} for b in boxes],
            }
        )
        if out_dir:
            for b in boxes:
                b.crop(img).save(out_dir / f"{Path(path).stem}_r{b.row}c{b.col}.png")
        print(f"{path}: {len(boxes)} cards", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cmd_scan(args) -> int:
    from .scan import collect, scan, write_debug

    matcher = _load_matcher(args)
    detect_layout = Layout.load(args.layout) if args.layout else None
    if not matcher.reject:
        print("warning: no reject thresholds for this gallery; run `eval --fit-reject` first", file=sys.stderr)
    shots = [load_image(p) for p in args.images]
    res = scan(matcher, shots, detect_layout, min_cards=args.min_cards)
    result = collect(res.sightings)
    result["screenshots"] = [
        {"path": p, "detected": res.detected[i], "rejected": res.rejected.get(i)}
        for i, p in enumerate(args.images)
    ]
    for i, reason in res.rejected.items():
        print(f"{args.images[i]}: rejected ({reason})", file=sys.stderr)
    if args.debug_dir:
        write_debug(Path(args.debug_dir), shots, args.images, res.sightings)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


def cmd_calibrate(args) -> int:
    from .assets import find_asset
    from .calibrate import fit_card_to_art
    from .evaluate import labelled_queries

    layout = Layout.load(args.layout)
    entries = expand_entries(load_cards(_cards_source(args)))
    pairs = []
    for crop, entry in labelled_queries(Path(args.labels), entries, layout):
        art_path = find_asset(Path(args.asset_dir), entry.key)
        if art_path is None:
            print(f"  no art for {entry.key}, skipped", file=sys.stderr)
            continue
        pairs.append((crop, load_image(art_path, layout.fill)))
    if not pairs:
        print("no usable samples", file=sys.stderr)
        return 1
    fitted, report = fit_card_to_art(layout, pairs, log=lambda m: print(m, file=sys.stderr))
    d = fitted.to_dict()
    d["name"] = Path(args.out).stem
    Path(args.out).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
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
        sp.add_argument("--min-score", type=float, help="override the gallery's fitted reject threshold")
        sp.add_argument("--min-margin", type=float, help="override the gallery's fitted reject threshold")

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
    sp.add_argument("--fit-reject", action="store_true", help="fit accept thresholds and store them in the gallery")
    sp.add_argument("--target-error", type=float, default=0.01, help="max share of wrong/unknown acceptances")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("detect", help="find cards in list screenshots (no gallery needed)")
    sp.add_argument("images", nargs="+")
    sp.add_argument("--layout", default="default")
    sp.add_argument("--out-dir", help="save each card crop here")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("scan", help="screenshots -> owned cards")
    query_args(sp)
    sp.add_argument("images", nargs="+")
    sp.add_argument("--layout", help="layout for card detection (default: the gallery's)")
    sp.add_argument("--debug-dir", help="write annotated screenshots, crops and a pre-filled labels.csv")
    sp.add_argument("--min-cards", type=int, help="reject screenshots with fewer cards (default: the layout's)")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("calibrate", help="fit the layout's card_to_art from labelled crops")
    data_args(sp)
    sp.add_argument("--labels", required=True, help="CSV path,card_id[,state] of card crops")
    sp.add_argument("--layout", default="default")
    sp.add_argument("--out", required=True, help="where to write the fitted layout JSON")
    sp.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)
