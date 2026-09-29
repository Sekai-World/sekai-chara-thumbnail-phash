"""Download card thumbnail art (thumbnail/chara) into a local cache directory."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .masterdata import CardEntry
from .net import FetchError, fetch_bytes

# Placeholders: {bundle} {state} {ext}. Point this at whichever asset mirror you use.
DEFAULT_ASSET_URL = (
    "https://storage.sekai.best/sekai-jp-assets/thumbnail/chara/{bundle}_{state}.{ext}"
)
DEFAULT_EXTS = ("webp", "png")
IMAGE_EXTS = ("webp", "png", "jpg", "jpeg")


@dataclass
class SyncReport:
    downloaded: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"downloaded={len(self.downloaded)} cached={len(self.cached)} "
            f"missing={len(self.missing)} failed={len(self.failed)}"
        )


def find_asset(asset_dir: Path, key: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = asset_dir / f"{key}.{ext}"
        if p.is_file():
            return p
    return None


class _Breaker:
    """Stop hammering a host that is down: after `limit` failures, skip the rest."""

    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
        self.lock = threading.Lock()

    def tripped(self) -> bool:
        return self.count >= self.limit

    def fail(self) -> None:
        with self.lock:
            self.count += 1


def _sync_one(entry: CardEntry, asset_dir: Path, url_template: str, exts, breaker: _Breaker) -> tuple[str, str | None]:
    if find_asset(asset_dir, entry.key):
        return "cached", None
    if breaker.tripped():
        return "failed", "skipped after too many failures"
    for ext in exts:
        url = url_template.format(bundle=entry.asset_bundle, state=entry.state, ext=ext)
        try:
            data = fetch_bytes(url)
        except FetchError as e:
            breaker.fail()
            return "failed", str(e)
        if data:
            tmp = asset_dir / f".{entry.key}.{ext}.part"
            tmp.write_bytes(data)
            tmp.replace(asset_dir / f"{entry.key}.{ext}")
            return "downloaded", None
    return "missing", None


def sync_assets(
    entries: list[CardEntry],
    asset_dir: Path,
    url_template: str = DEFAULT_ASSET_URL,
    exts=DEFAULT_EXTS,
    workers: int = 8,
    max_failures: int = 30,
) -> SyncReport:
    """Fetch every entry's art that is not cached yet.

    Missing art (e.g. a card announced in master data before its assets go
    live, or a card that only exists trained) is recorded and retried on the
    next run.
    """
    asset_dir.mkdir(parents=True, exist_ok=True)
    report = SyncReport()
    breaker = _Breaker(max_failures)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda e: _sync_one(e, asset_dir, url_template, exts, breaker), entries)
        for entry, (status, err) in zip(entries, results):
            if status == "failed":
                report.failed[entry.key] = err or ""
            else:
                getattr(report, status).append(entry.key)
    (asset_dir / "_sync_report.json").write_text(
        json.dumps(
            {"missing": report.missing, "failed": report.failed},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return report
