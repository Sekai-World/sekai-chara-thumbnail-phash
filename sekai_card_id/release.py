"""Package built galleries and their ONNX models as release assets.

Layout of what `pack` writes to the output directory, per variant NAME:

  gallery-NAME.tar.gz   gallery/{meta.json,entries.json,embeddings.npy,templates.npy,reject.json}
  embedder-NAME.onnx    the model the gallery was built with
  manifest.json         sha256 / size of every file, a content hash per variant, build info

The archive is byte-for-byte reproducible, and `content_hash` ignores build
timestamps, so a rebuild that changed nothing is recognised as unchanged and
need not be published again.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

from .embedders import _weights_id
from .gallery import REJECT_FILE, Gallery
from .preprocess import PREPROCESS_VERSION

MANIFEST_SCHEMA = 1
ARCHIVE_DIR = "gallery"
ARCHIVE_FILES = ("meta.json", "entries.json", "embeddings.npy", "templates.npy", REJECT_FILE)
HASHED_FILES = ("entries.json", "embeddings.npy", "templates.npy", REJECT_FILE)


class PackError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(gallery_dir: Path, model: Path) -> str:
    """Identifies the gallery's content and its model; ignores when it was built."""
    meta = json.loads((gallery_dir / "meta.json").read_text(encoding="utf-8"))
    meta.pop("built_at", None)
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
    for name in HASHED_FILES:
        h.update(name.encode())
        h.update(sha256_file(gallery_dir / name).encode())
    h.update(sha256_file(model).encode())
    return h.hexdigest()


def write_archive(gallery_dir: Path, dest: Path) -> None:
    """tar.gz with fixed metadata (mtime, owner, mode, member order): same input, same bytes."""
    with (
        open(dest, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar,
    ):
        top = tarfile.TarInfo(ARCHIVE_DIR)
        top.type, top.mode = tarfile.DIRTYPE, 0o755
        tar.addfile(top)
        for name in sorted(ARCHIVE_FILES):
            data = (gallery_dir / name).read_bytes()
            info = tarfile.TarInfo(f"{ARCHIVE_DIR}/{name}")
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))


def _check_pair(name: str, gallery_dir: Path, model: Path) -> Gallery:
    """Refuse to publish a gallery that is not what its model produces (or has no thresholds)."""
    missing = [f for f in ARCHIVE_FILES if not (gallery_dir / f).is_file()]
    if missing:
        raise PackError(f"{name}: {gallery_dir} is missing {', '.join(missing)} (was `eval --fit-reject` run?)")
    gallery = Gallery.load(gallery_dir)
    spec = gallery.meta["embedder"]
    if spec.get("kind") != "onnx":
        raise PackError(f"{name}: only galleries built with --model onnx can be published (this one: {spec.get('kind')})")
    if _weights_id(model, with_name=False) != spec["weights"]:
        raise PackError(f"{name}: {model} is not the model this gallery was built with")
    if gallery.reject is None:
        raise PackError(f"{name}: the thresholds in {REJECT_FILE} were fitted on a different gallery; rerun `eval --fit-reject`")
    return gallery


def _same_content(previous: dict | None, variants: dict) -> bool:
    if not previous or previous.get("schema") != MANIFEST_SCHEMA:
        return False
    old = previous.get("variants", {})
    return set(old) == set(variants) and all(old[n].get("content_hash") == v["content_hash"] for n, v in variants.items())


def pack(
    variants: dict[str, tuple[Path, Path]],
    out_dir: Path,
    previous: dict | None = None,
    commit: str | None = None,
) -> dict:
    """Write the assets for `variants` ({name: (gallery_dir, model_file)}) into `out_dir`.

    Returns the manifest plus `changed`: False when `previous` (the manifest of
    what is already published) has identical content for every variant.
    """
    if not variants:
        raise PackError("no variants to pack")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    newest = 0
    for name, (gallery_dir, model) in variants.items():
        gallery = _check_pair(name, gallery_dir, model)
        gallery_asset = out_dir / f"gallery-{name}.tar.gz"
        model_asset = out_dir / f"embedder-{name}.onnx"
        write_archive(gallery_dir, gallery_asset)
        shutil.copyfile(model, model_asset)
        newest = max([newest, *(e.release_at for e in gallery.entries)])
        entries[name] = {
            "gallery": {"file": gallery_asset.name, "sha256": sha256_file(gallery_asset), "size": gallery_asset.stat().st_size},
            "model": {"file": model_asset.name, "sha256": sha256_file(model_asset), "size": model_asset.stat().st_size},
            "embedder": gallery.meta["embedder"],
            "fingerprint": gallery.meta["fingerprint"],
            "entries": gallery.meta["count"],
            "content_hash": content_hash(gallery_dir, model),
        }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "preprocess_version": PREPROCESS_VERSION,
        "newest_card_release_at": newest,
        "variants": entries,
    }
    # The manifest is uploaded last: a consumer that sees it finds files that match it.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "changed": not _same_content(previous, entries)}
