import gzip
import json
import tarfile

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from sekai_card_id.cli import main
from sekai_card_id.embedders import OnnxEmbedder
from sekai_card_id.evaluate import fit_reject, synthetic_queries
from sekai_card_id.gallery import REJECT_FILE, Gallery, build_gallery
from sekai_card_id.masterdata import expand_entries
from sekai_card_id.preprocess import Layout
from sekai_card_id.release import PackError, content_hash, pack, sha256_file
from sekai_card_id.search import Matcher
from test_onnx import write_model


@pytest.fixture
def published(card_assets, tmp_path):
    """Two galleries (different models) built, fitted and ready to pack."""
    cards, asset_dir = card_assets
    layout = Layout.load("default")
    variants = {}
    for name, scale in (("a", 1.0), ("b", 2.0)):
        model = write_model(tmp_path / f"{name}.onnx", size=(16, 16), scale=scale)
        embedder = OnnxEmbedder(model)
        gallery, _ = build_gallery(expand_entries(cards), asset_dir, embedder, layout, log=lambda *_: None)
        gdir = tmp_path / f"gallery-{name}"
        gallery.save(gdir)
        matcher = Matcher(gallery, embedder)
        gallery.save_reject(gdir, fit_reject(matcher, synthetic_queries(gallery.entries, asset_dir, layout, 30, seed=1)))
        variants[name] = (gdir, model)
    return variants


def test_pack_writes_assets_and_a_manifest_that_matches_them(published, tmp_path):
    out = tmp_path / "dist"
    result = pack(published, out, commit="abc123")
    on_disk = json.loads((out / "manifest.json").read_text())
    assert "changed" not in on_disk and result["changed"] is True
    assert on_disk["commit"] == "abc123" and on_disk["schema"] == 1
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["manifest.json", "gallery-a.tar.gz", "gallery-b.tar.gz", "embedder-a.onnx", "embedder-b.onnx"]
    )
    for name, v in on_disk["variants"].items():
        for kind in ("gallery", "model"):
            f = out / v[kind]["file"]
            assert sha256_file(f) == v[kind]["sha256"] and f.stat().st_size == v[kind]["size"]
        assert v["embedder"]["kind"] == "onnx" and v["entries"] > 0
        assert (out / f"embedder-{name}.onnx").read_bytes() == published[name][1].read_bytes()


def test_archive_is_reproducible_and_loads(published, tmp_path):
    pack(published, tmp_path / "one")
    pack(published, tmp_path / "two")
    assert (tmp_path / "one" / "gallery-a.tar.gz").read_bytes() == (tmp_path / "two" / "gallery-a.tar.gz").read_bytes()
    with gzip.open(tmp_path / "one" / "gallery-a.tar.gz") as gz, tarfile.open(fileobj=gz) as tar:
        assert [m.mtime for m in tar.getmembers()] == [0] * len(tar.getmembers())
        tar.extractall(tmp_path / "x", filter="data")
    loaded = Gallery.load(tmp_path / "x" / "gallery")
    assert loaded.reject is not None and loaded.meta["count"] == len(loaded.entries)
    Matcher(loaded, OnnxEmbedder(published["a"][1]))  # model and gallery fit together


def test_unchanged_content_is_recognised_across_rebuild_timestamps(published, tmp_path):
    first = pack(published, tmp_path / "one")
    previous = json.loads((tmp_path / "one" / "manifest.json").read_text())
    # a rebuild that changes nothing but the build timestamp
    gdir, model = published["a"]
    meta = json.loads((gdir / "meta.json").read_text())
    meta["built_at"] = "2099-01-01T00:00:00Z"
    (gdir / "meta.json").write_text(json.dumps(meta))
    again = pack(published, tmp_path / "two", previous=previous)
    assert first["changed"] is True and again["changed"] is False
    assert content_hash(gdir, model) == previous["variants"]["a"]["content_hash"]


@pytest.mark.parametrize("what", ["gallery", "thresholds", "variants", "no_previous", "other_schema"])
def test_changed_content_is_published(published, tmp_path, what):
    pack(published, tmp_path / "one")
    previous = json.loads((tmp_path / "one" / "manifest.json").read_text())
    current = dict(published)
    gdir = published["a"][0]
    if what == "gallery":  # still a valid gallery, one pixel of one template differs
        templates = np.load(gdir / "templates.npy")
        templates[0, 0, 0, 0] ^= 1
        np.save(gdir / "templates.npy", templates)
    elif what == "thresholds":
        reject = json.loads((gdir / REJECT_FILE).read_text())
        reject["min_score"] = round(reject["min_score"] + 0.01, 4)
        (gdir / REJECT_FILE).write_text(json.dumps(reject))
    elif what == "variants":
        current = {"a": published["a"]}
    elif what == "no_previous":
        previous = None
    elif what == "other_schema":
        previous["schema"] = 99
    assert pack(current, tmp_path / "two", previous=previous)["changed"] is True


def test_refuses_a_model_that_is_not_the_one_the_gallery_was_built_with(published, tmp_path):
    with pytest.raises(PackError, match="not the model"):
        pack({"a": (published["a"][0], published["b"][1])}, tmp_path / "out")


def test_refuses_to_pack_without_thresholds(published, tmp_path):
    gdir, model = published["a"]
    (gdir / REJECT_FILE).unlink()
    with pytest.raises(PackError, match="eval --fit-reject"):
        pack({"a": (gdir, model)}, tmp_path / "out")


def test_refuses_thresholds_fitted_on_another_gallery(published, tmp_path):
    gdir, model = published["a"]
    reject = json.loads((gdir / REJECT_FILE).read_text())
    reject["fingerprint"] = "something-else"
    (gdir / REJECT_FILE).write_text(json.dumps(reject))
    with pytest.raises(PackError, match="different gallery"):
        pack({"a": (gdir, model)}, tmp_path / "out")


def test_cli_pack(published, tmp_path, capsys):
    out = tmp_path / "dist"
    args = ["pack", "--out", str(out), "--commit", "deadbeef"]
    for name, (gdir, model) in published.items():
        args += ["--variant", name, str(gdir), str(model)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] is True and result["commit"] == "deadbeef"
    assert main([*args, "--previous-manifest", str(out / "manifest.json")]) == 0
    assert json.loads(capsys.readouterr().out)["changed"] is False
    assert main([*args, "--previous-manifest", str(tmp_path / "missing.json")]) == 0  # unreadable: treat as changed
    assert json.loads(capsys.readouterr().out)["changed"] is True
