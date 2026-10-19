"""OnnxEmbedder plumbing, tested with a hand-built model (needs onnx + onnxruntime, not torch)."""

import json

import numpy as np
import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper

from sekai_card_id.embedders import ONNX_FORMAT_VERSION, ONNX_METADATA_KEY, OnnxEmbedder, make_embedder
from sekai_card_id.gallery import build_gallery
from sekai_card_id.masterdata import expand_entries
from sekai_card_id.preprocess import Layout
from sekai_card_id.search import Matcher

MEAN = (0.5, 0.4, 0.3)
STD = (0.25, 0.2, 0.1)


def write_model(path, size=(32, 32), mean=MEAN, std=STD, quant="fp32", with_metadata=True, scale=1.0):
    """image (N,3,H,W) -> channel means (N,3) * scale: a stand-in backbone."""
    nodes = [helper.make_node("ReduceMean", ["image"], ["pooled"], axes=[2, 3], keepdims=0)]
    nodes.append(helper.make_node("Mul", ["pooled", "scale"], ["embedding"]))
    graph = helper.make_graph(
        nodes,
        "mini",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, size[1], size[0]])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 3])],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT, [3], [scale, scale, scale])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    if with_metadata:
        info = {
            "format": ONNX_FORMAT_VERSION,
            "model": "mini",
            "img_size": size[0],
            "input_size": list(size),
            "mean": list(mean),
            "std": list(std),
            "pool": "avg",
            "quant": quant,
            "dim": 3,
        }
        entry = model.metadata_props.add()
        entry.key, entry.value = ONNX_METADATA_KEY, json.dumps(info)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def solid(color, size=40):
    return Image.new("RGB", (size, size), color)


def test_embeds_like_the_documented_preprocessing(tmp_path):
    emb = OnnxEmbedder(write_model(tmp_path / "m.onnx"))
    colors = [(255, 0, 0), (10, 200, 90), (128, 128, 128)]
    got = emb.embed([solid(c) for c in colors])
    want = (np.array(colors, np.float32) / 255 - MEAN) / STD
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    np.testing.assert_allclose(got, want, atol=2e-3)  # bicubic resize of a flat image is flat up to rounding
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1, atol=1e-5)


def test_batching_does_not_change_results(tmp_path):
    path = write_model(tmp_path / "m.onnx")
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)) for _ in range(5)]
    np.testing.assert_allclose(
        OnnxEmbedder(path, batch_size=2).embed(images), OnnxEmbedder(path, batch_size=16).embed(images), atol=1e-6
    )


def test_spec_identifies_the_model_by_content_not_name(tmp_path):
    a = write_model(tmp_path / "a.onnx")
    b = tmp_path / "renamed.onnx"
    b.write_bytes(a.read_bytes())
    c = write_model(tmp_path / "c.onnx", scale=2.0)
    ea, eb, ec = OnnxEmbedder(a), OnnxEmbedder(b), OnnxEmbedder(c)
    assert ea.spec["kind"] == "onnx" and ea.spec["quant"] == "fp32"
    assert ea.fingerprint() == eb.fingerprint()
    assert ea.fingerprint() != ec.fingerprint()


def test_make_embedder_from_name_and_stored_spec(tmp_path):
    path = write_model(tmp_path / "m.onnx")
    built = make_embedder("onnx", weights=path, threads=1, device="cuda", img_size=999)  # irrelevant args are ignored
    assert isinstance(built, OnnxEmbedder)
    again = make_embedder(built.spec, weights=path)
    assert again.fingerprint() == built.fingerprint()
    with pytest.raises(ValueError, match="--weights"):
        make_embedder(built.spec)
    with pytest.raises(ValueError, match="--weights"):
        make_embedder("onnx")


def test_rejects_files_not_written_by_export(tmp_path):
    with pytest.raises(ValueError, match="export-onnx"):
        OnnxEmbedder(write_model(tmp_path / "plain.onnx", with_metadata=False))


def test_gallery_is_tied_to_the_model_file(card_assets, tmp_path):
    cards, asset_dir = card_assets
    layout = Layout.load("default")
    model = write_model(tmp_path / "m.onnx", size=(16, 16))
    gallery, _ = build_gallery(expand_entries(cards), asset_dir, OnnxEmbedder(model), layout, log=lambda *_: None)
    assert gallery.meta["embedder"]["kind"] == "onnx"
    Matcher(gallery, OnnxEmbedder(model))  # same file: fine
    other = write_model(tmp_path / "other.onnx", size=(16, 16), scale=3.0)
    with pytest.raises(ValueError, match="does not match"):
        Matcher(gallery, OnnxEmbedder(other))
