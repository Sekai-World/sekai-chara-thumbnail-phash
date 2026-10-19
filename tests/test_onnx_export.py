"""export-onnx on a tiny randomly initialised ViT (needs torch, timm, onnx, onnxscript, onnxruntime)."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

from sekai_card_id.embedders import OnnxEmbedder, TimmEmbedder
from sekai_card_id.onnx_export import export_onnx

MODEL = "vit_tiny_patch16_224"
SIZE = 64


def images(n=4):
    rng = np.random.default_rng(1)
    return [Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)) for _ in range(n)]


def reference():
    torch.manual_seed(0)
    return TimmEmbedder(MODEL, img_size=SIZE, device="cpu", pretrained=False).embed(images())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    reports = {}
    for name, quantize in (("fp32", False), ("int8", True)):
        torch.manual_seed(0)
        reports[name] = export_onnx(
            d / f"{name}.onnx", model=MODEL, img_size=SIZE, quantize=quantize, pretrained=False, log=lambda *_: None
        )
    return d, reports


def test_fp32_export_reproduces_torch(exported):
    d, reports = exported
    got = OnnxEmbedder(d / "fp32.onnx").embed(images())
    assert float((got * reference()).sum(axis=1).min()) > 0.9999
    assert reports["fp32"]["fp32_min_cos_vs_torch"] > 0.9999


def test_int8_export_is_smaller_and_close(exported):
    d, reports = exported
    assert (d / "int8.onnx").stat().st_size < 0.6 * (d / "fp32.onnx").stat().st_size
    got = OnnxEmbedder(d / "int8.onnx").embed(images())
    assert float((got * reference()).sum(axis=1).min()) > 0.9
    assert reports["int8"]["quant"].startswith("int8")


def test_exported_file_describes_itself(exported):
    d, _ = exported
    spec = OnnxEmbedder(d / "int8.onnx").spec
    assert spec["model"] == MODEL and spec["img_size"] == SIZE and spec["pool"] == "cls+avg"
    assert spec["quant"].startswith("int8")
