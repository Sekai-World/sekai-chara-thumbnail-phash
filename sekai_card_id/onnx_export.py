"""timm backbone -> self-describing ONNX file for serving without torch.

Needs the `export` extra (torch, timm, onnx, onnxscript, onnxruntime). The
file it writes only needs `onnxruntime` to run (see OnnxEmbedder).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .embedders import DEFAULT_MODEL, ONNX_FORMAT_VERSION, ONNX_METADATA_KEY, TimmEmbedder, _l2, pooled_features

MAX_EXPORT_BATCH = 256
# fp32 ONNX must reproduce torch up to float rounding; anything looser means a broken export.
FP32_MIN_COS = 0.9999


def _quantize(src: Path, dst: Path) -> None:
    """int8 weights for the big matrix multiplications.

    Per-channel scales and MatMul only: per-tensor quantisation of everything
    (including the patch-embedding convolution) cost visibly more accuracy.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )


def _run(path: Path, batch: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _l2(session.run(None, {session.get_inputs()[0].name: batch})[0])


def export_onnx(
    out: str | Path,
    model: str = DEFAULT_MODEL,
    weights: str | None = None,
    img_size: int = 224,
    pool: str = "cls+avg",
    quantize: bool = True,
    pretrained: bool = True,
    log=print,
) -> dict:
    """Export `model` (pooled and normalised like TimmEmbedder) to `out`.

    Returns a report with the file size and how closely the result follows
    the torch model. Raises if the fp32 export does not reproduce torch.
    """
    import onnx
    import torch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emb = TimmEmbedder(model, weights=weights, img_size=img_size, pool=pool, device="cpu", pretrained=pretrained)

    class Pooled(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return pooled_features(self.backbone, x, pool)

    wrapper = Pooled(emb.model).eval()
    width, height = emb.input_size
    dummy = torch.zeros(2, 3, height, width)

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        log(f"exporting {model} ({width}x{height}) ...")
        program = torch.onnx.export(
            wrapper,
            (dummy,),
            dynamo=True,
            input_names=["image"],
            output_names=["embedding"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=MAX_EXPORT_BATCH)}},
        )
        program.save(str(fp32))

        probe = torch.randn(3, 3, height, width)
        with torch.inference_mode():
            reference = _l2(wrapper(probe).numpy())
        got = _run(fp32, probe.numpy())
        cos_fp32 = float((reference * got).sum(axis=1).min())
        if cos_fp32 < FP32_MIN_COS:
            raise RuntimeError(f"fp32 ONNX export does not match torch (min cosine {cos_fp32:.6f})")

        final = Path(tmp) / "final.onnx"
        if quantize:
            log("quantising to int8 ...")
            _quantize(fp32, final)
            cos_quant = float((got * _run(final, probe.numpy())).sum(axis=1).min())
        else:
            shutil.copy(fp32, final)
            cos_quant = None

        info = {
            "format": ONNX_FORMAT_VERSION,
            "model": model,
            "img_size": img_size,
            "input_size": [width, height],
            "mean": [float(v) for v in emb.mean],
            "std": [float(v) for v in emb.std],
            "pool": pool,
            "quant": "int8-dynamic-matmul-perchannel" if quantize else "fp32",
            "dim": int(reference.shape[1]),
        }
        proto = onnx.load(str(final))
        entry = proto.metadata_props.add()
        entry.key, entry.value = ONNX_METADATA_KEY, json.dumps(info, sort_keys=True)
        onnx.save(proto, str(out))

    return {
        "path": str(out),
        "size_mb": round(out.stat().st_size / 1e6, 1),
        "quant": info["quant"],
        "dim": info["dim"],
        "fp32_min_cos_vs_torch": round(cos_fp32, 7),
        "quant_min_cos_vs_fp32_on_noise": None if cos_quant is None else round(cos_quant, 4),
    }
