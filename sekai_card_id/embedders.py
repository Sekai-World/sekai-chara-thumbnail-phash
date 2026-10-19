"""Image -> L2-normalised vector.

Every embedder is described by a JSON-serialisable spec stored in the gallery
metadata, so queries are always embedded exactly like the gallery was.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"

# Bump when the layout of the metadata stored inside exported ONNX files changes.
ONNX_FORMAT_VERSION = 1
ONNX_METADATA_KEY = "sekai_card_id"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def prepare_batch(images: list[Image.Image], size: tuple[int, int], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Images -> normalised float32 (B, 3, H, W). Shared by every neural embedder.

    Plain resize (no centre crop): the whole art region matters.
    """
    arr = np.stack([np.asarray(im.convert("RGB").resize(size, Image.Resampling.BICUBIC), np.float32) / 255 for im in images])
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


def pooled_features(model, x, pool: str):
    """Backbone features -> concatenated, per-part L2-normalised pooled vector (torch).

    Used both for inference and as the body of the exported ONNX graph, so the
    two cannot drift apart.
    """
    import torch

    f = model.forward_features(x)
    if f.ndim == 4:  # convnet: B C H W, no class token
        parts = [f.mean(dim=(2, 3))]
    else:  # vit: B tokens C
        npt = getattr(model, "num_prefix_tokens", 1)
        cls, avg = f[:, 0], f[:, npt:].mean(dim=1)
        parts = {"cls": [cls], "avg": [avg], "cls+avg": [cls, avg]}[pool]
    return torch.cat([torch.nn.functional.normalize(p.float(), dim=-1) for p in parts], dim=-1)


class Embedder:
    spec: dict

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def fingerprint(self) -> str:
        return "|".join(f"{k}={self.spec[k]}" for k in sorted(self.spec) if k != "device")


class TinyEmbedder(Embedder):
    """Handcrafted colour + gradient descriptor. No model download.

    Weak; meant for tests and as a floor to compare pretrained models against.
    """

    def __init__(self):
        self.spec = {"kind": "tiny", "version": 1}

    def embed(self, images):
        feats = []
        for img in images:
            rgb = np.asarray(img.convert("RGB").resize((12, 12), Image.Resampling.BOX), np.float32) / 255
            color = (rgb - rgb.mean()).ravel()
            g = np.asarray(img.convert("L").resize((64, 64), Image.Resampling.BILINEAR), np.float32) / 255
            gx = np.zeros_like(g)
            gy = np.zeros_like(g)
            gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
            gy[1:-1, :] = g[2:, :] - g[:-2, :]
            mag = np.hypot(gx, gy)
            ori = ((np.arctan2(gy, gx) % np.pi) / np.pi * 8).astype(int) % 8
            hog = np.zeros((4, 4, 8), np.float32)
            for cy in range(4):
                for cx in range(4):
                    sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
                    hog[cy, cx] = np.bincount(ori[sl].ravel(), mag[sl].ravel(), minlength=8)
            hog = _l2(hog.reshape(16, 8)).ravel()
            feats.append(np.concatenate([_l2(color), 0.7 * _l2(hog)]))
        return _l2(np.stack(feats).astype(np.float32))


class TimmEmbedder(Embedder):
    """Any timm backbone, used frozen (no training).

    Default is DINOv2 ViT-S/14, whose self-supervised features are strong at
    instance-level retrieval. Weights come from the Hugging Face hub (honours
    HF_ENDPOINT, e.g. a mirror) unless a local `weights` file is given.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        weights: str | None = None,
        img_size: int = 224,
        pool: str = "cls+avg",
        device: str | None = None,
        batch_size: int = 32,
        pretrained: bool = True,
    ):
        import timm
        import torch

        if pool not in ("cls", "avg", "cls+avg"):
            raise ValueError(f"unknown pool {pool!r}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.pool = pool
        kwargs = {"num_classes": 0}
        if img_size:
            kwargs["img_size"] = img_size
        if weights:
            # .npz = original JAX/big_vision checkpoints (timm's custom loader);
            # anything else goes through the standard loader + checkpoint filter.
            kwargs["pretrained_cfg_overlay"] = {"file": str(weights), "custom_load": str(weights).endswith(".npz")}
        try:
            self.model = timm.create_model(model, pretrained=pretrained, **kwargs)
        except TypeError:  # convnets do not accept img_size
            kwargs.pop("img_size", None)
            self.model = timm.create_model(model, pretrained=pretrained, **kwargs)
        self.model.eval().to(self.device)
        cfg = timm.data.resolve_model_data_config(self.model)
        _, h, w = cfg["input_size"]
        # The data config reflects the pretrained resolution (518 for DINOv2),
        # not our override; convnets accept any size, so the override wins.
        self.input_size = (img_size, img_size) if img_size else (w, h)
        self.mean = np.asarray(cfg["mean"], np.float32)
        self.std = np.asarray(cfg["std"], np.float32)
        self.spec = {
            "kind": "timm",
            "model": model,
            "weights": _weights_id(weights),
            "img_size": img_size,
            "pool": pool,
        }

    def _tensor(self, images):
        return self.torch.from_numpy(prepare_batch(images, self.input_size, self.mean, self.std)).to(self.device)

    def embed(self, images):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                feats = pooled_features(self.model, self._tensor(images[i : i + self.batch_size]), self.pool)
                out.append(feats.cpu().numpy())
        return _l2(np.concatenate(out).astype(np.float32))


class OnnxEmbedder(Embedder):
    """A model written by `export-onnx`, run with ONNX Runtime (no torch / timm needed).

    The file carries everything needed to feed it (input size, mean/std) and
    the pooling is part of the graph, so the embedding matches what
    TimmEmbedder would produce from the same backbone. Quantisation, if any,
    is part of the model: gallery and queries must use the very same file,
    which the gallery fingerprint (a hash of the file) enforces.

    `batch_size` trades memory for speed: 8 was the fastest on a 4-core CPU
    and 4 uses a third less memory for ~5% more time; 16 was slower and larger.
    """

    def __init__(self, path: str | Path, threads: int | None = None, batch_size: int = 8):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        raw = self.session.get_modelmeta().custom_metadata_map.get(ONNX_METADATA_KEY)
        if raw is None:
            raise ValueError(f"{path} was not written by `sekai-card-id export-onnx`")
        info = json.loads(raw)
        if info.get("format") != ONNX_FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported ONNX metadata format {info.get('format')!r}")
        self.batch_size = batch_size
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = tuple(info["input_size"])
        self.mean = np.asarray(info["mean"], np.float32)
        self.std = np.asarray(info["std"], np.float32)
        self.spec = {
            "kind": "onnx",
            "model": info["model"],
            "weights": _weights_id(path, with_name=False),
            "img_size": info["img_size"],
            "pool": info["pool"],
            "quant": info["quant"],
        }

    def embed(self, images):
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = prepare_batch(images[i : i + self.batch_size], self.input_size, self.mean, self.std)
            out.append(self.session.run(None, {self.input_name: batch})[0])
        return _l2(np.concatenate(out).astype(np.float32))


def _weights_id(weights: str | Path | None, with_name: bool = True) -> str | None:
    """Identify a local weights file by content (and, optionally, name) so caches survive moves."""
    if not weights:
        return None
    h = hashlib.sha1()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()[:12]
    return f"{Path(weights).name}@{digest}" if with_name else digest


def _only(kwargs: dict, *names: str) -> dict:
    return {k: kwargs[k] for k in names if k in kwargs}


def make_embedder(spec_or_model: dict | str, weights: str | Path | None = None, **kwargs) -> Embedder:
    """Build an embedder from a model name ("tiny", "onnx" or a timm name) or a stored spec.

    `weights` is a local weights file for timm models and the exported model
    file for "onnx". Keyword arguments that do not apply to the kind of
    embedder (e.g. `device` for ONNX, `threads` for timm) are ignored.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if isinstance(spec_or_model, dict):
        spec = dict(spec_or_model)
        kind = spec.pop("kind")
        stored_weights = spec.pop("weights", None)
        if kind == "tiny":
            return TinyEmbedder()
        if kind == "onnx":
            if not weights:
                raise ValueError("this gallery was built with an ONNX model; pass that .onnx file with --weights")
            return OnnxEmbedder(weights, **_only(kwargs, "threads", "batch_size"))
        if stored_weights and not weights:  # a stored id, not a path
            raise ValueError("this gallery was built from a local weights file; pass the same file with --weights")
        spec.update(_only(kwargs, "device", "batch_size"))
        return TimmEmbedder(weights=weights, **_only(spec, "model", "img_size", "pool", "device", "batch_size"))
    if spec_or_model == "tiny":
        return TinyEmbedder()
    if spec_or_model == "onnx":
        if not weights:
            raise ValueError("--model onnx needs the exported model: --weights model.onnx (see `export-onnx`)")
        return OnnxEmbedder(weights, **_only(kwargs, "threads", "batch_size"))
    return TimmEmbedder(spec_or_model, weights=weights, **_only(kwargs, "img_size", "pool", "device", "batch_size"))
