"""Image -> L2-normalised vector.

Every embedder is described by a JSON-serialisable spec stored in the gallery
metadata, so queries are always embedded exactly like the gallery was.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


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
            self.model = timm.create_model(model, pretrained=True, **kwargs)
        except TypeError:  # convnets do not accept img_size
            kwargs.pop("img_size", None)
            self.model = timm.create_model(model, pretrained=True, **kwargs)
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
        # Plain resize (no centre crop): the whole art region matters.
        arr = np.stack(
            [np.asarray(im.convert("RGB").resize(self.input_size, Image.Resampling.BICUBIC), np.float32) / 255 for im in images]
        )
        arr = (arr - self.mean) / self.std
        return self.torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).to(self.device)

    def embed(self, images):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                f = self.model.forward_features(self._tensor(images[i : i + self.batch_size]))
                if f.ndim == 4:  # convnet: B C H W, no class token
                    parts = [f.mean(dim=(2, 3))]
                else:  # vit: B tokens C
                    npt = getattr(self.model, "num_prefix_tokens", 1)
                    cls, avg = f[:, 0], f[:, npt:].mean(dim=1)
                    parts = {"cls": [cls], "avg": [avg], "cls+avg": [cls, avg]}[self.pool]
                parts = [self.torch.nn.functional.normalize(p.float(), dim=-1) for p in parts]
                out.append(self.torch.cat(parts, dim=-1).cpu().numpy())
        return _l2(np.concatenate(out).astype(np.float32))


def _weights_id(weights: str | None) -> str | None:
    """Identify a local weights file by content, not path, so caches survive moves."""
    if not weights:
        return None
    h = hashlib.sha1()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{Path(weights).name}@{h.hexdigest()[:12]}"


def make_embedder(spec_or_model: dict | str, weights: str | None = None, **kwargs) -> Embedder:
    """Build an embedder from a model name ("tiny" or a timm name) or a stored spec."""
    if isinstance(spec_or_model, dict):
        spec = dict(spec_or_model)
        kind = spec.pop("kind")
        if kind == "tiny":
            return TinyEmbedder()
        if spec.pop("weights", None) and not weights:  # a stored id, not a path
            raise ValueError("this gallery was built from a local weights file; pass the same file with --weights")
        spec.update({k: v for k, v in kwargs.items() if v is not None})
        return TimmEmbedder(weights=weights, **spec)
    if spec_or_model == "tiny":
        return TinyEmbedder()
    return TimmEmbedder(spec_or_model, weights=weights, **{k: v for k, v in kwargs.items() if v is not None})
