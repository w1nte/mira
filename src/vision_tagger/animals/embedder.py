"""Describe an animal as a vector with DINOv2-small, so photos of the same animal match.

DINOv2 is a self-supervised vision transformer: its features tell apart fur colour, markings,
face and body shape well enough to recognise one dog or cat among others. Each animal is cut out
with a small margin and padded to a square (never stretched: a stretched dog looks like another
dog), scaled to 224 x 224 and normalised with the ImageNet mean and deviation.

The vector is the class token (384 numbers), averaged with the one of the mirrored crop (the side
an animal faces matters less) and L2-normalised. On real photos (9 dogs and cats with 5-6 photos
each, plus 60 photos of 10 breeds) the class token alone separated individual animals far better
than the mean of the patch tokens or both joined: photos of the same animal had a cosine
similarity of 0.80 on average (10th percentile 0.70), other animals of the same species 0.14
(99th percentile 0.49).
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#: padding colour: the ImageNet mean, which becomes 0 after normalisation
PAD_RGB = (124, 116, 104)
EMBEDDING_DIM = 384

Box = tuple[float, float, float, float]


def crop_box(
    image_size: tuple[int, int], bbox: Box, margin: float = 0.1
) -> tuple[int, int, int, int]:
    """The box grown by ``margin`` of its width / height on every side, inside the image."""
    w, h = image_size
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, round(x1 - mx)), max(0, round(y1 - my))
    right, bottom = min(w, round(x2 + mx)), min(h, round(y2 + my))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def square_crop(
    image: Image.Image, bbox: Box, *, margin: float = 0.1, size: int = INPUT_SIZE
) -> Image.Image:
    """The animal with a margin, padded (not stretched) to a square of ``size`` pixels."""
    crop = image.crop(crop_box(image.size, bbox, margin))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    side = max(crop.size)
    square = Image.new("RGB", (side, side), PAD_RGB)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square.resize((size, size), Image.Resampling.BICUBIC)


def to_tensor(crops: list[Image.Image], flip: bool = False) -> np.ndarray:
    """N x 3 x H x W float32, ImageNet-normalised RGB."""
    arr = np.stack([np.asarray(c, dtype=np.float32) for c in crops]) / 255.0
    if flip:
        arr = arr[:, :, ::-1, :]
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2), dtype=np.float32)


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


def pool(hidden: np.ndarray, mode: str = "cls") -> np.ndarray:
    """Vectors from the last hidden state (N x (1 + patches) x 384), L2-normalised.

    ``cls``: the class token (used); ``mean``: the mean patch token; ``cls+mean``: both joined.
    """
    cls_token = _unit(hidden[:, 0])
    if mode == "cls":
        return cls_token
    mean = _unit(hidden[:, 1:].mean(axis=1))
    if mode == "mean":
        return mean
    return _unit(np.concatenate([cls_token, mean], axis=1))


class DinoEmbedder:
    """A loaded DINOv2 ONNX model. ``embed`` is thread-safe."""

    def __init__(
        self,
        model_path,
        providers: list[str],
        *,
        margin: float = 0.1,
        batch: int = 16,
        tta: bool = True,
        mode: str = "cls",
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.margin = margin
        self.batch = max(1, batch)
        self.tta = tta
        self.mode = mode
        self._lock = threading.Lock()

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.session.run(None, {self.input_name: tensor})[0]

    def embed_crops(self, crops: list[Image.Image]) -> np.ndarray:
        out = []
        for start in range(0, len(crops), self.batch):
            part = crops[start : start + self.batch]
            vec = pool(self._run(to_tensor(part)), self.mode)
            if self.tta:
                vec = _unit(vec + pool(self._run(to_tensor(part, flip=True)), self.mode))
            out.append(vec)
        if not out:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def embed(self, image: Image.Image, boxes: list[Box]) -> np.ndarray:
        """One L2-normalised vector per box (boxes in ``image`` pixels)."""
        return self.embed_crops([square_crop(image, b, margin=self.margin) for b in boxes])
