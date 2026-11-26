"""Animal detection with YOLOX-s (COCO) through onnxruntime, pre- and post-processing in numpy.

The steps follow the YOLOX ONNX demo (``demo/ONNXRuntime`` in Megvii-BaseDetection/YOLOX):

1. **Letterbox**: the photo is scaled to fit 640 x 640 (aspect ratio kept), placed in the top
   left corner and the rest filled with the value 114. Channels are BGR (the model was trained on
   OpenCV images), values 0..255, no normalisation.
2. **Decode**: the network predicts, for each cell of three grids (strides 8, 16 and 32: 8400
   cells), a box relative to the cell, an objectness and 80 class probabilities. Box centre =
   (offset + cell) * stride, box size = exp(size) * stride.
3. **Score** = objectness x class probability. Each cell gets its best class; cells whose best
   class is an animal and whose score passes the threshold are kept. Taking the best class over
   all 80 classes keeps a teddy bear from showing up as a bear.
4. **NMS** over the animal boxes (any species): overlapping boxes of one animal are merged even
   if the cells disagree about the species. People and other objects never suppress an animal.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import numpy as np
from PIL import Image

from . import COCO_ANIMALS

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
PAD_VALUE = 114

Box = tuple[float, float, float, float]


@dataclass
class Detection:
    bbox: Box  # x1, y1, x2, y2 in the pixels of the image given to ``detect``
    species: str
    score: float


# --------------------------------------------------------------------------- pre-processing
def letterbox(image: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """The YOLOX input (1 x 3 x size x size, BGR, float32) and the scale factor used.

    Like YOLOX's ``preproc``: scaled by ``r = min(size / h, size / w)`` with the new size rounded
    down, top-left aligned, padded with 114.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    w, h = rgb.size
    r = min(size / h, size / w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:nh, :nw] = np.asarray(resized)[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), r


# --------------------------------------------------------------------------- post-processing
@cache
def grids(size: int = INPUT_SIZE, strides: tuple[int, ...] = STRIDES) -> tuple[np.ndarray, ...]:
    """Cell coordinates (N x 2) and the stride of each cell (N x 1), in the network's order."""
    cells, cell_strides = [], []
    for stride in strides:
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        cells.append(grid)
        cell_strides.append(np.full((len(grid), 1), stride))
    return (
        np.concatenate(cells).astype(np.float32),
        np.concatenate(cell_strides).astype(np.float32),
    )


def decode(raw: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Raw head output (N x 85) -> box centre and size in input pixels, objectness, classes."""
    cells, strides = grids(size)
    if raw.shape[0] != len(cells):
        raise ValueError(f"expected {len(cells)} predictions for {size} px, got {raw.shape[0]}")
    out = np.array(raw, dtype=np.float32, copy=True)
    out[:, :2] = (out[:, :2] + cells) * strides
    out[:, 2:4] = np.exp(np.clip(out[:, 2:4], -20.0, 20.0)) * strides
    return out


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Indices of the boxes kept by greedy non-maximum suppression, best first."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        overlap = iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][overlap <= iou_threshold]
    return keep


def postprocess(
    raw: np.ndarray,
    ratio: float,
    image_size: tuple[int, int],
    *,
    score_threshold: float,
    nms_iou: float = 0.45,
    classes: Mapping[int, str] = COCO_ANIMALS,
    input_size: int = INPUT_SIZE,
) -> list[Detection]:
    """Animals in a raw YOLOX output (N x 85), boxes in the original image's pixels."""
    pred = decode(raw, input_size)
    cls_scores = pred[:, 4:5] * pred[:, 5:]
    best = cls_scores.argmax(axis=1)
    score = cls_scores[np.arange(len(best)), best]
    wanted = np.array(sorted(classes), dtype=np.int64)
    mask = np.isin(best, wanted) & (score >= score_threshold)
    if not mask.any():
        return []
    p, best, score = pred[mask], best[mask], score[mask]
    boxes = np.stack(
        [
            p[:, 0] - p[:, 2] / 2,
            p[:, 1] - p[:, 3] / 2,
            p[:, 0] + p[:, 2] / 2,
            p[:, 1] + p[:, 3] / 2,
        ],
        axis=1,
    ) / max(ratio, 1e-9)
    w, h = image_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    out: list[Detection] = []
    for i in nms(boxes, score, nms_iou):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(Detection((x1, y1, x2, y2), classes[int(best[i])], float(score[i])))
    return out


# --------------------------------------------------------------------------- runtime
class YoloxDetector:
    """A loaded YOLOX ONNX model. ``detect`` is thread-safe."""

    def __init__(
        self,
        model_path,
        providers: list[str],
        *,
        score_threshold: float = 0.45,
        nms_iou: float = 0.45,
        classes: Mapping[int, str] = COCO_ANIMALS,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_size = shape[-1] if isinstance(shape[-1], int) else INPUT_SIZE
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.classes = dict(classes)
        self._lock = threading.Lock()

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def detect(self, image: Image.Image) -> list[Detection]:
        tensor, ratio = letterbox(image, self.input_size)
        with self._lock:
            raw = self.session.run(None, {self.input_name: tensor})[0]
        return postprocess(
            raw[0],
            ratio,
            image.size,
            score_threshold=self.score_threshold,
            nms_iou=self.nms_iou,
            classes=self.classes,
            input_size=self.input_size,
        )
