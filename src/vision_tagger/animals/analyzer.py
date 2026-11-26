"""Both animal models together: what the ``animals`` stage runs for every photo."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import COCO_ANIMALS
from .models import DINOV2_SMALL, YOLOX_S, default_model_dir, ensure_model

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


@dataclass
class DetectedAnimal:
    bbox: Box  # in the pixels of the analysed image
    species: str
    score: float
    embedding: np.ndarray  # L2-normalised

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


class AnimalAnalyzer:
    """Lazy-loading YOLOX detector + DINOv2 embedder on the face stage's providers."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.cfg = settings.animals
        self.providers: list[str] = []
        self._detector = None
        self._embedder = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._detector is not None:
                return
            from ..faces.analyzer import available_providers
            from .detector import YoloxDetector
            from .embedder import DinoEmbedder

            model_dir = default_model_dir(self.settings)
            yolox = ensure_model(model_dir, YOLOX_S)
            dino = ensure_model(model_dir, DINOV2_SMALL)
            providers = available_providers(self.settings.faces.providers)
            self._detector = YoloxDetector(
                yolox,
                providers,
                score_threshold=self.cfg.min_score,
                nms_iou=self.cfg.nms_iou,
                classes=COCO_ANIMALS,
            )
            self._embedder = DinoEmbedder(dino, providers, margin=self.cfg.crop_margin)
            self.providers = self._detector.providers
            log.info("animal models loaded with providers %s", self.providers)

    def unload(self) -> None:
        with self._lock:
            self._detector = None
            self._embedder = None

    @property
    def using_gpu(self) -> bool:
        return any(p != "CPUExecutionProvider" for p in self.providers)

    def detect(self, image: Image.Image) -> list[DetectedAnimal]:
        """The animals in ``image`` (boxes in its pixels), each with its embedding."""
        self.load()
        assert self._detector is not None and self._embedder is not None
        found = self._detector.detect(image)
        if not found:
            return []
        vectors = self._embedder.embed(image, [d.bbox for d in found])
        return [
            DetectedAnimal(d.bbox, d.species, d.score, vec)
            for d, vec in zip(found, vectors, strict=True)
        ]
