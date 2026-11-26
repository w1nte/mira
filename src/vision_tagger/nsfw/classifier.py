"""The NSFW classifier: a ViT through onnxruntime, pre- and post-processing in numpy.

Pre-processing follows the model's ``preprocessor_config.json`` (``ViTImageProcessor``): the
photo is scaled to 224 x 224 (bilinear, the aspect ratio is not kept), values to 0..1, then
normalised with mean 0.5 and standard deviation 0.5, channels first. The output is two logits
("normal", "nsfw"); the score is the softmax probability of "nsfw".
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

from .models import NSFW_CLASS, VIT_NSFW, default_model_dir, ensure_model

log = logging.getLogger(__name__)

INPUT_SIZE = 224
MEAN = 0.5
STD = 0.5


def preprocess(image: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """The model input for one photo: 1 x 3 x size x size, float32."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    resized = rgb.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def nsfw_probability(logits: np.ndarray, index: int = NSFW_CLASS) -> np.ndarray:
    """Softmax probability of the NSFW class for each row of ``logits`` (N x classes)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, index] / e.sum(axis=1)


class VitClassifier:
    """A loaded ONNX classifier. ``score`` is thread-safe."""

    def __init__(self, model_path, providers: list[str]) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.half = "float16" in (inp.type or "")
        size = inp.shape[-1]
        self.input_size = size if isinstance(size, int) else INPUT_SIZE
        self._lock = threading.Lock()

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def score(self, image: Image.Image) -> float:
        x = preprocess(image, self.input_size)
        if self.half:
            x = x.astype(np.float16)
        with self._lock:
            logits = self.session.run(None, {self.input_name: x})[0]
        return float(nsfw_probability(logits)[0])


class NsfwClassifier:
    """Lazy-loading classifier on the face stage's providers (what the ``nsfw`` stage runs)."""

    def __init__(self, settings, *, client=None) -> None:
        self.settings = settings
        self.client = client  # an httpx client for the model download (tests)
        self.providers: list[str] = []
        self._model: VitClassifier | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from ..faces.analyzer import available_providers

            path = ensure_model(default_model_dir(self.settings), VIT_NSFW, client=self.client)
            self._model = VitClassifier(path, available_providers(self.settings.faces.providers))
            self.providers = self._model.providers
            log.info("NSFW classifier loaded with providers %s", self.providers)

    def unload(self) -> None:
        with self._lock:
            self._model = None

    @property
    def using_gpu(self) -> bool:
        return any(p != "CPUExecutionProvider" for p in self.providers)

    def score(self, image: Image.Image) -> float:
        """How likely the photo is not safe for work (0..1)."""
        self.load()
        assert self._model is not None
        return self._model.score(image)
