"""The image and text encoders of visual search (onnxruntime, ``tokenizers`` and numpy only).

SigLIP 2 expects the photo squeezed (not cropped) to a square of ``size`` pixels with bilinear
resampling and the RGB values scaled to -1..1; the text lower-cased, as Gemma tokens, ended by
``<eos>`` and padded to 64 tokens. Both encoders return L2-normalised vectors, so the dot
product of a text and a photo vector is their cosine similarity.

:class:`ImageEncoder` runs in the ``visual`` pipeline stage on ``visual.providers`` (CUDA, else
CPU). :class:`TextEncoder` runs in the search, on the CPU: a request takes a few ten
milliseconds there, and the GPU stays free for the pipeline.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .models import VisualModel, current_model, default_model_dir, ensure_files, visual_version

log = logging.getLogger(__name__)


def unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """3 x size x size float32: squeezed to a square (bilinear), RGB scaled to -1..1."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = arr * (2.0 / 255.0) - 1.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def _session(path: Path, providers: list[str], threads: int = 0):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=providers)


class ImageEncoder:
    """Lazy-loading image encoder of the configured model (what the ``visual`` stage runs)."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.spec: VisualModel = current_model(settings)
        self.providers: list[str] = []
        self._session = None
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        """The vector space (``visual_embeddings.model``)."""
        return self.spec.key

    @property
    def version(self) -> str:
        return visual_version(self.settings)

    @property
    def size(self) -> int:
        return self.spec.size

    @property
    def using_gpu(self) -> bool:
        return any(p != "CPUExecutionProvider" for p in self.providers)

    def load(self, interrupted=None, report=None) -> None:
        with self._lock:
            if self._session is not None:
                return
            from ..faces.analyzer import available_providers

            # the text model too: the search needs it later, maybe without internet
            paths = ensure_files(
                default_model_dir(self.settings), self.spec, interrupted=interrupted, report=report
            )
            cfg = self.settings.visual
            providers = available_providers(cfg.providers or self.settings.faces.providers)
            self._session = _session(paths["vision"], providers)
            self.providers = self._session.get_providers()
            log.info("%s image encoder loaded with providers %s", self.spec.title, self.providers)

    def unload(self) -> None:
        with self._lock:
            self._session = None

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return preprocess(image, self.spec.size)

    def embed(self, pixels: np.ndarray) -> np.ndarray:
        """N x 3 x size x size (from :meth:`preprocess`) -> N x dim, L2-normalised."""
        self.load()
        assert self._session is not None
        out = self._session.run(["pooler_output"], {"pixel_values": pixels.astype(np.float32)})
        return unit(out[0])


class TextEncoder:
    """Lazy-loading text encoder for search requests, with a cache of recent requests.

    It never downloads: the ``visual`` stage downloads both models, and a search without the
    text model says so instead of waiting for hundreds of megabytes.
    """

    cache_size = 256

    def __init__(self, settings) -> None:
        self.settings = settings
        self.spec: VisualModel = current_model(settings)
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def model(self) -> str:
        return self.spec.key

    @property
    def low(self) -> float:
        return self.spec.low

    @property
    def high(self) -> float:
        return self.spec.high

    def _load(self) -> None:
        if self._session is not None:
            return
        from tokenizers import Tokenizer

        from ..animals.models import ModelUnavailable

        folder = default_model_dir(self.settings)
        paths = {name: folder / f.name for name, f in self.spec.files.items() if name != "vision"}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise ModelUnavailable(
                "the text model of visual search is missing (" + ", ".join(missing) + "); "
                "run the 'visual' stage once to download it"
            )
        tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
        tokenizer.enable_truncation(self.spec.max_tokens)
        tokenizer.enable_padding(
            length=self.spec.max_tokens, pad_id=self.spec.pad_id, pad_token=self.spec.pad_token
        )
        self._tokenizer = tokenizer
        self._session = _session(paths["text"], ["CPUExecutionProvider"], threads=4)
        log.info("%s text encoder loaded", self.spec.title)

    def unload(self) -> None:
        with self._lock:
            self._session = None
            self._tokenizer = None

    def embed(self, texts: list[str]) -> np.ndarray:
        """One L2-normalised vector per text (cached)."""
        keys = [" ".join(t.split()).lower() for t in texts]
        with self._lock:
            todo = [k for k in dict.fromkeys(keys) if k not in self._cache]
            if todo:
                self._load()
                assert self._tokenizer is not None and self._session is not None
                ids = np.array([e.ids for e in self._tokenizer.encode_batch(todo)], np.int64)
                out = unit(self._session.run(["pooler_output"], {"input_ids": ids})[0])
                for key, vec in zip(todo, out, strict=True):
                    self._cache[key] = vec
            for key in keys:
                self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return np.stack([self._cache[k] for k in keys])
