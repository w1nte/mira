"""Thin wrapper around InsightFace's ``FaceAnalysis`` with provider fallback.

InsightFace downloads a missing model pack itself, without checking what it got. For the
default pack (``buffalo_l``) Mira downloads and verifies it first (:func:`ensure_face_models`,
pinned SHA-256 like every other model), so InsightFace only ever loads checked files.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from ..animals.models import ModelFile, ensure_model
from ..config import FacesConfig
from .vectors import normalize

log = logging.getLogger(__name__)

#: InsightFace's buffalo_l pack of its v0.7 release (the file InsightFace itself downloads)
BUFFALO_L = ModelFile(
    name="buffalo_l.zip",
    url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    size=288_621_354,
    license="InsightFace models: non-commercial research use",
    title="InsightFace buffalo_l",
)
#: model packs Mira verifies: name -> (archive, the files it holds)
PINNED_PACKS = {
    "buffalo_l": (
        BUFFALO_L,
        ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"),
    ),
}


def ensure_face_models(cfg: FacesConfig, *, client: Any = None) -> None:
    """Download, verify and unpack a pinned model pack that is not complete yet.

    Only the expected ``.onnx`` files are taken from the archive, by name (nothing else, no
    folders). Another pack (``faces.model``) is left to InsightFace, unverified.
    """
    pinned = PINNED_PACKS.get(cfg.model)
    if pinned is None:
        log.warning("face model %r is not pinned: InsightFace downloads it unverified", cfg.model)
        return
    spec, files = pinned
    root = Path(cfg.model_root).expanduser() if cfg.model_root else Path.home() / ".insightface"
    folder = root / "models" / cfg.model
    if all((folder / name).is_file() for name in files):
        return
    archive = ensure_model(folder.parent, spec, client=client)  # raises ModelUnavailable
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/")).name
            if name not in files:
                continue
            tmp = folder / f"{name}.{uuid.uuid4().hex}.part"
            try:
                with zf.open(info) as src, tmp.open("wb") as out:
                    while block := src.read(1 << 20):
                        out.write(block)
                tmp.replace(folder / name)
            finally:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
    missing = [name for name in files if not (folder / name).is_file()]
    if missing:
        raise RuntimeError(f"{archive} does not contain {', '.join(missing)}")


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]  # in input image coordinates
    det_score: float
    embedding: np.ndarray  # L2-normalised, 512-d
    landmarks: list[list[float]] | None = None
    age: int | None = None
    gender: str | None = None

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


def available_providers(requested: list[str]) -> list[str]:
    import onnxruntime as ort

    if "CUDAExecutionProvider" in requested and hasattr(ort, "preload_dlls"):
        # onnxruntime-gpu >= 1.21: load CUDA/cuDNN DLLs from the nvidia-* pip packages
        # (needed on Windows when CUDA is not installed system-wide).
        try:
            ort.preload_dlls()
        except Exception as exc:  # pragma: no cover - platform specific
            log.debug("preload_dlls failed: %s", exc)
    available = set(ort.get_available_providers())
    chosen = [p for p in requested if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


class FaceAnalyzer:
    """Lazy-loading InsightFace detector + recognizer (+ age/gender)."""

    def __init__(self, cfg: FacesConfig) -> None:
        self.cfg = cfg
        self._app: Any = None
        self.providers: list[str] = []

    def load(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        ensure_face_models(self.cfg)  # verified files: InsightFace then downloads nothing
        providers = available_providers(self.cfg.providers)
        kwargs: dict[str, Any] = {
            "name": self.cfg.model,
            "providers": providers,
            "allowed_modules": ["detection", "recognition", "genderage"],
        }
        if self.cfg.model_root is not None:
            kwargs["root"] = str(self.cfg.model_root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = FaceAnalysis(**kwargs)
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            app.prepare(ctx_id=ctx_id, det_size=(self.cfg.det_size, self.cfg.det_size))
        self._app = app
        try:
            self.providers = list(app.models["detection"].session.get_providers())
        except Exception:  # pragma: no cover - defensive
            self.providers = providers
        log.info("InsightFace %s loaded with providers %s", self.cfg.model, self.providers)

    def unload(self) -> None:
        self._app = None

    @property
    def using_gpu(self) -> bool:
        return any(p != "CPUExecutionProvider" for p in self.providers)

    def detect(self, image: Image.Image, *, pad_if_empty: bool = False) -> list[DetectedFace]:
        """Detect faces. ``pad_if_empty`` retries tightly cropped portraits with a border."""
        faces = self._detect(image)
        if faces or not pad_if_empty:
            return faces
        pad = max(image.width, image.height) // 2
        padded = Image.new("RGB", (image.width + 2 * pad, image.height + 2 * pad), (127, 127, 127))
        padded.paste(image.convert("RGB"), (pad, pad))
        out = []
        for f in self._detect(padded):
            x1, y1, x2, y2 = f.bbox
            f.bbox = (
                max(0.0, x1 - pad),
                max(0.0, y1 - pad),
                min(float(image.width), x2 - pad),
                min(float(image.height), y2 - pad),
            )
            if f.landmarks:
                f.landmarks = [[x - pad, y - pad] for x, y in f.landmarks]
            if f.bbox[2] > f.bbox[0] and f.bbox[3] > f.bbox[1]:
                out.append(f)
        return out

    def _detect(self, image: Image.Image) -> list[DetectedFace]:
        self.load()
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._app.get(bgr)
        out: list[DetectedFace] = []
        for f in raw:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = normalize(f.embedding)
            gender = None
            if getattr(f, "gender", None) is not None:
                gender = "M" if int(f.gender) == 1 else "F"
            age = getattr(f, "age", None)
            kps = getattr(f, "kps", None)
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            out.append(
                DetectedFace(
                    bbox=(
                        max(0.0, x1),
                        max(0.0, y1),
                        min(float(image.width), x2),
                        min(float(image.height), y2),
                    ),
                    det_score=float(f.det_score),
                    embedding=np.asarray(emb, dtype=np.float32),
                    landmarks=kps.round(1).tolist() if kps is not None else None,
                    age=int(age) if age is not None else None,
                    gender=gender,
                )
            )
        return out

    def largest_face(self, image: Image.Image) -> DetectedFace | None:
        faces = self.detect(image, pad_if_empty=True)
        if not faces:
            return None
        return max(faces, key=lambda f: f.size * f.det_score)
