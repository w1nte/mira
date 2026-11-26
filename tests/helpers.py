from __future__ import annotations

from datetime import datetime

import numpy as np

from vision_tagger.db.models import Face, Image, StageStatus, Status
from vision_tagger.faces.vectors import normalize, to_blob


def add_image(s, n: int, **kw) -> Image:
    img = Image(
        path=kw.pop("path", f"/photos/img{n}.jpg"),
        path_key=f"/photos/img{n}.jpg",
        root="/photos",
        rel_path=f"img{n}.jpg",
        folder=kw.pop("folder", ""),
        filename=f"img{n}.jpg",
        ext=".jpg",
        size_bytes=1000,
        mtime=0.0,
        width=kw.pop("width", 640),
        height=kw.pop("height", 480),
        recorded_at=kw.pop("recorded_at", datetime(2020, 1, 1 + n % 28)),
        date_source="exif",
        phash=f"{n:016x}",
        meta=kw.pop("meta", {}),
        camera=kw.pop("camera", None),
        **kw,
    )
    img.stages = [StageStatus(stage=st, status=Status.PENDING) for st in ("faces", "vlm")]
    s.add(img)
    s.flush()
    return img


def person_centers(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(size=(n, 512)))


def noisy(center: np.ndarray, rng: np.random.Generator, r: float = 0.6) -> np.ndarray:
    noise = rng.normal(size=512)
    noise = noise / np.linalg.norm(noise) * r
    return normalize(center + noise)


def add_face(s, image_id: int, emb: np.ndarray, **kw) -> Face:
    face = Face(
        image_id=image_id,
        x1=kw.pop("x1", 10),
        y1=kw.pop("y1", 10),
        x2=kw.pop("x2", 110),
        y2=kw.pop("y2", 110),
        det_score=kw.pop("det_score", 0.9),
        embedding=to_blob(emb),
        face_px=kw.pop("face_px", 100),
        **kw,
    )
    s.add(face)
    s.flush()
    return face


def make_face_dataset(s, persons: int = 3, per_person: int = 6, seed: int = 0):
    """Return {person_index: [face_id, ...]} plus the centers."""
    rng = np.random.default_rng(seed + 1)
    centers = person_centers(persons, seed)
    out: dict[int, list[int]] = {}
    n = 0
    for p in range(persons):
        for _ in range(per_person):
            img = add_image(s, n)
            n += 1
            out.setdefault(p, []).append(add_face(s, img.id, noisy(centers[p], rng)).id)
    return out, centers


class FakeAgeEstimator:
    """Stands in for ``FaceAgeEstimator`` (no model download): keeps given ages, else 25."""

    def __init__(self, default: int = 25):
        self.default = default
        self.calls = 0

    def load(self):
        pass

    def unload(self):
        pass

    def estimate(self, image, boxes, insightface_ages=None):
        self.calls += 1
        if insightface_ages is not None:
            return [None if a is None else round(a) for a in insightface_ages]
        return [self.default] * len(boxes)


# --------------------------------------------------------------------------- animals
ANIMAL_DIM = 384


def animal_centers(n: int, seed: int = 7) -> np.ndarray:
    """``n`` well separated unit vectors (one per animal), DINOv2-sized."""
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(size=(n, ANIMAL_DIM)))


def near(center: np.ndarray, rng: np.random.Generator, r: float = 0.4) -> np.ndarray:
    """A photo of the animal ``center``: cosine similarity about 1 / sqrt(1 + r**2)."""
    noise = rng.normal(size=center.shape)
    return normalize(center + noise / np.linalg.norm(noise) * r)


class FakeAnimalAnalyzer:
    """Stands in for ``AnimalAnalyzer`` (no model download, no onnxruntime).

    ``animals(image)`` returns ``[(species, (x1, y1, x2, y2) as fractions, embedding), ...]``;
    by default every photo shows the same dog in its lower left quarter.
    """

    def __init__(self, animals=None, score: float = 0.9):
        self.providers = ["CPUExecutionProvider"]
        self.using_gpu = False
        self.loaded = False
        self.calls = 0
        self.score = score
        center = animal_centers(1)[0]
        self.animals = animals or (lambda image: [("dog", (0.05, 0.45, 0.5, 0.95), center)])

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def detect(self, image):
        from vision_tagger.animals.analyzer import DetectedAnimal

        self.calls += 1
        w, h = image.size
        return [
            DetectedAnimal((x1 * w, y1 * h, x2 * w, y2 * h), species, self.score, emb)
            for species, (x1, y1, x2, y2), emb in self.animals(image)
        ]


def add_animal(s, image_id: int, emb: np.ndarray, **kw):
    from vision_tagger.animals.grouping import to_blob
    from vision_tagger.db.models import Animal

    animal = Animal(
        image_id=image_id,
        x1=kw.pop("x1", 20),
        y1=kw.pop("y1", 200),
        x2=kw.pop("x2", 300),
        y2=kw.pop("y2", 460),
        species=kw.pop("species", "dog"),
        score=kw.pop("score", 0.9),
        embedding=to_blob(emb),
        box_px=kw.pop("box_px", 260),
        **kw,
    )
    s.add(animal)
    s.flush()
    return animal


# --------------------------------------------------------------------------- visual search
VISUAL_DIM = 8
#: colour words the fake text encoder understands (English and German), by RGB channel
COLOUR_WORDS = {
    "red": 0,
    "rot": 0,
    "rote": 0,
    "roten": 0,
    "green": 1,
    "grün": 1,
    "grüne": 1,
    "blue": 2,
    "blau": 2,
    "blaue": 2,
}


class FakeVisualEncoder:
    """Stands in for both visual encoders (no model download, no onnxruntime).

    A photo's vector is its mean colour (dimensions 0-2); a text's vector points to the colour
    words in it ("red" / "rot": the red channel), other words to dimensions no photo has. So
    "red" finds the reddest photos, in English and German, and an unknown word finds nothing.
    ``embed`` takes pixels (the image encoder) or a list of texts (the text encoder).
    """

    model = "fake-visual"
    version = "fake-visual-v1"
    size = 16
    low, high = 0.2, 0.8

    def __init__(self, on_embed=None):
        self.providers = ["CPUExecutionProvider"]
        self.using_gpu = False
        self.loaded = False
        self.embedded = 0
        self.texts: list[str] = []
        self.on_embed = on_embed

    def load(self, interrupted=None, report=None):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def preprocess(self, image):
        arr = np.asarray(image.convert("RGB").resize((self.size, self.size)), dtype=np.float32)
        return np.ascontiguousarray((arr / 255.0).transpose(2, 0, 1))

    def embed(self, data):
        if isinstance(data, list):
            return self._texts(data)
        if self.on_embed is not None:
            self.on_embed(len(data))
        self.embedded += len(data)
        out = np.zeros((len(data), VISUAL_DIM), dtype=np.float32)
        out[:, :3] = data.mean(axis=(2, 3)) + 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def _texts(self, texts):
        import zlib

        out = np.zeros((len(texts), VISUAL_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            self.texts.append(text)
            for word in text.casefold().split():
                if word in COLOUR_WORDS:
                    out[row, COLOUR_WORDS[word]] += 1.0
                else:
                    out[row, 3 + zlib.crc32(word.encode()) % (VISUAL_DIM - 3)] += 0.1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
