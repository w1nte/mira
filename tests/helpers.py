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
        x1=10,
        y1=10,
        x2=110,
        y2=110,
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
