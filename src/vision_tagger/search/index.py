"""In-memory matrix of image embeddings for fast cosine similarity."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import datetime

import numpy as np
from sqlalchemy import func, select

from ..db.models import ImageEmbedding
from ..db.session import Database


class EmbeddingMatrix:
    """Loads ``image_embeddings`` once and then only the rows added since the last load.

    About 150 MB for 100k images with 768-d vectors (float16).
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.Lock()
        self._model: str | None = None
        self._ids: list[int] = []
        self._pos: dict[int, int] = {}
        self._mat = np.zeros((0, 0), dtype=np.float16)
        self._count = 0
        self._newest: datetime | None = None

    def _full_load(self, model: str) -> None:
        with self.db.session() as s:
            rows = s.execute(
                select(ImageEmbedding.image_id, ImageEmbedding.vector, ImageEmbedding.created_at)
                .where(ImageEmbedding.model == model)
                .order_by(ImageEmbedding.image_id)
            ).all()
        self._model = model
        self._ids = [r[0] for r in rows]
        self._pos = {image_id: i for i, image_id in enumerate(self._ids)}
        self._mat = (
            np.vstack([np.frombuffer(r[1], dtype=np.float16) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float16)
        )
        self._count = len(rows)
        self._newest = max((r[2] for r in rows), default=None)

    def ensure(self, model: str) -> None:
        with self._lock:
            with self.db.session() as s:
                count, newest = s.execute(
                    select(func.count(), func.max(ImageEmbedding.created_at)).where(
                        ImageEmbedding.model == model
                    )
                ).one()
            if model != self._model or count < self._count or self._mat.size == 0:
                self._full_load(model)
                return
            if newest == self._newest and count == self._count:
                return
            # incremental: rows created/updated since the last load
            with self.db.session() as s:
                rows = s.execute(
                    select(
                        ImageEmbedding.image_id, ImageEmbedding.vector, ImageEmbedding.created_at
                    ).where(ImageEmbedding.model == model, ImageEmbedding.created_at > self._newest)
                ).all()
            new_vectors = []
            for image_id, blob, created in rows:
                vec = np.frombuffer(blob, dtype=np.float16)
                if vec.shape[0] != self._mat.shape[1]:
                    self._full_load(model)  # dimension changed (different model build)
                    return
                if image_id in self._pos:
                    self._mat[self._pos[image_id]] = vec
                else:
                    self._pos[image_id] = len(self._ids)
                    self._ids.append(image_id)
                    new_vectors.append(vec)
                if self._newest is None or created > self._newest:
                    self._newest = created
            if new_vectors:
                self._mat = np.vstack([self._mat, *new_vectors])
            self._count = len(self._ids)

    @property
    def count(self) -> int:
        return self._count

    def similarities(self, query: np.ndarray, image_ids: Iterable[int]) -> dict[int, float]:
        """Cosine similarity of ``query`` (normalised) to the given images that have vectors."""
        ids = [i for i in image_ids if i in self._pos]
        if not ids or self._mat.size == 0:
            return {}
        rows = np.fromiter((self._pos[i] for i in ids), dtype=np.int64, count=len(ids))
        q = np.asarray(query, dtype=np.float32)
        if q.shape[0] != self._mat.shape[1]:
            return {}
        sims = self._mat[rows].astype(np.float32) @ q
        return dict(zip(ids, sims.tolist(), strict=True))
