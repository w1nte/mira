"""In-memory matrices of image embeddings for fast cosine similarity.

One class serves both kinds of vectors: the text embeddings of the search documents
(``image_embeddings``) and the visual embeddings of the photos themselves
(``visual_embeddings``). Only the vectors of one model are loaded, as float16: about 77 MB for
50,000 photos with 768-d vectors.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import func, select

from ..db.models import ImageEmbedding
from ..db.session import Database

SIMILARITY_CHUNK = 16384


class EmbeddingMatrix:
    """Loads one model's rows of ``table`` once and then only the rows added since.

    ``table`` is a model with ``image_id``, ``model``, ``vector`` (float16 bytes, normalised)
    and ``created_at``. Nothing is read while the database is unchanged
    (``PRAGMA data_version``).
    """

    def __init__(self, db: Database, table: Any = ImageEmbedding) -> None:
        self.db = db
        self.table = table
        self._lock = threading.Lock()
        self._model: str | None = None
        self._ids: list[int] = []
        self._pos: dict[int, int] = {}
        self._mat = np.zeros((0, 0), dtype=np.float16)
        self._count = 0
        self._newest: datetime | None = None
        self._data_version: int | None = None

    def _rows(self, model: str, *where: Any) -> list[Any]:
        t = self.table
        with self.db.session() as s:
            return list(
                s.execute(
                    select(t.image_id, t.vector, t.created_at)
                    .where(t.model == model, *where)
                    .order_by(t.image_id)
                ).all()
            )

    def _full_load(self, model: str) -> None:
        rows = self._rows(model)
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
        version = self.db.data_version()
        with self._lock:
            if model == self._model and version is not None and version == self._data_version:
                return  # nothing was committed since the last check
            t = self.table
            with self.db.session() as s:
                count, newest = s.execute(
                    select(func.count(), func.max(t.created_at)).where(t.model == model)
                ).one()
            self._data_version = version
            if model != self._model or count < self._count or self._mat.size == 0:
                self._full_load(model)
                return
            if newest == self._newest and count == self._count:
                return
            # incremental: rows created/updated since the last load
            new_vectors = []
            for image_id, blob, created in self._rows(model, t.created_at > self._newest):
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

    @property
    def nbytes(self) -> int:
        return int(self._mat.nbytes)

    def has(self, image_id: int) -> bool:
        return image_id in self._pos

    def vector(self, image_id: int) -> np.ndarray | None:
        pos = self._pos.get(image_id)
        return None if pos is None else self._mat[pos].astype(np.float32)

    def _scores(self, rows: np.ndarray, query: np.ndarray) -> np.ndarray:
        # in chunks: a float32 copy of all rows would need ~300 MB for 100k images
        sims = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), SIMILARITY_CHUNK):
            part = rows[start : start + SIMILARITY_CHUNK]
            sims[start : start + len(part)] = self._mat[part].astype(np.float32) @ query
        return sims

    def similarities(self, query: np.ndarray, image_ids: Iterable[int]) -> dict[int, float]:
        """Cosine similarity of ``query`` (normalised) to the given images that have vectors."""
        ids = [i for i in image_ids if i in self._pos]
        if not ids or self._mat.size == 0:
            return {}
        q = np.asarray(query, dtype=np.float32)
        if q.shape[0] != self._mat.shape[1]:
            return {}
        rows = np.fromiter((self._pos[i] for i in ids), dtype=np.int64, count=len(ids))
        return dict(zip(ids, self._scores(rows, q).tolist(), strict=True))

    def nearest(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The ``k`` images most similar to ``query``, best first."""
        if self._mat.size == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape[0] != self._mat.shape[1]:
            return []
        sims = self._scores(np.arange(len(self._ids), dtype=np.int64), q)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(self._ids[i], float(sims[i])) for i in top]
