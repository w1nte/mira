"""Stage 3: build the semantic search index (text embeddings of each image's document)."""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from sqlalchemy import select

from ...db.models import Image, ImageEmbedding, ImageProperty, VlmResult, utcnow
from ...search.document import build_document, doc_hash
from ...vlm.backend import VlmBackend, make_backend
from ...vlm.schema import Vocabulary
from .base import Stage, StageAbort
from .ollama import ensure_model
from .versions import embed_stage_version

log = logging.getLogger(__name__)


class EmbedStage(Stage):
    name: ClassVar[str] = "embed"

    def __init__(self, db, settings, backend: VlmBackend | None = None) -> None:
        super().__init__(db, settings)
        self.backend = backend or make_backend(
            settings.vlm, Vocabulary.load(settings.vlm.vocabulary_file)
        )
        # one embedding request (and one transaction) per batch instead of per image
        self.batch = max(1, settings.search.embed_batch)

    @property
    def model(self) -> str:
        return self.settings.search.embed_model

    @property
    def version(self) -> str:
        return embed_stage_version(self.settings, fake=self.backend.model == "fake")

    def setup(self) -> None:
        # downloads a missing model first (vlm.auto_pull)
        if not ensure_model(self, self.backend, self.model):
            raise StageAbort(
                f"embedding model {self.model!r} is not available - is Ollama running? "
                f"Install it with: ollama pull {self.model} (or set vlm.auto_pull = true)"
            )

    def process(self, image_id: int) -> None:
        errors = self.process_batch([image_id])
        if image_id in errors:
            raise errors[image_id]

    def _documents(self, image_ids: list[int]) -> tuple[dict[int, str], dict[int, Exception]]:
        """Search documents that changed since they were embedded (others are skipped)."""
        docs: dict[int, str] = {}
        errors: dict[int, Exception] = {}
        with self.db.session() as s:
            images = {i.id: i for i in s.scalars(select(Image).where(Image.id.in_(image_ids)))}
            vlm = {
                v.image_id: v
                for v in s.scalars(select(VlmResult).where(VlmResult.image_id.in_(image_ids)))
            }
            props: dict[int, list[ImageProperty]] = {}
            for p in s.scalars(
                select(ImageProperty)
                .where(ImageProperty.image_id.in_(image_ids))
                .order_by(ImageProperty.image_id, ImageProperty.key, ImageProperty.id)
            ):
                props.setdefault(p.image_id, []).append(p)
            embedded = {
                image_id: (model, digest)
                for image_id, model, digest in s.execute(
                    select(
                        ImageEmbedding.image_id, ImageEmbedding.model, ImageEmbedding.doc_hash
                    ).where(ImageEmbedding.image_id.in_(image_ids))
                )
            }
            for image_id in image_ids:
                img = images.get(image_id)
                if img is None:
                    errors[image_id] = LookupError(f"image {image_id} not found")
                    continue
                document = build_document(img, props.get(image_id, []), vlm.get(image_id))
                if embedded.get(image_id) != (self.model, doc_hash(document)):
                    docs[image_id] = document
        return docs, errors

    def _embed(self, docs: dict[int, str]) -> tuple[dict[int, np.ndarray], dict[int, Exception]]:
        prefix = self.settings.search.document_prefix
        try:
            vectors = self.backend.embed([prefix + d for d in docs.values()], self.model)
            return dict(zip(docs, vectors, strict=True)), {}
        except Exception as exc:
            if len(docs) == 1:
                return {}, dict.fromkeys(docs, exc)
        # one bad document must not fail the whole batch: retry one by one
        out: dict[int, np.ndarray] = {}
        errors: dict[int, Exception] = {}
        for image_id, document in docs.items():
            try:
                out[image_id] = self.backend.embed([prefix + document], self.model)[0]
            except Exception as exc:
                errors[image_id] = exc
        return out, errors

    def process_batch(self, image_ids: list[int]) -> dict[int, Exception]:
        docs, errors = self._documents(image_ids)
        if not docs:
            return errors
        vectors, failed = self._embed(docs)
        errors.update(failed)
        if not vectors:
            return errors
        with self.db.write() as s:
            rows = {
                r.image_id: r
                for r in s.scalars(
                    select(ImageEmbedding).where(ImageEmbedding.image_id.in_(list(vectors)))
                )
            }
            for image_id, vec in vectors.items():
                row = rows.get(image_id)
                if row is None:
                    row = ImageEmbedding(image_id=image_id)
                    s.add(row)
                row.model = self.model
                row.dim = int(vec.shape[0])
                row.vector = np.asarray(vec, dtype=np.float16).tobytes()
                row.doc_hash = doc_hash(docs[image_id])
                row.created_at = utcnow()
        return errors
