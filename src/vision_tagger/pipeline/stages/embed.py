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
from .versions import embed_stage_version

log = logging.getLogger(__name__)


class EmbedStage(Stage):
    name: ClassVar[str] = "embed"

    def __init__(self, db, settings, backend: VlmBackend | None = None) -> None:
        super().__init__(db, settings)
        self.backend = backend or make_backend(
            settings.vlm, Vocabulary.load(settings.vlm.vocabulary_file)
        )

    @property
    def model(self) -> str:
        return self.settings.search.embed_model

    @property
    def version(self) -> str:
        return embed_stage_version(self.settings, fake=self.backend.model == "fake")

    def setup(self) -> None:
        if not self.backend.has_model(self.model):
            raise StageAbort(
                f"embedding model {self.model!r} is not available - is Ollama running? "
                f"Install it with: ollama pull {self.model}"
            )

    def process(self, image_id: int) -> None:
        with self.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise LookupError(f"image {image_id} not found")
            props = list(s.scalars(select(ImageProperty).where(ImageProperty.image_id == image_id)))
            vlm = s.get(VlmResult, image_id)
            document = build_document(img, props, vlm)
            existing = s.get(ImageEmbedding, image_id)
            unchanged = (
                existing is not None
                and existing.model == self.model
                and existing.doc_hash == doc_hash(document)
            )
        if unchanged:
            return
        vec = self.backend.embed([self.settings.search.document_prefix + document], self.model)[0]
        with self.db.write() as s:
            row = s.get(ImageEmbedding, image_id) or ImageEmbedding(image_id=image_id)
            row.model = self.model
            row.dim = int(vec.shape[0])
            row.vector = np.asarray(vec, dtype=np.float16).tobytes()
            row.doc_hash = doc_hash(document)
            row.created_at = utcnow()
            s.merge(row)
