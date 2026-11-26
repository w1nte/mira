"""Stage 0: embed every photo with the image encoder of visual search (SigLIP 2).

It runs first: it is the fastest stage, needs nothing from the others, and once it is done the
search finds photos by what they show, long before the vision model has described them.

Worker threads decode the photos of this batch and the next one (the gallery thumbnail when it
is large enough, else the original at a reduced size) while the model embeds the current batch;
the vectors of a batch are written in one short transaction afterwards.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, ClassVar

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import select

from ...animals.models import ModelUnavailable
from ...db.models import Image, VisualEmbedding, utcnow
from ...db.util import chunked
from ...imaging import load_image, thumbnail_is_fresh, thumbnail_path
from ...visual.encoder import ImageEncoder
from ...visual.models import DownloadInterrupted
from .base import Stage, StageInterrupted, StageUnavailable, decode_workers

log = logging.getLogger(__name__)


class VisualStage(Stage):
    name: ClassVar[str] = "visual"
    #: vectors of another model (or preprocessing) can't be compared with the current ones
    redo_outdated: ClassVar[bool] = True

    def __init__(self, db, settings, encoder: Any = None) -> None:
        super().__init__(db, settings)
        self.encoder = encoder or ImageEncoder(settings)
        self.prefetch = decode_workers()
        self.batch = max(1, settings.visual.batch_size)

    @property
    def version(self) -> str:
        return self.encoder.version

    def setup(self) -> None:
        try:
            self.encoder.load(interrupted=self.interrupted, report=self.report)
        except DownloadInterrupted as exc:
            raise StageInterrupted(str(exc)) from exc
        except ModelUnavailable as exc:  # offline: the other stages still run
            raise StageUnavailable(str(exc)) from exc

    def teardown(self) -> None:
        self.encoder.unload()
        gc.collect()

    def info(self) -> dict[str, object]:
        return {"providers": self.encoder.providers, "gpu": self.encoder.using_gpu}

    # ------------------------------------------------------------------ decoding (threads)
    def _open(self, image_id: int, path: str) -> PILImage.Image:
        """The gallery thumbnail if it is fresh and at least the model's input size on both
        sides (a few milliseconds), else the original, decoded at a reduced size."""
        size = self.encoder.size
        thumb = thumbnail_path(self.settings.thumbs_dir, image_id, self.settings.web.thumb_size)
        if thumbnail_is_fresh(thumb, path):
            try:
                with PILImage.open(thumb) as im:
                    if min(im.size) >= size:
                        im.load()
                        return im.convert("RGB")
            except OSError:
                pass  # damaged or locked: decode the original
        return load_image(path, draft_px=2 * size)

    def prepare(self, image_id: int) -> np.ndarray:
        with self.db.session() as s:
            path = s.scalar(select(Image.path).where(Image.id == image_id))
        if path is None:
            raise LookupError(f"image {image_id} not found")
        return self.encoder.preprocess(self._open(image_id, path))

    # ------------------------------------------------------------------ embedding
    def process(self, image_id: int) -> None:
        errors = self.process_batch([image_id])
        if image_id in errors:
            raise errors[image_id]

    def process_prepared(self, image_id: int, prepared: Any) -> None:  # visual.batch_size = 1
        errors = self.process_prepared_batch([image_id], {image_id: prepared})
        if image_id in errors:
            raise errors[image_id]

    def process_batch(self, image_ids: list[int]) -> dict[int, Exception]:
        prepared: dict[int, Any] = {}
        for image_id in image_ids:
            try:
                prepared[image_id] = self.prepare(image_id)
            except Exception as exc:
                prepared[image_id] = exc
        return self.process_prepared_batch(image_ids, prepared)

    def process_prepared_batch(
        self, image_ids: list[int], prepared: dict[int, Any]
    ) -> dict[int, Exception]:
        errors: dict[int, Exception] = {}
        good: list[int] = []
        for image_id in image_ids:
            value = prepared.get(image_id)
            if isinstance(value, Exception):
                errors[image_id] = value
            elif value is None:
                errors[image_id] = LookupError(f"image {image_id} was not prepared")
            else:
                good.append(image_id)
        if not good:
            return errors
        try:  # model inference: no transaction open
            vectors = self.encoder.embed(np.stack([prepared[i] for i in good]))
        except Exception as exc:
            log.warning("visual: embedding %d photos failed: %s", len(good), exc)
            errors.update(dict.fromkeys(good, exc))
            return errors
        model, version = self.encoder.model, self.encoder.version
        with self.db.write() as s:
            rows = {
                r.image_id: r
                for part in chunked(good)
                for r in s.scalars(
                    select(VisualEmbedding).where(VisualEmbedding.image_id.in_(part))
                )
            }
            now = utcnow()
            for image_id, vec in zip(good, vectors, strict=True):
                row = rows.get(image_id)
                if row is None:
                    row = VisualEmbedding(image_id=image_id)
                    s.add(row)
                row.model = model
                row.stage_version = version
                row.dim = int(vec.shape[0])
                row.vector = np.asarray(vec, dtype=np.float16).tobytes()
                row.created_at = now
        return errors
