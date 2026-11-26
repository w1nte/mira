"""The NSFW check: a fast local image classifier scores every photo (see ``nsfw``).

It runs first: the classifier needs a fraction of a second per photo, so a new library has its
covers long before the vision model got through it.
"""

from __future__ import annotations

import gc
import logging
from typing import ClassVar

from PIL.Image import Image as PILImage

from ...db.models import Image
from ...imaging import load_image
from ...nsfw.classifier import INPUT_SIZE, NsfwClassifier
from ...nsfw.models import ModelUnavailable
from ...nsfw.verdict import record_score
from .base import Stage, StageAbort, decode_workers
from .versions import nsfw_stage_version

log = logging.getLogger(__name__)

#: photos are decoded at about twice the model's input size (JPEGs at 1/2 .. 1/8 of their size)
DECODE_PX = 2 * INPUT_SIZE


class NsfwStage(Stage):
    name: ClassVar[str] = "nsfw"

    def __init__(self, db, settings, classifier: NsfwClassifier | None = None) -> None:
        super().__init__(db, settings)
        self.classifier = classifier or NsfwClassifier(settings)
        self.prefetch = decode_workers()

    @property
    def version(self) -> str:
        return nsfw_stage_version(self.settings)

    def setup(self) -> None:
        try:
            self.classifier.load()
        except ModelUnavailable as exc:  # offline: stop the run, the photos stay pending
            raise StageAbort(str(exc)) from exc

    def teardown(self) -> None:
        self.classifier.unload()
        gc.collect()

    def info(self) -> dict[str, object]:
        return {"providers": self.classifier.providers, "gpu": self.classifier.using_gpu}

    def prepare(self, image_id: int) -> PILImage:
        """Decode the photo (in a worker thread while the previous one is scored)."""
        with self.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise LookupError(f"image {image_id} not found")
            path = img.path
        return load_image(path, max_px=DECODE_PX, draft_px=DECODE_PX)

    def process(self, image_id: int) -> None:
        self.process_prepared(image_id, self.prepare(image_id))

    def process_prepared(self, image_id: int, prepared: PILImage) -> None:
        score = self.classifier.score(prepared)  # model inference: no transaction open
        with self.db.write() as s:
            record_score(s, image_id, score, self.settings.nsfw.threshold)
