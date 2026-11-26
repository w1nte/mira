"""Stage 1b: (re-)estimate the age of every detected face.

New faces get their age in the faces stage already (and this stage is marked done for them).
This stage brings faces found earlier, or with another age model, up to date without running
face detection again.
"""

from __future__ import annotations

import gc
from typing import ClassVar

from sqlalchemy import select, update

from ...db.models import Face, Image
from ...faces.agenet import FaceAgeEstimator
from ...imaging import load_image
from .base import Stage
from .versions import ages_stage_version


class AgesStage(Stage):
    name: ClassVar[str] = "ages"

    def __init__(self, db, settings, estimator: FaceAgeEstimator | None = None) -> None:
        super().__init__(db, settings)
        self.estimator = estimator or FaceAgeEstimator(settings.faces)

    @property
    def version(self) -> str:
        return ages_stage_version(self.settings)

    def setup(self) -> None:
        self.estimator.load()

    def teardown(self) -> None:
        self.estimator.unload()
        gc.collect()

    def process(self, image_id: int) -> None:
        with self.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise LookupError(f"image {image_id} not found")
            path, full_w = img.path, img.width
            faces = s.execute(
                select(Face.id, Face.x1, Face.y1, Face.x2, Face.y2).where(Face.image_id == image_id)
            ).all()
        if not faces:
            return
        im = load_image(path, max_px=self.settings.faces.max_image_px)
        scale = im.width / full_w if full_w else 1.0
        boxes = [(x1 * scale, y1 * scale, x2 * scale, y2 * scale) for _, x1, y1, x2, y2 in faces]
        ages = self.estimator.estimate(im, boxes)
        with self.db.write() as s:
            for (face_id, *_), age in zip(faces, ages, strict=True):
                s.execute(update(Face).where(Face.id == face_id).values(age=age))
