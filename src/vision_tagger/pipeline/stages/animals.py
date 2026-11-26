"""Stage 1c: find animals with YOLOX, describe them with DINOv2, then group them into pets."""

from __future__ import annotations

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

from PIL.Image import Image as PILImage
from sqlalchemy import func, select

from ...animals.analyzer import AnimalAnalyzer
from ...animals.grouping import regroup_db, replace_animals, to_blob
from ...db.models import Animal, Image, Pet
from ...imaging import ensure_face_crop, load_image, remove_face_crops
from .base import Stage, decode_workers
from .versions import animals_stage_version

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


class AnimalsStage(Stage):
    name: ClassVar[str] = "animals"

    def __init__(self, db, settings, analyzer: AnimalAnalyzer | None = None) -> None:
        super().__init__(db, settings)
        self.analyzer = analyzer or AnimalAnalyzer(settings)
        self.prefetch = decode_workers()
        self._crops: ThreadPoolExecutor | None = None
        self._crop_slots = threading.Semaphore(8)  # bounds the decoded images held for it

    @property
    def version(self) -> str:
        return animals_stage_version(self.settings)

    def setup(self) -> None:
        self.analyzer.load()
        self._crops = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-tagger-crops")

    def teardown(self) -> None:
        if self._crops is not None:
            self._crops.shutdown(wait=True)
            self._crops = None
        self.analyzer.unload()
        gc.collect()

    def info(self) -> dict[str, object]:
        return {"providers": self.analyzer.providers, "gpu": self.analyzer.using_gpu}

    def _save_crops(self, im: PILImage, created: list[tuple[int, Box]]) -> None:
        cfg = self.settings.animals
        for animal_id, bbox in created:
            ensure_face_crop(
                self.settings.animal_crops_dir, animal_id, im, bbox, size=cfg.crop_size, margin=0.1
            )

    def _save_crops_later(self, im: PILImage, created: list[tuple[int, Box]]) -> None:
        try:
            self._save_crops(im, created)
        except Exception as exc:  # a cache: the web UI recreates missing crops on demand
            log.warning("cannot write animal crops %s: %s", [a for a, _ in created], exc)
        finally:
            self._crop_slots.release()

    def prepare(self, image_id: int) -> tuple[PILImage, int | None]:
        """Decode the image (in a worker thread while the previous one is analysed)."""
        with self.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise LookupError(f"image {image_id} not found")
            path, full_w = img.path, img.width
        return load_image(path, max_px=self.settings.animals.max_image_px), full_w

    def process(self, image_id: int) -> None:
        self.process_prepared(image_id, self.prepare(image_id))

    def process_prepared(self, image_id: int, prepared: tuple[PILImage, int | None]) -> None:
        cfg = self.settings.animals
        im, full_w = prepared
        scale = (full_w / im.width) if full_w else 1.0
        rows: list[Animal] = []
        boxes: list[Box] = []
        for a in self.analyzer.detect(im):  # model inference: no transaction open
            box_px = round(a.size * scale)
            if box_px < cfg.min_box_px:
                continue
            x1, y1, x2, y2 = a.bbox
            rows.append(
                Animal(
                    x1=x1 * scale,
                    y1=y1 * scale,
                    x2=x2 * scale,
                    y2=y2 * scale,
                    species=a.species,
                    score=a.score,
                    embedding=to_blob(a.embedding),
                    box_px=box_px,
                    groupable=a.score >= cfg.group_min_score and box_px >= cfg.group_min_px,
                )
            )
            boxes.append(a.bbox)
        with self.db.write() as s:  # keeps pet / manual assignment / removal of re-detections
            removed = replace_animals(s, image_id, rows)
            created = [(row.id, box) for row, box in zip(rows, boxes, strict=True)]
        remove_face_crops(self.settings.animal_crops_dir, removed)
        if self._crops is None:
            self._save_crops(im, created)
        elif created:
            self._crop_slots.acquire()
            self._crops.submit(self._save_crops_later, im, created)

    def after_stage(self, processed: int) -> str | None:
        if processed <= 0:
            return None
        return group_animals(self.db, self.settings, full=None)


def group_animals(db, settings, *, full: bool | None = None) -> str:
    """Group the animals into pets (incremental unless ``full``; ``None``: full if there are no
    pets yet) and update the photos' properties. Returns a summary for the run log."""
    if full is None:
        with db.session() as s:
            full = (s.scalar(select(func.count()).select_from(Pet)) or 0) == 0
    stats = regroup_db(db, settings.animals, full=full)  # DBSCAN without the write lock
    msg = f"{'full' if full else 'incremental'} grouping: {stats}"
    log.info(msg)
    return msg
