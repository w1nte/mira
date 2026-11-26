"""Stage 1: detect faces with InsightFace, then cluster them and match the reference DB."""

from __future__ import annotations

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

from PIL.Image import Image as PILImage
from sqlalchemy import func, select

from ...db.models import Cluster, Face, Image
from ...faces.agenet import FaceAgeEstimator
from ...faces.analyzer import FaceAnalyzer
from ...faces.clustering import recluster_db, replace_faces
from ...faces.references import match_clusters
from ...faces.vectors import to_blob
from ...imaging import ensure_face_crop, load_image, remove_face_crops
from ..control import mark_done
from .base import Stage, decode_workers
from .versions import ages_stage_version, faces_stage_version

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


class FacesStage(Stage):
    name: ClassVar[str] = "faces"

    def __init__(
        self,
        db,
        settings,
        analyzer: FaceAnalyzer | None = None,
        age_estimator: FaceAgeEstimator | None = None,
    ) -> None:
        super().__init__(db, settings)
        self.analyzer = analyzer or FaceAnalyzer(settings.faces)
        self.ages = age_estimator or FaceAgeEstimator(settings.faces)
        # decoding a 12 MP photo takes longer on the CPU than detection on the GPU
        self.prefetch = decode_workers()
        # face crops are written by a background thread during a run (see setup)
        self._crops: ThreadPoolExecutor | None = None
        self._crop_slots = threading.Semaphore(8)  # bounds the decoded images held for it

    @property
    def version(self) -> str:
        return faces_stage_version(self.settings)

    def setup(self) -> None:
        self.analyzer.load()
        self.ages.load()
        self._crops = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-tagger-crops")

    def teardown(self) -> None:
        if self._crops is not None:
            self._crops.shutdown(wait=True)  # every crop of this run is on disk afterwards
            self._crops = None
        self.analyzer.unload()
        self.ages.unload()
        gc.collect()

    def _save_crops(self, im: PILImage, created: list[tuple[int, Box]]) -> None:
        for face_id, bbox in created:
            ensure_face_crop(
                self.settings.face_crops_dir, face_id, im, bbox, size=self.settings.faces.crop_size
            )

    def _save_crops_later(self, im: PILImage, created: list[tuple[int, Box]]) -> None:
        try:
            self._save_crops(im, created)
        except Exception as exc:  # a cache: the web UI recreates missing crops on demand
            log.warning("cannot write face crops %s: %s", [f for f, _ in created], exc)
        finally:
            self._crop_slots.release()

    def info(self) -> dict[str, object]:
        return {"providers": self.analyzer.providers, "gpu": self.analyzer.using_gpu}

    def prepare(self, image_id: int) -> tuple[PILImage, int | None]:
        """Decode the image (in a worker thread while the GPU works on the previous one)."""
        with self.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise LookupError(f"image {image_id} not found")
            path, full_w = img.path, img.width
        return load_image(path, max_px=self.settings.faces.max_image_px), full_w

    def process(self, image_id: int) -> None:
        self.process_prepared(image_id, self.prepare(image_id))

    def process_prepared(self, image_id: int, prepared: tuple[PILImage, int | None]) -> None:
        cfg = self.settings.faces
        im, full_w = prepared
        scale = (full_w / im.width) if full_w else 1.0
        # small, tightly cropped portraits are retried with a border (detector needs context)
        detected = self.analyzer.detect(im, pad_if_empty=max(im.size) < 640)
        # InsightFace's age is poor for children and teens; combine it with AgeNet
        ages = self.ages.estimate(im, [f.bbox for f in detected], [f.age for f in detected])

        new_faces: list[Face] = []
        boxes: list[tuple[float, float, float, float]] = []
        for f, age in zip(detected, ages, strict=True):
            x1, y1, x2, y2 = f.bbox
            face_px = round(min(x2 - x1, y2 - y1) * scale)
            new_faces.append(
                Face(
                    x1=x1 * scale,
                    y1=y1 * scale,
                    x2=x2 * scale,
                    y2=y2 * scale,
                    det_score=f.det_score,
                    landmarks=[[round(x * scale, 1), round(y * scale, 1)] for x, y in f.landmarks]
                    if f.landmarks
                    else None,
                    embedding=to_blob(f.embedding),
                    age=age,
                    gender=f.gender,
                    face_px=face_px,
                    clusterable=f.det_score >= cfg.min_det_score and face_px >= cfg.min_face_px,
                )
            )
            boxes.append(f.bbox)
        # keeps cluster / manual assignment / ignore state of re-detected faces
        with self.db.write() as s:
            removed = replace_faces(s, image_id, new_faces)
            mark_done(s, "ages", [image_id], ages_stage_version(self.settings))
            created = [(face.id, bbox) for face, bbox in zip(new_faces, boxes, strict=True)]

        remove_face_crops(self.settings.face_crops_dir, removed)
        if self._crops is None:
            self._save_crops(im, created)
        elif created:  # while the next image is on the GPU
            self._crop_slots.acquire()
            self._crops.submit(self._save_crops_later, im, created)

    def after_stage(self, processed: int) -> str | None:
        if processed <= 0:
            return None
        return cluster_and_match(self.db, self.settings, full=None)


def cluster_and_match(db, settings, *, full: bool | None = None) -> str:
    """Run clustering (incremental unless ``full``) followed by reference matching.

    ``full=None`` picks a full run when no clusters exist yet.
    """
    if full is None:
        with db.session() as s:
            full = (s.scalar(select(func.count()).select_from(Cluster)) or 0) == 0
    # DBSCAN runs without holding the write lock (see recluster_db)
    stats = recluster_db(db, settings.faces, full=full)
    with db.write() as s:
        matched = match_clusters(s, settings.faces)
    msg = f"{'full' if full else 'incremental'} clustering: {stats}; references: " + (
        f"{matched['assigned']} assigned, {matched['suggested']} suggested"
    )
    log.info(msg)
    return msg
