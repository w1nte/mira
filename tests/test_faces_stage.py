from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy import func, select

from vision_tagger.db.models import Cluster, Face, Image, Person, StageStatus, Status
from vision_tagger.faces.analyzer import DetectedFace
from vision_tagger.faces.references import import_references
from vision_tagger.imaging import face_crop_path
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.faces import FacesStage, cluster_and_match

from .conftest import make_image
from .helpers import person_centers


class FakeAnalyzer:
    """Returns two faces per image with fixed embeddings (two 'people')."""

    def __init__(self):
        self.providers = ["CPUExecutionProvider"]
        self.using_gpu = False
        self.centers = person_centers(2)
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def detect(self, image, pad_if_empty=False):
        w, h = image.size
        return [
            DetectedFace(
                (0.1 * w, 0.1 * h, 0.4 * w, 0.5 * h), 0.95, self.centers[0], age=30, gender="F"
            ),
            DetectedFace(
                (0.6 * w, 0.1 * h, 0.9 * w, 0.5 * h), 0.90, self.centers[1], age=40, gender="M"
            ),
        ]

    def largest_face(self, image):
        return self.detect(image)[0]


def test_faces_stage_with_fake_analyzer(db, settings, photos):
    settings.faces.max_image_px = 200  # force downscale -> bbox must be scaled back
    for i in range(4):
        make_image(photos / f"{i}.jpg", seed=i, size=(800, 600))
    Indexer(db, settings, stages=["faces"]).run()
    analyzer = FakeAnalyzer()
    stage = FacesStage(db, settings, analyzer=analyzer)
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["faces"])
    assert summary.status == "completed"
    assert "full clustering" in summary.messages[0]
    with db.session() as s:
        faces = s.scalars(select(Face)).all()
        assert len(faces) == 8
        f = faces[0]
        assert f.x1 == pytest.approx(80, abs=2) and f.x2 == pytest.approx(320, abs=2)
        assert f.face_px >= 200
        assert all(face_crop_path(settings.face_crops_dir, x.id).exists() for x in faces)
        assert s.scalar(select(func.count()).select_from(Cluster)) == 2
        assert {st.status for st in s.scalars(select(StageStatus))} == {Status.DONE}

    # reprocessing an image replaces its faces instead of duplicating them
    stage.process(1)
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(Face).where(Face.image_id == 1)) == 2
        assert s.scalar(select(func.count()).select_from(Face)) == 8


def test_reference_import_with_fake_analyzer(db, settings, tmp_path):
    settings.paths.references_dir = tmp_path / "refs"
    make_image(tmp_path / "refs" / "Anna" / "a.jpg", seed=1)
    make_image(tmp_path / "refs" / "Anna" / "b.jpg", seed=2)
    (tmp_path / "refs" / "empty").mkdir()
    stats = import_references(db, settings, analyzer=FakeAnalyzer())
    assert stats.persons == 1 and stats.added == 2
    again = import_references(db, settings, analyzer=FakeAnalyzer())
    assert again.added == 0 and again.skipped == 2


# --------------------------------------------------------------------------- real model
@pytest.mark.integration
def test_insightface_end_to_end(db, settings, photos, tmp_path):
    insightface = pytest.importorskip("insightface")
    from insightface.data import get_image

    bgr = get_image("t1")  # group photo bundled with insightface (6 faces)
    rgb = PILImage.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    rgb.save(photos / "group.jpg", quality=95)
    ImageOps.mirror(rgb).save(photos / "group_mirrored.jpg", quality=95)
    w, h = rgb.size
    rgb.crop((0, 0, int(w * 0.92), h)).resize((int(w * 1.1), int(h * 1.2))).save(
        photos / "group_crop.jpg", quality=95
    )
    assert insightface is not None

    settings.faces.providers = ["CPUExecutionProvider"]
    settings.faces.cluster_min_samples = 2
    Indexer(db, settings, stages=["faces"]).run()
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(Image)) == 3

    stage = FacesStage(db, settings)
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["faces"])
    assert summary.status == "completed", summary.messages
    with db.session() as s:
        n_faces = s.scalar(select(func.count()).select_from(Face))
        assert n_faces >= 15  # ~6 faces per image
        n_clusters = s.scalar(select(func.count()).select_from(Cluster))
        assert 4 <= n_clusters <= 8
        biggest = s.scalars(select(Cluster).order_by(Cluster.face_count.desc())).first()
        assert biggest.face_count >= 2
        face = s.scalars(select(Face).where(Face.cluster_id == biggest.id)).first()
        img = s.get(Image, face.image_id)
        box = (face.x1, face.y1, face.x2, face.y2)
        assert 0 <= face.x1 < face.x2 <= img.width

    # reference DB: a crop of that person names the cluster automatically
    settings.paths.references_dir = tmp_path / "refs"
    ref_dir = tmp_path / "refs" / "Reference Person"
    ref_dir.mkdir(parents=True)
    with PILImage.open(img.path) as src:
        pad = 40
        src.crop((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)).save(ref_dir / "r.jpg")
    stats = import_references(db, settings)
    assert stats.added == 1, stats
    cluster_and_match(db, settings)
    with db.session() as s:
        named = s.get(Cluster, biggest.id)
        person = s.scalar(select(Person))
        assert named.person_id == person.id or named.suggested_person_id == person.id
