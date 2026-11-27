from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy import func, select

from vision_tagger.config import Settings
from vision_tagger.db.models import Cluster, Face, Image, Person, StageStatus, Status
from vision_tagger.faces.analyzer import DetectedFace
from vision_tagger.faces.references import import_references
from vision_tagger.imaging import face_crop_path
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.faces import FacesStage, cluster_and_match

from .conftest import make_image
from .helpers import FakeAgeEstimator, person_centers


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
    stage = FacesStage(db, settings, analyzer=analyzer, age_estimator=FakeAgeEstimator())
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


def test_rerun_keeps_names_ignores_and_manual_moves(db, settings, photos):
    """Re-running the faces stage (e.g. 're-queue outdated') must not wipe manual work."""
    from vision_tagger.faces import clustering as C
    from vision_tagger.pipeline import control

    for i in range(4):
        make_image(photos / f"{i}.jpg", seed=i, size=(800, 600))
    Indexer(db, settings, stages=["faces"]).run()
    analyzer = FakeAnalyzer()
    stage = FacesStage(db, settings, analyzer=analyzer, age_estimator=FakeAgeEstimator())
    runner = PipelineRunner(db, settings, lambda _n: stage)
    runner.run(["faces"])
    with db.write() as s:
        a, b = sorted(s.scalars(select(Cluster.id)))
        C.name_cluster(s, a, "Anna", references=2)
        faces_b = list(s.scalars(select(Face).where(Face.cluster_id == b).order_by(Face.id)))
        C.ignore_faces(s, [faces_b[0].id])
        C.move_faces(s, [faces_b[1].id], a)  # deliberate manual move
        moved_image = faces_b[1].image_id
        ignored_image = faces_b[0].image_id

    control.requeue(db, "faces")
    runner.run(["faces"])

    with db.session() as s:
        assert sorted(s.scalars(select(Cluster.id))) == [a, b]  # no cluster deleted
        assert s.get(Cluster, a).person.name == "Anna"
        img_faces = list(s.scalars(select(Face).where(Face.image_id == ignored_image)))
        assert any(f.ignored for f in img_faces)
        moved = [
            f
            for f in s.scalars(select(Face).where(Face.image_id == moved_image))
            if f.cluster_id == a and f.assignment == "manual"
        ]
        assert moved, "manual move lost"
        assert s.scalar(select(func.count()).select_from(Face)) == 8
        # references point at existing faces only
        from vision_tagger.db.models import PersonReference

        ref_faces = [r.face_id for r in s.scalars(select(PersonReference))]
        assert ref_faces and all(s.get(Face, fid) is not None for fid in ref_faces)


def test_reference_folder_with_non_ascii_name(db, settings, tmp_path):
    settings.paths.references_dir = tmp_path / "refs"
    make_image(tmp_path / "refs" / "Анна" / "a.jpg", seed=1)
    make_image(tmp_path / "refs" / "Анна" / "b.jpg", seed=2)
    make_image(tmp_path / "refs" / "Özlem" / "c.jpg", seed=3)
    stats = import_references(db, settings, analyzer=FakeAnalyzer())
    assert stats.added == 3 and not stats.errors
    with db.session() as s:
        assert sorted(p.name for p in s.scalars(select(Person))) == ["Özlem", "Анна"]


def test_faces_stage_sets_ages_and_marks_the_ages_stage_done(db, settings, photos):
    from vision_tagger.db.models import StageStatus, Status
    from vision_tagger.indexer import Indexer, ensure_stage_rows
    from vision_tagger.pipeline.stages import stage_versions

    make_image(photos / "a.jpg", seed=1)
    Indexer(db, settings).run()
    ensure_stage_rows(db, settings.pipeline.stages)
    ages = FakeAgeEstimator()
    stage = FacesStage(db, settings, analyzer=FakeAnalyzer(), age_estimator=ages)
    stage.process(1)
    assert ages.calls == 1
    with db.session() as s:
        assert sorted(f.age for f in s.scalars(select(Face))) == [30, 40]
        row = s.get(StageStatus, (1, "ages"))
        assert row.status == Status.DONE
        assert row.stage_version == stage_versions(settings)["ages"]


def test_ages_stage_updates_existing_faces(db, settings, photos):
    from vision_tagger.pipeline.stages.ages import AgesStage

    make_image(photos / "a.jpg", seed=1)
    from vision_tagger.indexer import Indexer

    Indexer(db, settings).run()
    FacesStage(db, settings, analyzer=FakeAnalyzer(), age_estimator=FakeAgeEstimator()).process(1)
    stage = AgesStage(db, settings, estimator=FakeAgeEstimator(default=7))
    stage.process(1)
    stage.process(1)  # idempotent
    with db.session() as s:
        assert [f.age for f in s.scalars(select(Face))] == [7, 7]


def test_ages_stage_is_added_with_faces():
    from vision_tagger.config import PipelineConfig
    from vision_tagger.pipeline.stages import STAGE_NAMES, stage_versions

    assert PipelineConfig(stages=["faces", "vlm"]).stages == ["faces", "ages", "vlm"]
    assert PipelineConfig(stages=["vlm"]).stages == ["vlm"]
    assert "ages" in STAGE_NAMES and STAGE_NAMES.index("ages") == STAGE_NAMES.index("faces") + 1
    assert "ages" in stage_versions(Settings())


# --------------------------------------------------------------------------- verified models
def _fake_pack(tmp_path, monkeypatch, payload: bytes | None = None):
    """A small stand-in for buffalo_l, served by a fake download; returns (cfg, requests)."""
    import hashlib
    import io
    import zipfile

    import httpx

    from vision_tagger.animals.models import ModelFile
    from vision_tagger.config import FacesConfig
    from vision_tagger.faces import analyzer as A

    files = ("det_10g.onnx", "w600k_r50.onnx")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in files:
            zf.writestr(name, name.encode())
        zf.writestr("../../evil.onnx", b"evil")  # never extracted
        zf.writestr("notes.txt", b"ignored")
    data = buf.getvalue()
    spec = ModelFile(
        name="buffalo_l.zip",
        url="https://example.invalid/buffalo_l.zip",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        license="test",
        title="test pack",
    )
    monkeypatch.setitem(A.PINNED_PACKS, "buffalo_l", (spec, files))
    requests: list[str] = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, content=data if payload is None else payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FacesConfig(model_root=tmp_path / "root"), client, requests


def test_face_models_are_verified_before_insightface_sees_them(tmp_path, monkeypatch):
    from vision_tagger.faces.analyzer import ensure_face_models

    cfg, client, requests = _fake_pack(tmp_path, monkeypatch)
    ensure_face_models(cfg, client=client)
    folder = tmp_path / "root" / "models" / "buffalo_l"
    assert sorted(p.name for p in folder.iterdir()) == ["det_10g.onnx", "w600k_r50.onnx"]
    assert not list(tmp_path.rglob("evil.onnx")) and not list(tmp_path.rglob("notes.txt"))
    ensure_face_models(cfg, client=client)  # complete: nothing downloaded again
    assert len(requests) == 1


def test_tampered_face_models_are_not_used(tmp_path, monkeypatch):
    from vision_tagger.animals.models import ModelUnavailable
    from vision_tagger.faces.analyzer import ensure_face_models

    cfg, client, _requests = _fake_pack(tmp_path, monkeypatch, payload=b"PK not the pack")
    with pytest.raises(ModelUnavailable, match="not the expected file"):
        ensure_face_models(cfg, client=client)
    assert not (tmp_path / "root" / "models" / "buffalo_l").exists()
    assert not list((tmp_path / "root" / "models").glob("*.zip"))


def test_other_face_model_packs_are_left_to_insightface(tmp_path, monkeypatch):
    from vision_tagger.faces.analyzer import ensure_face_models

    cfg, client, requests = _fake_pack(tmp_path, monkeypatch)
    cfg.model = "antelopev2"
    ensure_face_models(cfg, client=client)
    assert requests == []
