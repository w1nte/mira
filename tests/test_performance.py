"""Performance work must not change results: caches, batching, prefetching, query rewrites."""

from __future__ import annotations

import random
import sqlite3
import threading
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import func, select

from vision_tagger.db.cache import VersionedCache
from vision_tagger.db.models import (
    Cluster,
    Face,
    Image,
    ImageEmbedding,
    ImageProperty,
    Person,
    StageStatus,
    Status,
)
from vision_tagger.db.session import Database
from vision_tagger.faces import generations as G
from vision_tagger.faces.ages import analyse_person
from vision_tagger.pipeline import control
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.base import Stage, StageAbort
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.query import ImageFilter, property_counts, property_keys
from vision_tagger.search.document import build_document, doc_hash
from vision_tagger.vlm.backend import FakeBackend, VlmError
from vision_tagger.vlm.schema import Vocabulary

from .helpers import add_face, add_image, noisy, person_centers


# --------------------------------------------------------------------------- versioned cache
def test_versioned_cache_recomputes_after_any_commit(db):
    cache = VersionedCache(db)
    calls = []

    def count(s):
        calls.append(1)
        return s.scalar(select(func.count()).select_from(Image))

    assert cache.get("n", count) == 0
    assert cache.get("n", count) == 0 and len(calls) == 1  # served from the cache
    with db.write() as s:
        add_image(s, 1)
    assert cache.get("n", count) == 1 and len(calls) == 2
    # a commit from another connection (e.g. the pipeline in another process) counts too
    con = sqlite3.connect(db.engine.url.database)
    con.execute("DELETE FROM images")
    con.commit()
    con.close()
    assert cache.get("n", count) == 0 and len(calls) == 3


def test_versioned_cache_never_caches_in_memory_databases():
    db = Database("sqlite://")
    db.create_all()
    assert db.data_version() is None
    calls = []
    cache = VersionedCache(db)
    for _ in range(2):
        cache.get("k", lambda _s: calls.append(1))
    assert len(calls) == 2
    db.dispose()


# --------------------------------------------------------------------------- query helpers
def test_property_keys_and_counts_match_plain_queries(db):
    with db.write() as s:
        for n in range(6):
            img = add_image(s, n, missing=n == 5)
            for k, v in [("objects", f"o{n % 3}"), ("year", "2020"), (f"k{n}", "x")]:
                s.add(ImageProperty(image_id=img.id, key=k, value=v))

    def naive(s, missing_ok: bool):
        stmt = (
            select(ImageProperty.key, ImageProperty.value, func.count(ImageProperty.image_id))
            .join(Image, Image.id == ImageProperty.image_id)
            .group_by(ImageProperty.key, ImageProperty.value)
            .order_by(
                ImageProperty.key, func.count(ImageProperty.image_id).desc(), ImageProperty.value
            )
        )
        if not missing_ok:
            stmt = stmt.where(Image.missing.is_(False))
        out: dict[str, list[tuple[str, int]]] = {}
        for k, v, c in s.execute(stmt):
            out.setdefault(k, []).append((v, c))
        return out

    with db.session() as s:
        assert property_keys(s) == sorted(s.scalars(select(ImageProperty.key).distinct()))
        assert property_counts(s) == naive(s, missing_ok=False)  # one image is missing
        assert property_counts(s, ImageFilter(include_missing=True)) == naive(s, missing_ok=True)
    with db.write() as s:
        s.get(Image, 6).missing = False
    with db.session() as s:
        assert property_counts(s) == naive(s, missing_ok=True)  # no missing image left


# --------------------------------------------------------------------------- runner
@pytest.fixture
def images(db):
    with db.write() as s:
        for n in range(10):
            add_image(s, n)
    return db


def statuses(db, stage="faces"):
    with db.session() as s:
        return {
            r.image_id: r.status
            for r in s.scalars(select(StageStatus).where(StageStatus.stage == stage))
        }


class PrefetchStage(Stage):
    name = "faces"

    def __init__(self, db, settings, *, fail_prepare=(), abort_at=None):
        super().__init__(db, settings)
        self.prefetch = 2
        self.fail_prepare = set(fail_prepare)
        self.abort_at = abort_at
        self.prepared: list[int] = []
        self.processed: list[tuple[int, str]] = []
        self.lock = threading.Lock()

    def prepare(self, image_id):
        with self.lock:
            self.prepared.append(image_id)
        if image_id == self.abort_at:
            raise StageAbort("server gone")
        if image_id in self.fail_prepare:
            raise OSError(f"cannot read {image_id}")
        return f"decoded {image_id}"

    def process(self, image_id):
        self.process_prepared(image_id, self.prepare(image_id))

    def process_prepared(self, image_id, prepared):
        self.processed.append((image_id, prepared))


def test_prefetch_prepares_ahead_and_keeps_results_and_errors(images, settings):
    stage = PrefetchStage(images, settings, fail_prepare={4})
    summary = PipelineRunner(images, settings, lambda _n: stage).run(["faces"], batch_size=3)
    assert summary.processed["faces"] == 10 and summary.failed["faces"] == 1
    assert stage.processed == [(i, f"decoded {i}") for i in range(1, 11) if i != 4]
    assert sorted(stage.prepared) == list(range(1, 11))  # each image decoded exactly once
    st = statuses(images)
    assert st[4] == Status.ERROR and sum(v == Status.DONE for v in st.values()) == 9
    with images.session() as s:
        assert "cannot read 4" in s.get(StageStatus, (4, "faces")).last_error


def test_prefetch_respects_limit_and_abort(images, settings):
    stage = PrefetchStage(images, settings)
    PipelineRunner(images, settings, lambda _n: stage).run(["faces"], batch_size=3, limit=4)
    assert [i for i, _ in stage.processed] == [1, 2, 3, 4]
    assert max(stage.prepared) <= 4  # nothing beyond the limit is decoded

    stage = PrefetchStage(images, settings, abort_at=7)
    summary = PipelineRunner(images, settings, lambda _n: stage).run(["faces"], batch_size=3)
    assert summary.status == "failed" and "server gone" in summary.messages[0]
    st = statuses(images)
    assert st[7] == Status.PENDING and Status.RUNNING not in st.values()


class BatchStage(Stage):
    name = "faces"
    batch = 3

    def __init__(self, db, settings, *, fail=(), stop_after=None):
        super().__init__(db, settings)
        self.fail = set(fail)
        self.stop_after = stop_after
        self.calls: list[list[int]] = []

    def process(self, image_id):  # pragma: no cover - the runner uses process_batch
        raise AssertionError("not used")

    def process_batch(self, image_ids):
        self.calls.append(list(image_ids))
        if self.stop_after and len(self.calls) == self.stop_after:
            control.request_state(self.db, control.STOPPED)
        return {i: RuntimeError(f"bad {i}") for i in image_ids if i in self.fail}


def test_batch_stage_marks_each_image_and_stops_between_batches(images, settings):
    stage = BatchStage(images, settings, fail={5})
    summary = PipelineRunner(images, settings, lambda _n: stage).run(["faces"], batch_size=4)
    assert stage.calls == [[1, 2, 3], [4], [5, 6, 7], [8], [9, 10]]
    assert summary.processed["faces"] == 10 and summary.failed["faces"] == 1
    st = statuses(images)
    assert st[5] == Status.ERROR and sum(v == Status.DONE for v in st.values()) == 9

    control.requeue(images, "faces")
    stage = BatchStage(images, settings, stop_after=1)
    summary = PipelineRunner(images, settings, lambda _n: stage).run(["faces"], batch_size=6)
    assert summary.status == "stopped" and stage.calls == [[1, 2, 3]]
    st = statuses(images)
    assert {i for i, v in st.items() if v == Status.DONE} == {1, 2, 3}
    assert Status.RUNNING not in st.values()


def test_claim_takes_pending_before_retryable_errors_in_image_order(images, settings):
    with images.write() as s:
        for image_id, status in [(1, Status.ERROR), (2, Status.DONE), (3, Status.ERROR)]:
            s.get(StageStatus, (image_id, "faces")).status = status
    runner = PipelineRunner(images, settings, lambda _n: None)
    assert runner._upcoming("faces", 99, 20) == [4, 5, 6, 7, 8, 9, 10, 1, 3]
    assert runner._upcoming("faces", 99, 3) == [4, 5, 6]
    with images.write() as s:  # failed in this run: not retried by the same run
        s.get(StageStatus, (1, "faces")).run_id = 99
    assert runner._upcoming("faces", 99, 20)[-1] == 3


# --------------------------------------------------------------------------- embed stage
def _library(db, n=7):
    with db.write() as s:
        for i in range(n):
            img = add_image(s, i, folder=f"trip/{i}")
            img.stages.append(StageStatus(stage="embed", status=Status.PENDING))
            for k, v in [("objects", "boat"), ("location_type", f"place{i % 3}"), ("tags", "z")]:
                s.add(ImageProperty(image_id=img.id, key=k, value=v, source="vlm"))
            s.add(ImageProperty(image_id=img.id, key="objects", value="anchor", source="vlm"))


class CountingBackend(FakeBackend):
    def __init__(self, vocab, fail_word=None):
        super().__init__(vocab)
        self.embed_calls: list[int] = []
        self.fail_word = fail_word

    def embed(self, texts, model):
        self.embed_calls.append(len(texts))
        if self.fail_word and any(self.fail_word in t for t in texts):
            raise VlmError("model choked")
        return super().embed(texts, model)


def test_embed_stage_batches_requests_with_identical_documents(db, settings):
    _library(db)
    settings.search.embed_batch = 3
    backend = CountingBackend(Vocabulary.load())
    stage = EmbedStage(db, settings, backend=backend)
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["embed"], batch_size=5)
    assert summary.processed["embed"] == 7 and summary.failed["embed"] == 0
    assert backend.embed_calls == [3, 2, 2]  # batches of the runner's claims (5 + 2)
    prefix = settings.search.document_prefix
    with db.session() as s:
        for img in s.scalars(select(Image)):
            props = list(s.scalars(select(ImageProperty).where(ImageProperty.image_id == img.id)))
            document = build_document(img, props, None)
            row = s.get(ImageEmbedding, img.id)
            assert row.doc_hash == doc_hash(document)
            expected = FakeBackend(Vocabulary.load()).embed([prefix + document], stage.model)[0]
            assert np.array_equal(row.vector, expected.astype(np.float16).tobytes())
    # unchanged documents are skipped without a request
    backend.embed_calls.clear()
    assert stage.process_batch([1, 2, 3]) == {}
    assert backend.embed_calls == []


def test_embed_stage_isolates_a_failing_document(db, settings):
    _library(db, n=4)
    with db.write() as s:
        s.add(ImageProperty(image_id=2, key="tags", value="poison", source="vlm"))
    backend = CountingBackend(Vocabulary.load(), fail_word="poison")
    stage = EmbedStage(db, settings, backend=backend)
    errors = stage.process_batch([1, 2, 3, 4, 99])
    assert set(errors) == {2, 99}
    assert isinstance(errors[99], LookupError) and isinstance(errors[2], VlmError)
    with db.session() as s:
        assert sorted(s.scalars(select(ImageEmbedding.image_id))) == [1, 3, 4]
    with pytest.raises(LookupError):
        stage.process(99)


# --------------------------------------------------------------------------- generations
def _old_growup_frames(s, person_id, max_frames=G.MAX_FRAMES):
    """The straightforward version: quality of every face, best per age (reference)."""
    person = s.get(Person, person_id)
    ages = analyse_person(s, person)
    by_face = {f.face_id: f for f in ages.faces}
    best: dict[int, G.Frame] = {}
    for face_id, det, px, landmarks, clusterable in s.execute(
        select(Face.id, Face.det_score, Face.face_px, Face.landmarks, Face.clusterable)
        .where(Face.id.in_(list(by_face)))
        .order_by(Face.id)
    ):
        info = by_face[face_id]
        if not clusterable:
            continue
        if info.calendar_age is not None and not info.suspicious:
            age, estimated = info.calendar_age, False
        elif info.face_age is not None:
            age, estimated = info.face_age, True
        else:
            continue
        frame = G.Frame(
            face_id,
            info.image_id,
            max(0, age),
            estimated,
            info.recorded_at,
            G._quality(det, px, landmarks) * (0.7 if estimated else 1.0),
            person.name,
        )
        current = best.get(frame.age)
        if current is None or frame.quality > current.quality:
            best[frame.age] = frame
    frames = [best[a] for a in sorted(best)]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]
    return frames


def test_growup_frames_pick_exactly_the_best_face_per_age(db):
    rng = np.random.default_rng(3)
    pyrng = random.Random(3)
    center = person_centers(1)[0]
    with db.write() as s:
        cluster = Cluster(locked=True, person=Person(name="Kim", birth_year=1990))
        s.add(cluster)
        s.flush()
        for n in range(400):
            img = add_image(s, n, recorded_at=datetime(1990 + n % 45, 1 + n % 12, 1))
            yaw = pyrng.uniform(-0.5, 0.5)
            landmarks = (
                None
                if n % 17 == 0
                else [[30, 40], [70, 40], [50 + 40 * yaw, 60], [35, 80], [65, 80]]
            )
            add_face(
                s,
                img.id,
                noisy(center, rng),
                cluster_id=cluster.id,
                age=n % 45 + pyrng.choice([-9, 0, 0, 1]),
                # many exact ties in det_score / size
                det_score=pyrng.choice([0.7, 0.8, 0.9, 0.9]),
                face_px=pyrng.choice([60, 120, 200, 200]),
                landmarks=landmarks,
                clusterable=n % 13 != 0,
            )
        pid = cluster.person_id
    with db.session() as s:
        expected = _old_growup_frames(s, pid)
        assert expected
        assert G.growup_frames(s, pid) == expected
        assert G.growup_frames(s, pid, max_frames=100) == _old_growup_frames(s, pid, 100)


# --------------------------------------------------------------------------- gallery / search
def test_date_neighbours_with_filters_nulls_and_ties(db):
    from vision_tagger.query import all_image_ids, neighbours

    with db.write() as s:
        for n in range(14):
            when = None if n % 5 == 0 else datetime(2020, 1 + n % 3, 1)
            img = add_image(s, n, recorded_at=when)
            if n % 2:
                s.add(ImageProperty(image_id=img.id, key="season", value="summer"))
    for sort in ("date_desc", "date_asc"):
        for f in (ImageFilter(sort=sort), ImageFilter(sort=sort, props=[("season", "summer")])):
            with db.session() as s:
                order = all_image_ids(s, f)
                for pos, image_id in enumerate(order):
                    assert neighbours(s, f, image_id) == (
                        order[pos - 1] if pos else None,
                        order[pos + 1] if pos + 1 < len(order) else None,
                    ), (sort, f.props, pos)
                assert neighbours(s, f, 999) == (None, None)


def test_keyword_value_matching_equals_sql_ilike(db, settings):
    from vision_tagger.db.util import like_escape
    from vision_tagger.search.engine import SearchEngine

    values = ["Café", "CAFÉ", "café au lait", "Beach_Day", "100% sure", "a_b", "Ärger", "ärger"]
    with db.write() as s:
        for n, value in enumerate(values):
            img = add_image(s, n)
            s.add(ImageProperty(image_id=img.id, key="tags", value=value))
    engine = SearchEngine(db, settings, backend=FakeBackend(Vocabulary.load()))
    with db.session() as s:
        for word in ["caf", "CAFÉ", "é", "beach_", "%", "_", "ä", "Ä", "rger", "x"]:
            like = f"%{like_escape(word)}%"
            expected = set(
                s.scalars(
                    select(ImageProperty.image_id).where(
                        ImageProperty.value.ilike(like, escape="\\")
                    )
                )
            )
            assert engine._images_with_value(s, word) == expected, word


def test_connections_use_read_pragmas(db):
    from vision_tagger.db.session import CACHE_KIB

    with db.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA cache_size").scalar() == -CACHE_KIB
        assert conn.exec_driver_sql("PRAGMA temp_store").scalar() == 2  # memory
        assert conn.exec_driver_sql("PRAGMA mmap_size").scalar() > 0
