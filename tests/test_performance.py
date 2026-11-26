"""Performance work must not change results: caches, batching, prefetching, query rewrites."""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime

import numpy as np
from sqlalchemy import func, select

from vision_tagger.db.cache import VersionedCache
from vision_tagger.db.models import (
    Cluster,
    Face,
    Image,
    ImageProperty,
    Person,
)
from vision_tagger.db.session import Database
from vision_tagger.faces import generations as G
from vision_tagger.faces.ages import analyse_person
from vision_tagger.query import ImageFilter, property_counts, property_keys
from vision_tagger.vlm.backend import FakeBackend
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
