"""Performance work must not change results: caches, batching, prefetching, query rewrites."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from sqlalchemy import func, select

from vision_tagger.db.cache import VersionedCache
from vision_tagger.db.models import (
    Image,
    ImageProperty,
)
from vision_tagger.db.session import Database
from vision_tagger.query import ImageFilter, property_counts, property_keys
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary

from .helpers import add_image


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
