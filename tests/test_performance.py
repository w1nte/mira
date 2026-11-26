"""Performance work must not change results: caches, batching, prefetching, query rewrites."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from vision_tagger.db.models import (
    Image,
    ImageProperty,
)
from vision_tagger.query import ImageFilter, property_counts, property_keys

from .helpers import add_image


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
