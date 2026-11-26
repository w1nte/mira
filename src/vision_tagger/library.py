"""What the home page shows: library statistics, memories and collections.

A *collection* is a named gallery filter (an :class:`~vision_tagger.query.ImageFilter` query
string). Some are suggested from what the pipeline found (beach, birthdays, pets, the people
with the most photos, …) and only appear when there are photos for them; others are saved by
the user from any gallery filter (table ``collections``).

Everything returned here is plain data, so it can be cached (see ``db.cache``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import extract, func, select
from sqlalchemy.orm import Session
from starlette.datastructures import QueryParams

from .db.models import Cluster, Collection, Face, Image, ImageProperty, Person
from .query import ImageFilter, count_images, page_images, person_counts, property_counts

MIN_PHOTOS = 3  # a suggestion needs at least this many photos

# (key, value, title, icon): suggested when the pipeline found the value
SUGGESTIONS: list[tuple[str, str, str, str]] = [
    ("event_type", "vacation", "Vacations", "image"),
    ("location_type", "beach", "At the beach", "image"),
    ("location_type", "mountain", "In the mountains", "image"),
    ("event_type", "birthday", "Birthdays", "calendar"),
    ("event_type", "christmas", "Christmas", "calendar"),
    ("event_type", "wedding", "Weddings", "calendar"),
    ("event_type", "family_gathering", "Family gatherings", "users"),
    ("composition", "group", "Group photos", "users"),
    ("composition", "selfie", "Selfies", "user"),
    ("animals", "dog", "Dogs", "image"),
    ("animals", "cat", "Cats", "image"),
    ("composition", "food", "Food", "image"),
    ("season", "winter", "Winter", "calendar"),
    ("time_of_day", "night", "At night", "image"),
    ("aesthetic", "high", "Best shots", "image"),
    ("work_image", "yes", "Work & documents", "tag"),
]


@dataclass
class CollectionView:
    key: str  # "saved:<id>" or "auto:<slug>"
    name: str
    query: str
    count: int
    covers: list[int] = field(default_factory=list)  # image ids, newest first
    saved_id: int | None = None
    icon: str = "image"


@dataclass
class LibraryStats:
    photos: int
    first_year: int | None
    last_year: int | None
    people: int
    faces: int
    with_faces: int
    unnamed_clusters: int
    size_bytes: int
    places: int  # distinct kinds of places


@dataclass
class Memory:
    year: int
    years_ago: int
    image_ids: list[int]


def stats(s: Session) -> LibraryStats:
    present = Image.missing.is_(False)
    photos, first, last, size = s.execute(
        select(
            func.count(Image.id),
            func.min(Image.recorded_at),
            func.max(Image.recorded_at),
            func.coalesce(func.sum(Image.size_bytes), 0),
        ).where(present)
    ).one()
    faces = s.scalar(select(func.count(Face.id)).where(Face.ignored.is_(False))) or 0
    with_faces = (
        s.scalar(select(func.count(func.distinct(Face.image_id))).where(Face.ignored.is_(False)))
        or 0
    )
    unnamed = s.scalar(select(func.count(Cluster.id)).where(Cluster.person_id.is_(None))) or 0
    places = (
        s.scalar(
            select(func.count(func.distinct(ImageProperty.value))).where(
                ImageProperty.key == "location_type",
                ImageProperty.value.not_in(["other", "unknown", "none"]),
            )
        )
        or 0
    )
    return LibraryStats(
        photos=photos or 0,
        first_year=first.year if first else None,
        last_year=last.year if last else None,
        people=s.scalar(select(func.count(Person.id))) or 0,
        faces=faces,
        with_faces=with_faces,
        unnamed_clusters=unnamed,
        size_bytes=size or 0,
        places=places,
    )


def on_this_day(s: Session, today: date | None = None, per_year: int = 6) -> list[Memory]:
    """Photos taken on today's date in earlier years (newest year first)."""
    today = today or date.today()
    rows = s.execute(
        select(extract("year", Image.recorded_at), Image.id)
        .where(
            Image.missing.is_(False),
            Image.recorded_at.is_not(None),
            extract("month", Image.recorded_at) == today.month,
            extract("day", Image.recorded_at) == today.day,
            extract("year", Image.recorded_at) < today.year,
        )
        .order_by(Image.recorded_at.desc())
        .limit(200)
    )
    by_year: dict[int, list[int]] = {}
    for year, image_id in rows:
        ids = by_year.setdefault(int(year), [])
        if len(ids) < per_year:
            ids.append(image_id)
    return [Memory(y, today.year - y, ids) for y, ids in sorted(by_year.items(), reverse=True)]


def recently_added(s: Session, limit: int = 12) -> list[int]:
    return list(
        s.scalars(
            select(Image.id)
            .where(Image.missing.is_(False))
            .order_by(Image.indexed_at.desc(), Image.id.desc())
            .limit(limit)
        )
    )


def _filter(query: str) -> ImageFilter:
    return ImageFilter.from_params(QueryParams(query))


def _view_for(s: Session, key: str, name: str, query: str, **extra) -> CollectionView:
    f = _filter(query)
    count = count_images(s, f)
    covers = [img.id for img in page_images(s, f, 0, 4)] if count else []
    return CollectionView(key=key, name=name, query=query, count=count, covers=covers, **extra)


def saved(s: Session) -> list[CollectionView]:
    return [
        _view_for(s, f"saved:{c.id}", c.name, c.query, saved_id=c.id)
        for c in s.scalars(select(Collection).order_by(Collection.position, Collection.id))
    ]


def suggested(s: Session, limit: int = 12, people: int = 4) -> list[CollectionView]:
    """Collections worth looking at, from what the pipeline found in *this* library."""
    counts = property_counts(s, None, keys={k for k, *_ in SUGGESTIONS})
    out: list[CollectionView] = []
    for person, n in person_counts(s, ImageFilter())[:people]:
        if n >= MIN_PHOTOS:
            out.append(
                _view_for(s, f"auto:person-{person.id}", person.name, f"p={person.id}", icon="user")
            )
    for key, value, title, icon in SUGGESTIONS:
        n = dict(counts.get(key, [])).get(value, 0)
        if n < MIN_PHOTOS:
            continue
        covers = list(
            s.scalars(
                select(ImageProperty.image_id)
                .join(Image, Image.id == ImageProperty.image_id)
                .where(
                    ImageProperty.key == key,
                    ImageProperty.value == value,
                    Image.missing.is_(False),
                )
                .order_by(ImageProperty.image_id.desc())
                .limit(4)
            )
        )
        query = str(QueryParams({"t": f"{key}:{value}"}))
        out.append(CollectionView(f"auto:{key}-{value}", title, query, n, covers, icon=icon))
    year = date.today().year
    this_year = _view_for(s, "auto:this-year", f"This year ({year})", f"from={year}&to={year}")
    if this_year.count >= MIN_PHOTOS:
        out.insert(0, this_year)
    return out[:limit]


# --------------------------------------------------------------------------- editing
def add(s: Session, name: str, query: str) -> Collection:
    name = name.strip()[:200]
    if not name:
        raise ValueError("A collection needs a name.")
    query = _filter(query).query_string()  # normalised; unknown parameters dropped
    position = (s.scalar(select(func.max(Collection.position))) or 0) + 1
    collection = Collection(name=name, query=query, position=position)
    s.add(collection)
    s.flush()
    return collection


def rename(s: Session, collection_id: int, name: str) -> None:
    c = s.get(Collection, collection_id)
    if c is None:
        raise LookupError("collection not found")
    if not name.strip():
        raise ValueError("A collection needs a name.")
    c.name = name.strip()[:200]


def delete(s: Session, collection_id: int) -> None:
    c = s.get(Collection, collection_id)
    if c is not None:
        s.delete(c)


def move(s: Session, collection_id: int, delta: int) -> None:
    """Move a collection up (-1) or down (+1) in the order."""
    items = list(s.scalars(select(Collection).order_by(Collection.position, Collection.id)))
    index = next((i for i, c in enumerate(items) if c.id == collection_id), None)
    if index is None:
        return
    target = max(0, min(len(items) - 1, index + delta))
    items.insert(target, items.pop(index))
    for position, c in enumerate(items):
        c.position = position
