"""Image filter builder shared by the gallery, facet counts, timeline and export.

Filters are encoded in the URL query string so every view can be bookmarked:

=========  =====================================================  =========
parameter  meaning                                                example
=========  =====================================================  =========
``p``      person id (repeatable)                                 ``p=3``
``c``      face cluster id (repeatable, for unnamed clusters)     ``c=17``
``pet``    pet id (repeatable; see ``animals``)                   ``pet=5``
``pm``     combine persons/clusters/pets: ``all`` (AND) / ``any`` ``pm=any``
``t``      property ``key:value`` (repeatable: AND); ``|``        ``t=season:summer``
           separates alternatives of one condition (OR)           ``t=city:Paris|city:Rome``
``tm``     older URLs: ``any`` makes all ``t`` one OR condition   ``tm=any``
``xp``     exclude person id                                      ``xp=4``
``xpet``   exclude pet id                                         ``xpet=5``
``xt``     exclude property ``key:value``                         ``xt=work_image:yes``
``from``   recorded on/after date (YYYY-MM-DD, YYYY-MM or YYYY)   ``from=2021``
``to``     recorded on/before date                                ``to=2021-08``
``d``      date range ``FROM..TO`` or a period (repeatable: OR)   ``d=2019&d=2021-06..2021-08``
``folder`` folder prefix (relative to the root)                   ``folder=2021/Italy``
``camera`` camera model                                           ``camera=iPhone 13``
``faces``  ``yes`` / ``no``                                       ``faces=yes``
``stage``  + ``status``: pipeline state                           ``stage=vlm&status=error``
``q``      text search in file name, path and caption             ``q=birthday``
``place``  place id (see ``geo.places``; repeatable: OR)          ``place=12&place=14``
``bbox``   map area: min lon, min lat, max lon, max lat           ``bbox=13.3,52.4,13.5,52.6``
``sort``   ``date_desc`` (default), ``date_asc``, ``name``, ``path``, ``faces``, ``size``
=========  =====================================================  =========

Alternatives ("or"): the alternatives of one condition are OR-ed, conditions are AND-ed and
exclusions always apply. ``t=location_type:beach|location_type:mountain&t=season:summer`` is
"(beach or mountain) and summer". A photo has one date and one place, so repeated ``d`` and
``place`` values are alternatives too. Persons, clusters and pets are combined by ``pm``.
Older URLs and saved collections keep their meaning: a ``t`` without ``|`` is one condition,
and ``tm=any`` (the former "any property" switch) makes all of them one OR condition.
"""

from __future__ import annotations

import calendar
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from sqlalchemy import Select, and_, exists, func, not_, or_, select
from sqlalchemy.orm import Session

from .db.models import (
    Animal,
    Cluster,
    Face,
    Image,
    ImageProperty,
    Person,
    Pet,
    StageStatus,
    VlmResult,
)
from .db.util import chunked, like_escape

SORTS = ("date_desc", "date_asc", "name", "path", "faces", "size")


def _parse_date(value: str | None, end: bool = False) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        if len(value) == 4:
            y = int(value)
            return datetime(y, 12, 31, 23, 59, 59) if end else datetime(y, 1, 1)
        if len(value) == 7:
            y, m = int(value[:4]), int(value[5:7])
            if end:
                return datetime(y, m, calendar.monthrange(y, m)[1], 23, 59, 59)
            return datetime(y, m, 1)
        d = datetime.strptime(value[:10], "%Y-%m-%d")
        return d.replace(hour=23, minute=59, second=59) if end else d
    except ValueError:
        return None


def _ints(values: Iterable[Any]) -> list[int]:
    out = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _pairs(values: Iterable[str]) -> list[tuple[str, str]]:
    out = []
    for v in values:
        if v and ":" in v:
            k, _, val = v.partition(":")
            if k and val:
                out.append((k, val))
    return out


#: ``|`` followed by ``key:`` starts the next alternative of a ``t`` value. Keys are slugs, so
#: a ``|`` inside a value (normalised values never have one) does not split it.
_ALTERNATIVE = re.compile(r"\|(?=[\w-]+:)")


def _groups(values: Iterable[str]) -> list[list[tuple[str, str]]]:
    """``t`` values: each is one condition; ``key:value|key:value`` lists alternatives (OR)."""
    out = []
    for v in values:
        alternatives = _pairs(_ALTERNATIVE.split(v))
        if alternatives:
            out.append(alternatives)
    return out


def _ranges(values: Iterable[str]) -> list[tuple[str | None, str | None]]:
    """``d`` values: ``FROM..TO`` (either end may be empty) or one period (``2019-07``)."""
    out: list[tuple[str | None, str | None]] = []
    for v in values:
        first, sep, last = str(v).partition("..")
        start, end = first.strip() or None, (last if sep else first).strip() or None
        if start is None and end is None:
            continue
        if (start and _parse_date(start) is None) or (end and _parse_date(end, end=True) is None):
            continue
        out.append((start, end))
    return out


def range_param(start: str | None, end: str | None) -> str:
    """A date range as a ``d`` value: ``2019`` (a whole period) or ``2019-06..2019-08``."""
    if start and start == end:
        return start
    return f"{start or ''}..{end or ''}"


def _unique(items: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(items))


def _bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """``"minlon,minlat,maxlon,maxlat"``; min lon > max lon crosses the date line."""
    try:
        w, s, e, n = (float(v) for v in (value or "").split(","))
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (w, s, e, n)):
        return None
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= n <= 90):
        return None
    return w, s, e, n


def _coord(v: float) -> str:
    return f"{v:.5f}".rstrip("0").rstrip(".")


def _getlist(params: Mapping[str, Any], key: str) -> list[str]:
    if hasattr(params, "getlist"):
        return [v for v in params.getlist(key) if v not in (None, "")]
    v = params.get(key)
    if v is None or v == "":
        return []
    return list(v) if isinstance(v, list | tuple) else [v]


@dataclass
class ImageFilter:
    persons: list[int] = field(default_factory=list)
    clusters: list[int] = field(default_factory=list)
    pets: list[int] = field(default_factory=list)
    persons_mode: str = "all"  # persons, clusters and pets together
    props: list[tuple[str, str]] = field(default_factory=list)
    props_mode: str = "all"  # "any" (older URLs, tm=any): normalised into one of prop_groups
    prop_groups: list[list[tuple[str, str]]] = field(default_factory=list)  # each: OR
    exclude_persons: list[int] = field(default_factory=list)
    exclude_pets: list[int] = field(default_factory=list)
    exclude_props: list[tuple[str, str]] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    dates: list[tuple[str | None, str | None]] = field(default_factory=list)  # OR-ed ranges
    folder: str | None = None
    camera: str | None = None
    has_faces: bool | None = None
    stage: str | None = None
    status: str | None = None
    text: str | None = None
    places: list[int] = field(default_factory=list)  # OR: a photo has one place
    bbox: tuple[float, float, float, float] | None = None
    include_missing: bool = False
    sort: str = "date_desc"

    def __post_init__(self) -> None:
        """Normalise the alternatives, so equal filters have equal URLs."""
        groups = [[tuple(a) for a in g] for g in self.prop_groups]
        props = [tuple(p) for p in self.props]
        if self.props_mode == "any":  # the former "any property" switch: one OR condition
            groups, props = [[*props, *(a for g in groups for a in g)]], []
        self.props_mode = "all"
        self.prop_groups = []
        for group in groups:
            alternatives = _unique(group)
            if len(alternatives) == 1:
                if alternatives[0] not in props:
                    props.append(alternatives[0])
            elif alternatives and alternatives not in self.prop_groups:
                self.prop_groups.append(alternatives)
        self.props = props
        dates = _unique((a or None, b or None) for a, b in self.dates if a or b)
        if len(dates) == 1 and not self.date_from and not self.date_to:
            (self.date_from, self.date_to), dates = dates[0], []
        self.dates = dates
        self.places = _unique(self.places)

    # ------------------------------------------------------------------ (de)serialisation
    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ImageFilter:
        faces = params.get("faces")
        sort = params.get("sort") or "date_desc"
        groups = _groups(_getlist(params, "t"))
        return cls(
            persons=_ints(_getlist(params, "p")),
            clusters=_ints(_getlist(params, "c")),
            pets=_ints(_getlist(params, "pet")),
            persons_mode="any" if params.get("pm") == "any" else "all",
            props=[g[0] for g in groups if len(g) == 1],
            props_mode="any" if params.get("tm") == "any" else "all",
            prop_groups=[g for g in groups if len(g) > 1],
            exclude_persons=_ints(_getlist(params, "xp")),
            exclude_pets=_ints(_getlist(params, "xpet")),
            exclude_props=_pairs(_getlist(params, "xt")),
            date_from=params.get("from") or None,
            date_to=params.get("to") or None,
            dates=_ranges(_getlist(params, "d")),
            folder=params.get("folder") or None,
            camera=params.get("camera") or None,
            has_faces=True if faces == "yes" else False if faces == "no" else None,
            stage=params.get("stage") or None,
            status=params.get("status") or None,
            text=(params.get("q") or "").strip() or None,
            places=_ints(_getlist(params, "place")),
            bbox=_bbox(params.get("bbox")),
            include_missing=params.get("missing") == "1",
            sort=sort if sort in SORTS else "date_desc",
        )

    def to_params(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        out += [("p", str(p)) for p in self.persons]
        out += [("c", str(c)) for c in self.clusters]
        out += [("pet", str(p)) for p in self.pets]
        if self.persons_mode == "any" and self.who_count > 1:
            out.append(("pm", "any"))
        out += [("t", f"{k}:{v}") for k, v in self.props]
        out += [("t", "|".join(f"{k}:{v}" for k, v in g)) for g in self.prop_groups]
        out += [("xp", str(p)) for p in self.exclude_persons]
        out += [("xpet", str(p)) for p in self.exclude_pets]
        out += [("xt", f"{k}:{v}") for k, v in self.exclude_props]
        for key, val in (
            ("from", self.date_from),
            ("to", self.date_to),
            ("folder", self.folder),
            ("camera", self.camera),
            ("stage", self.stage),
            ("status", self.status),
            ("q", self.text),
        ):
            if val:
                out.append((key, val))
        out += [("d", range_param(a, b)) for a, b in self.dates]
        if self.has_faces is not None:
            out.append(("faces", "yes" if self.has_faces else "no"))
        out += [("place", str(p)) for p in self.places]
        if self.bbox is not None:
            out.append(("bbox", ",".join(_coord(v) for v in self.bbox)))
        if self.include_missing:
            out.append(("missing", "1"))
        if self.sort != "date_desc":
            out.append(("sort", self.sort))
        return out

    def query_string(self, **changes: Any) -> str:
        f = self.copy(**changes) if changes else self
        return urlencode(f.to_params())

    def copy(self, **changes: Any) -> ImageFilter:
        data = {k: _copy_value(v) for k, v in self.__dict__.items()}
        data.update(changes)
        return ImageFilter(**data)

    @property
    def who_count(self) -> int:
        """Persons, clusters and pets: the filters ``persons_mode`` combines."""
        return len(self.persons) + len(self.clusters) + len(self.pets)

    @property
    def is_empty(self) -> bool:
        return not [p for p in self.to_params() if p[0] != "sort"]

    # convenience for templates ---------------------------------------
    def without_person(self, pid: int) -> str:
        return self.query_string(persons=[p for p in self.persons if p != pid])

    def without_cluster(self, cid: int) -> str:
        return self.query_string(clusters=[c for c in self.clusters if c != cid])

    def without_pet(self, pid: int) -> str:
        return self.query_string(pets=[p for p in self.pets if p != pid])

    def without_exclude_pet(self, pid: int) -> str:
        return self.query_string(exclude_pets=[p for p in self.exclude_pets if p != pid])

    def with_pet(self, pid: int) -> str:
        if pid in self.pets:
            return self.query_string()
        return self.query_string(pets=[*self.pets, pid])

    def has_prop(self, key: str, value: str) -> bool:
        """``key: value`` is required, alone or as an alternative of a condition."""
        pair = (key, value)
        return pair in self.props or any(pair in g for g in self.prop_groups)

    @property
    def prop_keys(self) -> set[str]:
        """Keys with a condition: their facets offer alternatives."""
        return {k for k, _ in self.props} | {k for g in self.prop_groups for k, _ in g}

    @property
    def grouped_keys(self) -> set[str]:
        """Keys that are part of an OR condition already."""
        return {k for g in self.prop_groups for k, _ in g}

    def without_key(self, key: str) -> ImageFilter:
        """The filter without its conditions on ``key``: which alternatives there are."""
        return self.copy(
            props=[p for p in self.props if p[0] != key],
            prop_groups=[g for g in self.prop_groups if all(k != key for k, _ in g)],
        )

    def without_prop(self, key: str, value: str) -> str:
        """Without ``key: value``; an OR condition left with one alternative becomes plain."""
        pair = (key, value)
        return self.query_string(
            props=[p for p in self.props if p != pair],
            prop_groups=[[a for a in g if a != pair] for g in self.prop_groups],
        )

    def without_exclude_prop(self, key: str, value: str) -> str:
        return self.query_string(exclude_props=[p for p in self.exclude_props if p != (key, value)])

    def without_exclude_person(self, pid: int) -> str:
        return self.query_string(exclude_persons=[p for p in self.exclude_persons if p != pid])

    def with_prop(self, key: str, value: str) -> str:
        if self.has_prop(key, value):
            return self.query_string()
        return self.query_string(props=[*self.props, (key, value)])

    def with_prop_alt(self, key: str, value: str) -> str:
        """``key: value`` as an alternative (OR) to the condition on the same key.

        It joins the first OR condition with that key, or makes the last plain condition on the
        key an OR condition; without a condition on the key it is simply required.
        """
        pair = (key, value)
        if self.has_prop(key, value):
            return self.query_string()
        groups = [list(g) for g in self.prop_groups]
        for g in groups:
            if any(k == key for k, _ in g):
                g.append(pair)
                return self.query_string(prop_groups=groups)
        same = [p for p in self.props if p[0] == key]
        if not same:
            return self.query_string(props=[*self.props, pair])
        props = list(self.props)
        props.remove(same[-1])
        return self.query_string(props=props, prop_groups=[*groups, [same[-1], pair]])

    def without_place(self, place_id: int) -> str:
        return self.query_string(places=[p for p in self.places if p != place_id])

    def without_range(self, start: str | None, end: str | None) -> str:
        return self.query_string(dates=[r for r in self.dates if r != (start, end)])

    def with_exclude_prop(self, key: str, value: str) -> str:
        return self.query_string(exclude_props=[*self.exclude_props, (key, value)])

    def with_person(self, pid: int) -> str:
        if pid in self.persons:
            return self.query_string()
        return self.query_string(persons=[*self.persons, pid])

    @property
    def who_alternatives(self) -> bool:
        """Persons / pets can get an alternative: one is chosen, or several with "any"."""
        return self.who_count == 1 or (self.who_count > 1 and self.persons_mode == "any")

    def with_person_alt(self, pid: int) -> str:
        """A person as an alternative: photos with any of the chosen persons and pets."""
        persons = self.persons if pid in self.persons else [*self.persons, pid]
        return self.query_string(persons=persons, persons_mode="any")

    def with_pet_alt(self, pid: int) -> str:
        pets = self.pets if pid in self.pets else [*self.pets, pid]
        return self.query_string(pets=pets, persons_mode="any")


def _copy_value(v: Any) -> Any:
    if isinstance(v, list):
        return [list(x) if isinstance(x, list) else x for x in v]
    return v


# --------------------------------------------------------------------------- SQL building
# Person / cluster / property conditions are "image id IN (ids from an index)" rather than
# correlated EXISTS: SQLite builds each id set once from a covering index and can drive the
# query from the smallest one, instead of probing every image of the library.
# (``image_id`` is never NULL, so NOT IN is the exact negation.)
def _person_images(person_id: int):
    return Image.id.in_(
        select(Face.image_id)
        .join(Cluster, Cluster.id == Face.cluster_id)
        .where(Face.ignored.is_(False), Cluster.person_id == person_id)
    )


def _cluster_images(cluster_id: int):
    return Image.id.in_(
        select(Face.image_id).where(Face.ignored.is_(False), Face.cluster_id == cluster_id)
    )


def _pet_images(pet_id: int):
    return Image.id.in_(
        select(Animal.image_id).where(Animal.ignored.is_(False), Animal.pet_id == pet_id)
    )


def _prop_images(key: str, value: str):
    return Image.id.in_(
        select(ImageProperty.image_id).where(ImageProperty.key == key, ImageProperty.value == value)
    )


def _any_prop_images(alternatives: list[tuple[str, str]]):
    """An OR condition: one id set per key (``value IN (...)``) from the covering index."""
    by_key: dict[str, list[str]] = {}
    for k, v in alternatives:
        by_key.setdefault(k, []).append(v)
    parts = [
        Image.id.in_(
            select(ImageProperty.image_id).where(
                ImageProperty.key == k, ImageProperty.value.in_(part)
            )
        )
        for k, values in by_key.items()
        for part in chunked(values)
    ]
    return parts[0] if len(parts) == 1 else or_(*parts)


def _date_range(start: str | None, end: str | None) -> list[Any]:
    lo, hi = _parse_date(start), _parse_date(end, end=True)
    out = []
    if lo:
        out.append(Image.recorded_at >= lo)
    if hi:
        out.append(Image.recorded_at <= hi)
    return out


def conditions(f: ImageFilter) -> list[Any]:
    conds: list[Any] = []
    if not f.include_missing:
        conds.append(Image.missing.is_(False))
    people = (
        [_person_images(p) for p in f.persons]
        + [_cluster_images(c) for c in f.clusters]
        + [_pet_images(p) for p in f.pets]
    )
    if people:
        conds.append(or_(*people) if f.persons_mode == "any" else and_(*people))
    conds += [_prop_images(k, v) for k, v in f.props]  # props_mode is normalised to "all"
    conds += [_any_prop_images(g) for g in f.prop_groups]
    conds += [not_(_person_images(p)) for p in f.exclude_persons]
    conds += [not_(_pet_images(p)) for p in f.exclude_pets]
    conds += [not_(_prop_images(k, v)) for k, v in f.exclude_props]
    conds += _date_range(f.date_from, f.date_to)
    ranges = [and_(*r) for r in (_date_range(a, b) for a, b in f.dates) if r]
    if ranges:
        conds.append(or_(*ranges))
    if f.folder:
        folder = f.folder.replace("\\", "/").strip("/")
        conds.append(
            or_(
                Image.folder == folder,
                Image.folder.like(like_escape(folder) + "/%", escape="\\"),
            )
        )
    if f.camera:
        conds.append(Image.camera == f.camera)
    if f.places:
        conds.append(or_(*(Image.place_id.in_(part) for part in chunked(f.places))))
    if f.bbox is not None:
        w, s, e, n = f.bbox
        conds.append(Image.lat.between(s, n))
        if w <= e:
            conds.append(Image.lon.between(w, e))
        else:  # the area crosses the date line
            conds.append(or_(Image.lon >= w, Image.lon <= e))
    if f.has_faces is not None:
        face_exists = exists(
            select(Face.id).where(Face.image_id == Image.id, Face.ignored.is_(False))
        )
        conds.append(face_exists if f.has_faces else not_(face_exists))
    if f.stage:
        st = select(StageStatus.image_id).where(
            StageStatus.image_id == Image.id, StageStatus.stage == f.stage
        )
        if f.status:
            st = st.where(StageStatus.status == f.status)
        conds.append(exists(st))
    if f.text:
        like = f"%{like_escape(f.text)}%"
        conds.append(
            or_(
                Image.filename.ilike(like, escape="\\"),
                Image.rel_path.ilike(like, escape="\\"),
                exists(
                    select(VlmResult.image_id).where(
                        VlmResult.image_id == Image.id, VlmResult.caption.ilike(like, escape="\\")
                    )
                ),
            )
        )
    return conds


def face_count_expr():
    return (
        select(func.count(Face.id))
        .where(Face.image_id == Image.id, Face.ignored.is_(False))
        .scalar_subquery()
    )


def _ordered(stmt: Select, f: ImageFilter) -> Select:
    if f.sort == "date_asc":
        return stmt.order_by(Image.recorded_at.asc(), Image.id.asc())
    if f.sort == "name":
        return stmt.order_by(Image.filename.asc(), Image.id.asc())
    if f.sort == "path":
        return stmt.order_by(Image.rel_path.asc(), Image.id.asc())
    if f.sort == "faces":
        return stmt.order_by(face_count_expr().desc(), Image.id.asc())
    if f.sort == "size":
        return stmt.order_by(Image.size_bytes.desc(), Image.id.asc())
    return stmt.order_by(Image.recorded_at.desc(), Image.id.desc())


# sort mode -> (sort expression, column descending?, id descending?)  (must match _ordered)
_KEYSET: dict[str, tuple[Callable[[], Any], bool, bool]] = {
    "name": (lambda: Image.filename, False, False),
    "path": (lambda: Image.rel_path, False, False),
    "size": (lambda: Image.size_bytes, True, False),
}


def _date_neighbours(
    s: Session, f: ImageFilter, image_id: int, value: datetime | None
) -> tuple[int | None, int | None]:
    """Keyset neighbours for the date sorts, as index range scans on ``recorded_at``.

    The gallery order puts undated images last (``date_desc``) or first (``date_asc``), like
    SQLite's NULL ordering, so they are handled as a separate segment instead of comparing a
    ``coalesce()`` expression, which no index can serve.
    """
    conds = conditions(f)
    col = Image.recorded_at

    def first(*where: Any, ascending: bool) -> int | None:
        order = (col.asc(), Image.id.asc()) if ascending else (col.desc(), Image.id.desc())
        return s.scalar(select(Image.id).where(*conds, *where).order_by(*order).limit(1))

    # "up": the next item in ascending (date, id) order, undated before dated; "down": reverse
    if value is None:
        up = first(col.is_(None), Image.id > image_id, ascending=True)
        if up is None:
            up = first(col.is_not(None), ascending=True)
        down = first(col.is_(None), Image.id < image_id, ascending=False)
    else:
        up = first(col >= value, or_(col > value, Image.id > image_id), ascending=True)
        down = first(col <= value, or_(col < value, Image.id < image_id), ascending=False)
        if down is None:
            down = first(col.is_(None), ascending=False)
    return (up, down) if f.sort == "date_desc" else (down, up)


def neighbours(s: Session, f: ImageFilter, image_id: int) -> tuple[int | None, int | None]:
    """Previous / next image in the gallery order of ``f`` (keyset queries, no full id list)."""
    if f.sort in ("date_desc", "date_asc"):
        row = s.execute(select(Image.recorded_at).where(Image.id == image_id)).first()
        return (None, None) if row is None else _date_neighbours(s, f, image_id, row[0])
    if f.sort not in _KEYSET:
        ids = all_image_ids(s, f)
        if image_id not in ids:
            return None, None
        pos = ids.index(image_id)
        return (ids[pos - 1] if pos > 0 else None, ids[pos + 1] if pos + 1 < len(ids) else None)
    make_expr, col_desc, id_desc = _KEYSET[f.sort]
    expr = make_expr()
    value = s.scalar(select(expr).where(Image.id == image_id))
    if value is None:
        return None, None
    conds = conditions(f)

    def step(forward: bool) -> int | None:
        later_col = (expr < value) if (col_desc == forward) else (expr > value)
        later_id = (Image.id < image_id) if (id_desc == forward) else (Image.id > image_id)
        order_col = expr.desc() if (col_desc == forward) else expr.asc()
        order_id = Image.id.desc() if (id_desc == forward) else Image.id.asc()
        return s.scalar(
            select(Image.id)
            .where(*conds, or_(later_col, and_(expr == value, later_id)))
            .order_by(order_col, order_id)
            .limit(1)
        )

    return step(False), step(True)


def folder_children(s: Session, f: ImageFilter) -> list[tuple[str, str, int]]:
    """Immediate sub-folders of ``f.folder`` (or of the roots) with image counts.

    Returns ``(full_folder, display_name, count)``; images directly inside the current folder
    are reported with ``display_name == ""``.
    """
    prefix = (f.folder or "").replace("\\", "/").strip("/")
    counts: dict[str, int] = {}
    stmt = select(Image.folder, func.count(Image.id)).where(*conditions(f)).group_by(Image.folder)
    for folder, n in s.execute(stmt):
        folder = (folder or "").strip("/")
        rest = folder[len(prefix) :].lstrip("/") if prefix else folder
        child = rest.split("/", 1)[0] if rest else ""
        counts[child] = counts.get(child, 0) + n
    out = [("/".join(p for p in (prefix, child) if p), child, n) for child, n in counts.items()]
    out.sort(key=lambda t: (t[1] == "", -t[2], t[1].casefold()))
    return out


def image_ids_subquery(f: ImageFilter):
    return select(Image.id).where(*conditions(f))


def select_images(f: ImageFilter) -> Select:
    return _ordered(select(Image).where(*conditions(f)), f)


def count_images(s: Session, f: ImageFilter) -> int:
    return s.scalar(select(func.count()).select_from(Image).where(*conditions(f))) or 0


def page_images(s: Session, f: ImageFilter, offset: int = 0, limit: int = 120) -> list[Image]:
    return list(s.scalars(select_images(f).offset(offset).limit(limit)))


def all_image_ids(s: Session, f: ImageFilter) -> list[int]:
    return list(s.scalars(_ordered(select(Image.id).where(*conditions(f)), f)))


# --------------------------------------------------------------------------- facets
def property_keys(s: Session) -> list[str]:
    """All distinct property keys, sorted.

    A loose index scan: one index seek per key instead of reading every property row.
    """
    keys: list[str] = []
    key = s.scalar(select(func.min(ImageProperty.key)))
    while key is not None:
        keys.append(key)
        key = s.scalar(select(func.min(ImageProperty.key)).where(ImageProperty.key > key))
    return sorted(keys)


def property_counts(
    s: Session,
    f: ImageFilter | None = None,
    key: str | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, list[tuple[str, int]]]:
    """``{key: [(value, count), ...]}`` among the filtered images, most frequent first.

    ``key`` / ``keys`` restrict the (expensive) aggregation to the properties actually shown.
    """
    stmt = select(ImageProperty.key, ImageProperty.value, func.count(ImageProperty.image_id))
    if f is not None and not f.is_empty:
        stmt = stmt.where(ImageProperty.image_id.in_(image_ids_subquery(f)))
    elif s.scalar(select(Image.id).where(Image.missing.is_(True)).limit(1)) is not None:
        # Every property row belongs to an image (foreign key), so "not one of the few
        # missing images" equals "a present image" - and is much cheaper to check for
        # millions of rows. Without missing images no check is needed at all.
        stmt = stmt.where(
            ImageProperty.image_id.not_in(select(Image.id).where(Image.missing.is_(True)))
        )
    if key:
        stmt = stmt.where(ImageProperty.key == key)
    if keys is not None:
        stmt = stmt.where(ImageProperty.key.in_(list(keys)))
    stmt = stmt.group_by(ImageProperty.key, ImageProperty.value).order_by(
        ImageProperty.key, func.count(ImageProperty.image_id).desc(), ImageProperty.value
    )
    out: dict[str, list[tuple[str, int]]] = {}
    for k, v, n in s.execute(stmt):
        out.setdefault(k, []).append((v, n))
    return out


def person_counts(s: Session, f: ImageFilter | None = None) -> list[tuple[Person, int]]:
    stmt = (
        select(Person, func.count(func.distinct(Face.image_id)))
        .join(Cluster, Cluster.person_id == Person.id)
        .join(Face, Face.cluster_id == Cluster.id)
        .where(Face.ignored.is_(False))
    )
    if f is not None and not f.is_empty:
        stmt = stmt.where(Face.image_id.in_(image_ids_subquery(f)))
    stmt = stmt.group_by(Person.id).order_by(
        func.count(func.distinct(Face.image_id)).desc(), Person.id
    )
    return [(p, n) for p, n in s.execute(stmt)]


def pet_counts(s: Session, f: ImageFilter | None = None) -> list[tuple[Pet, int]]:
    """Named pets with the number of (filtered) photos they are in, most first."""
    photos = func.count(func.distinct(Animal.image_id))
    stmt = (
        select(Pet, photos)
        .join(Animal, Animal.pet_id == Pet.id)
        .where(Animal.ignored.is_(False), Pet.name.is_not(None))
    )
    if f is not None and not f.is_empty:
        stmt = stmt.where(Animal.image_id.in_(image_ids_subquery(f)))
    stmt = stmt.group_by(Pet.id).order_by(photos.desc(), Pet.id)
    return [(p, n) for p, n in s.execute(stmt)]


def date_histogram(s: Session, f: ImageFilter, level: str = "year") -> list[tuple[str, int]]:
    fmt = {"year": "%Y", "month": "%Y-%m", "day": "%Y-%m-%d"}[level]
    bucket = func.strftime(fmt, Image.recorded_at)
    stmt = (
        select(bucket, func.count(Image.id))
        .where(*conditions(f), Image.recorded_at.is_not(None))
        .group_by(bucket)
        .order_by(bucket)
    )
    return [(b, n) for b, n in s.execute(stmt)]


def column_counts(
    s: Session, f: ImageFilter, column: Any, limit: int = 50
) -> list[tuple[Any, int]]:
    stmt = (
        select(column, func.count(Image.id))
        .where(*conditions(f))
        .group_by(column)
        .order_by(func.count(Image.id).desc())
        .limit(limit)
    )
    return [(v, n) for v, n in s.execute(stmt)]
