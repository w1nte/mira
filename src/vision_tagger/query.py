"""Image filter builder shared by the gallery, facet counts, timeline and export.

Filters are encoded in the URL query string so every view can be bookmarked:

=========  =====================================================  =========
parameter  meaning                                                example
=========  =====================================================  =========
``p``      person id (repeatable)                                 ``p=3``
``c``      face cluster id (repeatable, for unnamed clusters)     ``c=17``
``pm``     combine persons/clusters with ``all`` (AND) / ``any``  ``pm=any``
``t``      property ``key:value`` (repeatable)                    ``t=location_type:beach``
``tm``     combine properties with ``all`` (AND) / ``any`` (OR)   ``tm=all``
``xp``     exclude person id                                      ``xp=4``
``xt``     exclude property ``key:value``                         ``xt=work_image:yes``
``from``   recorded on/after date (YYYY-MM-DD, YYYY-MM or YYYY)   ``from=2021``
``to``     recorded on/before date                                ``to=2021-08``
``folder`` folder prefix (relative to the root)                   ``folder=2021/Italy``
``camera`` camera model                                           ``camera=iPhone 13``
``faces``  ``yes`` / ``no``                                       ``faces=yes``
``stage``  + ``status``: pipeline state                           ``stage=vlm&status=error``
``q``      text search in file name, path and caption             ``q=birthday``
``sort``   ``date_desc`` (default), ``date_asc``, ``name``, ``path``, ``faces``, ``size``
=========  =====================================================  =========
"""

from __future__ import annotations

import calendar
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from sqlalchemy import Select, and_, exists, func, not_, or_, select
from sqlalchemy.orm import Session

from .db.models import Cluster, Face, Image, ImageProperty, Person, StageStatus, VlmResult
from .db.util import like_escape

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
    persons_mode: str = "all"
    props: list[tuple[str, str]] = field(default_factory=list)
    props_mode: str = "all"
    exclude_persons: list[int] = field(default_factory=list)
    exclude_props: list[tuple[str, str]] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    folder: str | None = None
    camera: str | None = None
    has_faces: bool | None = None
    stage: str | None = None
    status: str | None = None
    text: str | None = None
    include_missing: bool = False
    sort: str = "date_desc"

    # ------------------------------------------------------------------ (de)serialisation
    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ImageFilter:
        faces = params.get("faces")
        sort = params.get("sort") or "date_desc"
        return cls(
            persons=_ints(_getlist(params, "p")),
            clusters=_ints(_getlist(params, "c")),
            persons_mode="any" if params.get("pm") == "any" else "all",
            props=_pairs(_getlist(params, "t")),
            props_mode="any" if params.get("tm") == "any" else "all",
            exclude_persons=_ints(_getlist(params, "xp")),
            exclude_props=_pairs(_getlist(params, "xt")),
            date_from=params.get("from") or None,
            date_to=params.get("to") or None,
            folder=params.get("folder") or None,
            camera=params.get("camera") or None,
            has_faces=True if faces == "yes" else False if faces == "no" else None,
            stage=params.get("stage") or None,
            status=params.get("status") or None,
            text=(params.get("q") or "").strip() or None,
            include_missing=params.get("missing") == "1",
            sort=sort if sort in SORTS else "date_desc",
        )

    def to_params(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        out += [("p", str(p)) for p in self.persons]
        out += [("c", str(c)) for c in self.clusters]
        if self.persons_mode == "any" and (len(self.persons) + len(self.clusters)) > 1:
            out.append(("pm", "any"))
        out += [("t", f"{k}:{v}") for k, v in self.props]
        if self.props_mode == "any" and len(self.props) > 1:
            out.append(("tm", "any"))
        out += [("xp", str(p)) for p in self.exclude_persons]
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
        if self.has_faces is not None:
            out.append(("faces", "yes" if self.has_faces else "no"))
        if self.include_missing:
            out.append(("missing", "1"))
        if self.sort != "date_desc":
            out.append(("sort", self.sort))
        return out

    def query_string(self, **changes: Any) -> str:
        f = self.copy(**changes) if changes else self
        return urlencode(f.to_params())

    def copy(self, **changes: Any) -> ImageFilter:
        data = {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}
        data.update(changes)
        return ImageFilter(**data)

    @property
    def is_empty(self) -> bool:
        return not [p for p in self.to_params() if p[0] != "sort"]

    # convenience for templates ---------------------------------------
    def without_person(self, pid: int) -> str:
        return self.query_string(persons=[p for p in self.persons if p != pid])

    def without_cluster(self, cid: int) -> str:
        return self.query_string(clusters=[c for c in self.clusters if c != cid])

    def without_prop(self, key: str, value: str) -> str:
        return self.query_string(props=[p for p in self.props if p != (key, value)])

    def without_exclude_prop(self, key: str, value: str) -> str:
        return self.query_string(exclude_props=[p for p in self.exclude_props if p != (key, value)])

    def without_exclude_person(self, pid: int) -> str:
        return self.query_string(exclude_persons=[p for p in self.exclude_persons if p != pid])

    def with_prop(self, key: str, value: str) -> str:
        if (key, value) in self.props:
            return self.query_string()
        return self.query_string(props=[*self.props, (key, value)])

    def with_exclude_prop(self, key: str, value: str) -> str:
        return self.query_string(exclude_props=[*self.exclude_props, (key, value)])

    def with_person(self, pid: int) -> str:
        if pid in self.persons:
            return self.query_string()
        return self.query_string(persons=[*self.persons, pid])


# --------------------------------------------------------------------------- SQL building
def _person_exists(person_id: int):
    return exists(
        select(Face.id)
        .join(Cluster, Cluster.id == Face.cluster_id)
        .where(Face.image_id == Image.id, Face.ignored.is_(False), Cluster.person_id == person_id)
    )


def _cluster_exists(cluster_id: int):
    return exists(
        select(Face.id).where(
            Face.image_id == Image.id, Face.ignored.is_(False), Face.cluster_id == cluster_id
        )
    )


def _prop_exists(key: str, value: str):
    return exists(
        select(ImageProperty.id).where(
            ImageProperty.image_id == Image.id,
            ImageProperty.key == key,
            ImageProperty.value == value,
        )
    )


def conditions(f: ImageFilter) -> list[Any]:
    conds: list[Any] = []
    if not f.include_missing:
        conds.append(Image.missing.is_(False))
    people = [_person_exists(p) for p in f.persons] + [_cluster_exists(c) for c in f.clusters]
    if people:
        conds.append(or_(*people) if f.persons_mode == "any" else and_(*people))
    props = [_prop_exists(k, v) for k, v in f.props]
    if props:
        conds.append(or_(*props) if f.props_mode == "any" else and_(*props))
    conds += [not_(_person_exists(p)) for p in f.exclude_persons]
    conds += [not_(_prop_exists(k, v)) for k, v in f.exclude_props]
    start, end = _parse_date(f.date_from), _parse_date(f.date_to, end=True)
    if start:
        conds.append(Image.recorded_at >= start)
    if end:
        conds.append(Image.recorded_at <= end)
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


_MIN_DATE = datetime(1, 1, 1)

# sort mode -> (sort expression, column descending?, id descending?)  (must match _ordered)
_KEYSET: dict[str, tuple[Callable[[], Any], bool, bool]] = {
    "date_desc": (lambda: func.coalesce(Image.recorded_at, _MIN_DATE), True, True),
    "date_asc": (lambda: func.coalesce(Image.recorded_at, _MIN_DATE), False, False),
    "name": (lambda: Image.filename, False, False),
    "path": (lambda: Image.rel_path, False, False),
    "size": (lambda: Image.size_bytes, True, False),
}


def neighbours(s: Session, f: ImageFilter, image_id: int) -> tuple[int | None, int | None]:
    """Previous / next image in the gallery order of ``f`` (keyset queries, no full id list)."""
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
    elif f is None or not f.include_missing:
        stmt = stmt.where(
            ImageProperty.image_id.in_(select(Image.id).where(Image.missing.is_(False)))
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
