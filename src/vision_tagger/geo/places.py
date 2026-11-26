"""Places: GPS-tagged photos grouped by distance, each with a name to filter by.

1. Photos the user named a place for keep it: a user-named place claims every photo within its
   radius (plus ``geo.cluster_km``) of its centre, like a locked face cluster.
2. All other photos are grouped with DBSCAN (haversine distance, ``eps = geo.cluster_km``,
   every photo belongs to a place). Groups wider than ``geo.max_place_km`` are split in two
   until they fit, so a road trip does not become one place. Photos are first gathered into
   small cells, so 100,000 photos taken at home cost no more than a few hundred.
3. Groups take over the id of the old place they share most photos with, so links such as
   ``/images?place=12`` stay valid; places left without photos are removed.
4. Names come from the offline GeoNames towns (``geo.gazetteer``) and, if enabled, from
   OpenStreetMap (``geo.nominatim``). A place keeps its name while its centre stays put.
5. Every photo gets ``images.place_id`` and the properties ``place``, ``city`` and ``country``
   (source ``place``), so the usual property filters, facets and search documents work;
   the search index is re-queued for photos whose place changed.

Clustering and naming happen outside any write transaction; writes are short and chunked.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import delete, exists, insert, select, update

from ..db.models import Image, ImageProperty, Place, PlaceSource, PropertyAlias
from ..db.util import chunked
from ..i18n import _
from ..pipeline.control import mark_pending
from . import EARTH_KM, PlaceName, coords_name, haversine_km

if TYPE_CHECKING:
    from ..config import Settings
    from ..db.session import Database
    from .gazetteer import Gazetteer
    from .nominatim import Nominatim

log = logging.getLogger(__name__)

PROPERTY_SOURCE = "place"
PLACE_KEYS = ("place", "city", "country")
#: a place keeps its name while its centre moves less than this
RENAME_AFTER_KM = 0.5


@dataclass
class PlacesResult:
    places: int = 0
    new: int = 0
    removed: int = 0
    photos: int = 0
    changed: int = 0
    named: int = 0
    osm_requests: int = 0
    skipped: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.skipped:
            return _("Places: up to date")
        text = _(
            "Places: %(places)s (%(new)s new, %(removed)s removed), %(photos)s photos with GPS, "
            "%(changed)s changed",
            places=self.places,
            new=self.new,
            removed=self.removed,
            photos=self.photos,
            changed=self.changed,
        )
        return text + "".join(f"; {n}" for n in self.notes)


@dataclass(eq=False)
class _Group:
    image_ids: list[int]
    lat: float
    lon: float
    radius_m: float
    place_id: int | None = None
    locked: bool = False
    name: PlaceName | None = None
    index: np.ndarray | None = None  # positions of the photos in find_places' arrays


# --------------------------------------------------------------------------- geometry
def _centroid(lat: np.ndarray, lon: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean position (via unit vectors, so it works across the date line)."""
    la, lo = np.radians(lat), np.radians(lon)
    x = float(np.sum(w * np.cos(la) * np.cos(lo)))
    y = float(np.sum(w * np.cos(la) * np.sin(lo)))
    z = float(np.sum(w * np.sin(la)))
    hyp = (x * x + y * y) ** 0.5
    return float(np.degrees(np.arctan2(z, hyp))), float(np.degrees(np.arctan2(y, x)))


def _distances_km(lat: np.ndarray, lon: np.ndarray, clat: float, clon: float) -> np.ndarray:
    p1, p2 = np.radians(lat), np.radians(clat)
    dp, dl = p2 - p1, np.radians(clon - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _cells(lat: np.ndarray, lon: np.ndarray, size_km: float):
    """Gather positions into grid cells: (cell lat, cell lon, weights, cell index per point)."""
    step = max(size_km, 0.001) / 111.32
    keys = np.stack([np.round(lat / step), np.round(lon / step)], axis=1).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)
    clat = np.bincount(inverse, weights=lat, minlength=n) / counts
    clon = np.bincount(inverse, weights=lon, minlength=n) / counts
    return clat, clon, counts.astype(np.float64), inverse


def _split(
    idx: np.ndarray, lat: np.ndarray, lon: np.ndarray, w: np.ndarray, max_km: float
) -> list[np.ndarray]:
    """Halve a group of cells until each part is at most ``max_km`` from its centre.

    Each cut is at the weighted median along the group's longest axis (principal component of
    the positions in a local km grid): deterministic, and both halves get similar weight.
    """
    out: list[np.ndarray] = []
    todo = [idx]
    while todo:
        part = todo.pop()
        clat, clon = _centroid(lat[part], lon[part], w[part])
        if len(part) < 2 or _distances_km(lat[part], lon[part], clat, clon).max() <= max_km:
            out.append(part)
            continue
        x = (lon[part] - clon) * 111.32 * np.cos(np.radians(clat))
        y = (lat[part] - clat) * 111.32
        xy = np.column_stack([x, y])
        cov = np.cov(xy, rowvar=False, aweights=w[part])
        axis = np.linalg.eigh(cov)[1][:, -1]
        proj = xy @ axis
        order = np.argsort(proj, kind="stable")
        cum = np.cumsum(w[part][order])
        cut = int(np.searchsorted(cum, cum[-1] / 2))
        cut = min(max(cut, 1), len(part) - 1)  # never an empty half
        todo.extend([part[order[:cut]], part[order[cut:]]])
    return out


def cluster_positions(
    lat: np.ndarray, lon: np.ndarray, eps_km: float, max_km: float
) -> list[np.ndarray]:
    """Indices of the positions in each place (DBSCAN on cells, large groups split)."""
    if len(lat) == 0:
        return []
    from sklearn.cluster import DBSCAN

    clat, clon, weights, inverse = _cells(lat, lon, eps_km / 4)
    labels = DBSCAN(
        eps=eps_km / EARTH_KM, min_samples=1, metric="haversine", algorithm="ball_tree"
    ).fit_predict(np.radians(np.column_stack([clat, clon])), sample_weight=weights)
    cell_group = np.empty(len(clat), dtype=np.int64)
    n_groups = 0
    for label in np.unique(labels):
        for part in _split(np.flatnonzero(labels == label), clat, clon, weights, max_km):
            cell_group[part] = n_groups
            n_groups += 1
    point_group = cell_group[inverse]
    order = np.argsort(point_group, kind="stable")
    bounds = np.searchsorted(point_group[order], np.arange(n_groups + 1))
    return [order[bounds[g] : bounds[g + 1]] for g in range(n_groups)]


def _group(ids: np.ndarray, lat: np.ndarray, lon: np.ndarray, index: np.ndarray) -> _Group:
    glat, glon = _centroid(lat[index], lon[index], np.ones(len(index)))
    radius = float(_distances_km(lat[index], lon[index], glat, glon).max()) * 1000
    return _Group([int(i) for i in ids[index]], glat, glon, round(radius, 1), index=index)


def _merge_same_names(
    groups: list[_Group], ids: np.ndarray, lat: np.ndarray, lon: np.ndarray, max_km: float
) -> list[_Group]:
    """Neighbouring groups with the same automatic name become one place, as long as it stays
    within ``max_km``: a walk through a town with a photo every kilometre is one place, not ten.
    """
    out = [g for g in groups if g.locked or g.name is None or g.index is None]
    by_name: dict[tuple[str, str], list[_Group]] = defaultdict(list)
    for g in groups:
        if not (g.locked or g.name is None or g.index is None):
            by_name[(g.name.name, g.name.source)].append(g)
    for same in by_name.values():
        todo = sorted(same, key=lambda g: (-len(g.image_ids), g.lat, g.lon))
        while todo:
            base = todo.pop(0)
            grown = True
            while grown and todo:
                grown = False
                todo.sort(key=lambda o: haversine_km(base.lat, base.lon, o.lat, o.lon))
                for other in todo:
                    if haversine_km(base.lat, base.lon, other.lat, other.lon) > 2 * max_km:
                        break
                    assert base.index is not None and other.index is not None
                    merged = _group(ids, lat, lon, np.concatenate([base.index, other.index]))
                    if merged.radius_m > max_km * 1000:
                        continue
                    merged.name = base.name
                    merged.place_id = base.place_id if base.place_id is not None else other.place_id
                    base = merged
                    todo.remove(other)
                    grown = True
                    break
            out.append(base)
    return out


# --------------------------------------------------------------------------- naming
class Namer:
    """Names positions: OpenStreetMap (if enabled), else GeoNames, else the coordinates."""

    def __init__(
        self,
        db: Database,
        settings: Settings,
        *,
        gazetteer: Gazetteer | None = None,
        geocoder: Nominatim | None = None,
        refresh: bool = False,
    ) -> None:
        self.db = db
        self.settings = settings
        self.refresh = refresh
        self._gazetteer = gazetteer
        self._gazetteer_loaded = gazetteer is not None
        self._geocoder = geocoder
        self._own_geocoder = False
        self.osm_enabled = settings.geo.nominatim or geocoder is not None
        self.osm_error: str | None = None

    @property
    def gazetteer(self) -> Gazetteer | None:
        if not self._gazetteer_loaded:
            from .gazetteer import load_gazetteer

            self._gazetteer = load_gazetteer(self.settings)
            self._gazetteer_loaded = True
        return self._gazetteer

    @property
    def preferred_source(self) -> str:
        return PlaceSource.OSM if self.osm_enabled and not self.osm_error else PlaceSource.GEONAMES

    def _osm(self, lat: float, lon: float) -> PlaceName | None:
        from .nominatim import GeocoderUnavailable, Nominatim, osm_name

        if not self.osm_enabled or self.osm_error:
            return None
        if self._geocoder is None:
            cfg = self.settings.geo
            self._geocoder = Nominatim(cfg.nominatim_url, language=cfg.language)
            self._own_geocoder = True
        try:
            return osm_name(self._geocoder.lookup(self.db, lat, lon, refresh=self.refresh))
        except GeocoderUnavailable as exc:
            self.osm_error = str(exc)
            log.warning("OpenStreetMap names unavailable for this run: %s", exc)
            return None

    def name(self, lat: float, lon: float, *, offline_only: bool = False) -> PlaceName:
        if not offline_only:
            osm = self._osm(lat, lon)
            if osm is not None:
                return osm
        gaz = self.gazetteer
        named = gaz.describe(lat, lon) if gaz is not None else None
        if named is not None:
            return named
        return PlaceName(coords_name(lat, lon), None, None, None, PlaceSource.COORDS)

    @property
    def osm_requests(self) -> int:
        return self._geocoder.requests if self._geocoder is not None else 0

    def close(self) -> None:
        if self._own_geocoder and self._geocoder is not None:
            self._geocoder.close()


def _keeps_name(old: Place, group: _Group, namer: Namer) -> bool:
    if namer.refresh or old.source in (PlaceSource.COORDS, PlaceSource.USER):
        return False
    if old.source == PlaceSource.GEONAMES and namer.preferred_source == PlaceSource.OSM:
        return False  # OpenStreetMap names were switched on
    return haversine_km(old.lat, old.lon, group.lat, group.lon) < RENAME_AFTER_KM


# --------------------------------------------------------------------------- main entry
def needs_update(db: Database, settings: Settings) -> bool:
    """Photos without a place (or with a place but no position) or unnamed places exist."""
    with db.session() as s:
        if s.scalar(
            select(Image.id).where(Image.lat.is_not(None), Image.place_id.is_(None)).limit(1)
        ):
            return True
        if s.scalar(
            select(Image.id).where(Image.lat.is_(None), Image.place_id.is_not(None)).limit(1)
        ):
            return True
        pending = [PlaceSource.COORDS]
        if settings.geo.nominatim:
            pending.append(PlaceSource.GEONAMES)  # waiting for an OpenStreetMap name
        return bool(
            s.scalar(
                select(Place.id)
                .where(Place.user_named.is_(False), Place.source.in_(pending))
                .limit(1)
            )
        )


def find_places(
    db: Database,
    settings: Settings,
    *,
    refresh: bool = False,
    gazetteer: Gazetteer | None = None,
    geocoder: Nominatim | None = None,
    progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PlacesResult:
    """Group the GPS-tagged photos into places and name them (see the module docstring).

    Without ``refresh`` nothing happens unless :func:`needs_update`; ``refresh`` regroups
    everything and names every place again (OpenStreetMap answers are asked again too).
    """
    progress = progress or (lambda _m: None)
    should_stop = should_stop or (lambda: False)
    result = PlacesResult()
    if not refresh and not needs_update(db, settings):
        result.skipped = True
        return result
    cfg = settings.geo
    eps_km = max(0.01, cfg.cluster_km)
    max_km = max(eps_km, cfg.max_place_km)

    # ---------------------------------------------------------------- read (no write lock)
    with db.session() as s:
        rows = s.execute(
            select(Image.id, Image.lat, Image.lon, Image.place_id).where(Image.lat.is_not(None))
        ).all()
        unpositioned = list(
            s.scalars(select(Image.id).where(Image.lat.is_(None), Image.place_id.is_not(None)))
        )
        old_places = {p.id: p for p in s.scalars(select(Place))}
        s.expunge_all()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    lat = np.array([r[1] for r in rows], dtype=np.float64)
    lon = np.array([r[2] for r in rows], dtype=np.float64)
    current = {int(r[0]): r[3] for r in rows}
    result.photos = len(rows)
    progress(_("Grouping %(n)s photos with GPS…", n=len(rows)))

    # ---------------------------------------------------------------- locked places claim
    locked = [p for p in old_places.values() if p.user_named]
    claimed = np.full(len(ids), -1, dtype=np.int64)
    if locked and len(ids):
        best = np.full(len(ids), np.inf)
        for p in locked:
            d = _distances_km(lat, lon, p.lat, p.lon)
            reach = p.radius_m / 1000 + eps_km
            take = (d <= reach) & (d < best)
            claimed[take] = p.id
            best[take] = d[take]
    groups: list[_Group] = []
    for p in locked:
        members = np.flatnonzero(claimed == p.id)
        if len(members):
            groups.append(
                _Group(
                    [int(i) for i in ids[members]],
                    p.lat,
                    p.lon,
                    p.radius_m,
                    place_id=p.id,
                    locked=True,
                )
            )

    # ---------------------------------------------------------------- cluster the rest
    free = np.flatnonzero(claimed < 0)
    new_groups = [
        _group(ids, lat, lon, free[members])
        for members in cluster_positions(lat[free], lon[free], eps_km, max_km)
    ]

    # ---------------------------------------------------------------- keep ids stable
    candidates: list[tuple[int, int, int]] = []
    for gi, g in enumerate(new_groups):
        overlap = Counter(current.get(i) for i in g.image_ids)
        for pid, n in overlap.items():
            if pid is not None and pid in old_places and not old_places[pid].user_named:
                candidates.append((n, gi, pid))
    taken: set[int] = set()
    for _n, gi, pid in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
        if new_groups[gi].place_id is None and pid not in taken:
            new_groups[gi].place_id = pid
            taken.add(pid)
    groups += new_groups

    # ---------------------------------------------------------------- names
    namer = Namer(db, settings, gazetteer=gazetteer, geocoder=geocoder, refresh=refresh)
    try:
        todo = []
        for g in groups:
            old = old_places.get(g.place_id) if g.place_id is not None else None
            if g.locked:
                continue
            if old is not None and _keeps_name(old, g, namer):
                g.name = PlaceName(old.name, old.locality, old.admin, old.country, old.source)
                continue
            todo.append(g)
        stopped = False
        for n, g in enumerate(sorted(todo, key=lambda g: -len(g.image_ids)), 1):
            if not stopped and should_stop():
                stopped = True
                result.notes.append(_("stopped: the remaining places got offline names"))
            if n % 10 == 1 or n == len(todo):
                progress(_("Naming places: %(done)s of %(total)s", done=n, total=len(todo)))
            g.name = namer.name(g.lat, g.lon, offline_only=stopped)
            result.named += 1
        result.osm_requests = namer.osm_requests
        if namer.osm_error:
            result.notes.append(
                _("OpenStreetMap names unavailable (%(error)s)", error=namer.osm_error)
            )
    finally:
        namer.close()
    groups = _merge_same_names(groups, ids, lat, lon, max_km)

    # ---------------------------------------------------------------- write places
    progress(_("Saving places…"))
    with db.write() as s:
        for g in groups:
            if g.locked:
                continue
            place = s.get(Place, g.place_id) if g.place_id is not None else None
            name = g.name
            assert name is not None
            if place is None:  # new (or removed since it was read)
                place = Place(name=name.name, lat=g.lat, lon=g.lon)
                s.add(place)
                result.new += 1
            place.lat, place.lon, place.radius_m = g.lat, g.lon, g.radius_m
            place.name = name.name
            place.locality = name.locality
            place.admin = name.admin
            place.country = name.country
            place.source = name.source
            s.flush()
            g.place_id = place.id

    # ---------------------------------------------------------------- assign photos
    desired: dict[int, int] = {}
    for g in groups:
        assert g.place_id is not None
        for i in g.image_ids:
            desired[i] = g.place_id
    moves: dict[int | None, list[int]] = defaultdict(list)
    for image_id, pid in desired.items():
        if current.get(image_id) != pid:
            moves[pid].append(image_id)
    moves[None].extend(unpositioned)
    for pid, image_ids in moves.items():
        for part in chunked(sorted(image_ids), 500):
            with db.write() as s:
                s.execute(update(Image).where(Image.id.in_(part)).values(place_id=pid))
    result.changed = sum(len(v) for v in moves.values())

    # ---------------------------------------------------------------- properties
    progress(_("Updating the place properties…"))
    _sync_properties(db, desired, unpositioned)

    with db.write() as s:
        res = s.execute(
            delete(Place).where(
                Place.user_named.is_(False),
                ~exists(select(Image.id).where(Image.place_id == Place.id)),
            )
        )
        result.removed = res.rowcount or 0
        result.places = len({pid for pid in desired.values()})
    log.info(result.message)
    return result


def _aliases(db: Database) -> dict[tuple[str, str], str | None]:
    with db.session() as s:
        return {
            (a.key, a.from_value): a.to_value
            for a in s.scalars(select(PropertyAlias).where(PropertyAlias.key.in_(PLACE_KEYS)))
        }


def place_properties(
    place: Place | Any, aliases: dict[tuple[str, str], str | None] | None = None
) -> set[tuple[str, str]]:
    """The ``(key, value)`` properties of a photo in ``place`` (renames of the properties page
    applied: a value merged or deleted there stays merged or deleted)."""
    out: set[tuple[str, str]] = set()
    for key, value in (("place", place.name), ("city", place.locality), ("country", place.country)):
        if not value:
            continue
        value = (aliases or {}).get((key, value), value)
        if value:
            out.add((key, value))
    return out


def _sync_properties(
    db: Database, desired: dict[int, int], unpositioned: Iterable[int], *, full: bool = True
) -> None:
    """Make the place properties of photos match their place.

    ``full`` also removes place properties of photos outside ``desired`` (no position any
    more); otherwise only the photos in ``desired`` are looked at.
    """
    aliases = _aliases(db)
    want: dict[int, set[tuple[str, str]]] = {}
    have: dict[int, set[tuple[str, str]]] = defaultdict(set)
    stmt = select(ImageProperty.image_id, ImageProperty.key, ImageProperty.value).where(
        ImageProperty.key.in_(PLACE_KEYS)
    )
    with db.session() as s:
        places = {p.id: place_properties(p, aliases) for p in s.scalars(select(Place))}
        want = {image_id: places.get(pid, set()) for image_id, pid in desired.items()}
        for image_id in unpositioned:
            want.setdefault(image_id, set())
        parts = (
            [stmt]
            if full
            else [stmt.where(ImageProperty.image_id.in_(p)) for p in chunked(sorted(want))]
        )
        for query in parts:
            for image_id, key, value in s.execute(query):
                have[image_id].add((key, value))
    scope = set(want) | set(have) if full else set(want)
    stale = [i for i in scope if want.get(i, set()) != have.get(i, set())]
    for part in chunked(sorted(stale), 500):
        with db.write() as s:
            s.execute(
                delete(ImageProperty).where(
                    ImageProperty.image_id.in_(part), ImageProperty.key.in_(PLACE_KEYS)
                )
            )
            rows = [
                {"image_id": i, "key": k, "value": v, "source": PROPERTY_SOURCE}
                for i in part
                for k, v in sorted(want.get(i, set()))
            ]
            if rows:
                s.execute(insert(ImageProperty), rows)
            mark_pending(s, "embed", part)  # the search document mentions the place


# --------------------------------------------------------------------------- user actions
def rename_place(db: Database, settings: Settings, place_id: int, name: str) -> Place:
    """Name a place (kept from now on); an empty name returns to the automatic name."""
    name = " ".join(name.split())[:200]
    with db.session() as s:
        place = s.get(Place, place_id)
        if place is None:
            raise LookupError(f"place {place_id} not found")
        lat, lon = place.lat, place.lon
    auto: PlaceName | None = None
    if not name:  # naming happens before the write transaction (it may load data)
        namer = Namer(db, settings)
        try:
            auto = namer.name(lat, lon, offline_only=True)
        finally:
            namer.close()
    with db.write() as s:
        place = s.get(Place, place_id)
        if place is None:
            raise LookupError(f"place {place_id} not found")
        if auto is not None:
            place.name, place.source, place.user_named = auto.name, auto.source, False
            place.locality, place.admin, place.country = auto.locality, auto.admin, auto.country
        else:
            place.name, place.source, place.user_named = name, PlaceSource.USER, True
        s.flush()
        s.expunge(place)
    _resync_place(db, place_id)
    return place


def _resync_place(db: Database, place_id: int) -> None:
    with db.session() as s:
        image_ids = list(s.scalars(select(Image.id).where(Image.place_id == place_id)))
    _sync_properties(db, dict.fromkeys(image_ids, place_id), [], full=False)
