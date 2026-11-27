"""Date groups: photos that belong together, to check and correct their dates at once.

Photo dates go wrong in batches: scans and copies carry the time they were scanned or copied,
messenger apps strip EXIF, a camera clock was never set. Fixing them one by one is slow, so
:func:`group_photos` puts the photos that seem to belong together into groups, and
:func:`apply_dates` corrects the selected photos of a group in one go.

Two photos are linked into the same group when

* they are in the same folder and next to each other in time, and both dates are trustworthy
  (EXIF, file name, set by hand) and at most :data:`EVENT_GAP` apart (one event);
* they are in the same folder, next to each other in file name order, and either look alike
  (visual vectors, if the "Visual search" stage ran) or are consecutive shots (``IMG_0104``,
  ``IMG_0105``) or scans (``scan_007``, ``scan_008``);
* they are copies of each other (the same pHash), wherever they are.

Look, file name and copies never link groups whose trustworthy dates are more than
:data:`FAR_APART` apart: those are different moments (the same room at two Christmases), not one
wrong date, and a photo with a file time must not bridge them. A photo
with a file-time date, though, joins the group it looks or is named like, and then gets the
date of its group: the date of its nearest trustworthy neighbour in file name order
(:attr:`DateGroup.suggestions`).

Nothing is stored: groups are computed when the page asks (one pass over the library's rows;
the visual vectors come from the search index already in memory).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from itertools import pairwise

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import Image, ImageProperty
from .db.util import chunked
from .faces.ages import reliability

log = logging.getLogger(__name__)

# photos of one event: trustworthy dates at most this far apart (in the same folder)
EVENT_GAP = timedelta(hours=4)
# two trustworthy dates further apart than this are different moments: never linked by look
FAR_APART = timedelta(days=2)
# a photo whose date is further than this from its group's date is shown as off
OFF_BY = timedelta(days=1)
# cosine of the visual vectors above which two neighbouring photos show the same scene
LOOK_ALIKE = 0.82
# file numbers at most this far apart are consecutive shots
NUMBER_STEP = 3
# reliability (faces.ages.DATE_RELIABILITY) from which a date counts as trustworthy
TRUSTED = 0.6
# the earliest year a date can be set to
MIN_YEAR = 1800

_NUMBERED = re.compile(r"^(.*?)(\d+)(\D*)$")

VectorLookup = Callable[[int], "np.ndarray | None"]


@dataclass
class Photo:
    id: int
    folder: str
    filename: str
    recorded_at: datetime | None
    date_source: str
    camera: str | None
    phash: str

    @property
    def trusted(self) -> bool:
        return self.recorded_at is not None and reliability(self.date_source) >= TRUSTED


@dataclass
class DateGroup:
    photos: list[Photo]  # by date (undated last), then file name
    folders: list[str]
    start: datetime | None  # date range of all photos
    end: datetime | None
    date: datetime | None  # the group's date: median of the trustworthy dates
    suggestions: dict[int, datetime] = field(default_factory=dict)  # photo id -> better date
    off: set[int] = field(default_factory=set)  # photos whose date is off (or missing)

    @property
    def size(self) -> int:
        return len(self.photos)

    @property
    def status(self) -> str:
        """``mixed``: some dates don't fit the group; ``unreliable``: no trustworthy date at
        all (only file times); ``ok``: the dates agree."""
        if self.date is None:
            return "unreliable"
        return "mixed" if self.off else "ok"

    @property
    def needs_attention(self) -> bool:
        return self.status != "ok"

    @property
    def key(self) -> int:
        """Stable id within one computation (the smallest photo id), for anchors."""
        return min(p.id for p in self.photos)


class _Links:
    """Union-find over photos that also keeps each group's range of trustworthy dates."""

    def __init__(self, photos: Iterable[Photo]) -> None:
        self.parent: dict[int, int] = {}
        self.span: dict[int, tuple[datetime, datetime] | None] = {}
        for p in photos:
            self.parent[p.id] = p.id
            self.span[p.id] = (p.recorded_at, p.recorded_at) if p.trusted else None  # type: ignore[assignment]

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def _merged(self, ra: int, rb: int) -> tuple[datetime, datetime] | None:
        a, b = self.span[ra], self.span[rb]
        if a is None or b is None:
            return a or b
        return (min(a[0], b[0]), max(a[1], b[1]))

    def union(self, a: int, b: int, *, keep_together: bool = False) -> None:
        """Link the groups of ``a`` and ``b``. With ``keep_together`` only if their trustworthy
        dates stay within :data:`FAR_APART`: a photo with a file time must not bridge two
        different moments."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        span = self._merged(ra, rb)
        if keep_together and span is not None and span[1] - span[0] > FAR_APART:
            return
        root, child = min(ra, rb), max(ra, rb)
        self.parent[child] = root
        self.span[root] = span
        del self.span[child]


def _name_key(filename: str) -> tuple[str, int, str]:
    """Natural sort key: ``IMG_9.jpg`` before ``IMG_10.jpg``."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    m = _NUMBERED.match(stem)
    if not m:
        return (stem.casefold(), -1, "")
    return (m.group(1).casefold(), int(m.group(2)), m.group(3).casefold())


def _consecutive(a: Photo, b: Photo) -> bool:
    ka, kb = _name_key(a.filename), _name_key(b.filename)
    if ka[1] < 0 or kb[1] < 0 or ka[0] != kb[0] or ka[2] != kb[2]:
        return False
    if abs(ka[1] - kb[1]) > NUMBER_STEP:
        return False
    return a.camera is None or b.camera is None or a.camera == b.camera


def _look_alike(a: Photo, b: Photo, vectors: VectorLookup | None) -> bool:
    if vectors is None:
        return False
    va, vb = vectors(a.id), vectors(b.id)
    if va is None or vb is None or va.shape != vb.shape:
        return False
    return float(np.dot(va, vb)) >= LOOK_ALIKE


def load_photos(s: Session, image_ids: Iterable[int] | None = None) -> list[Photo]:
    """The rows grouping needs, of all present photos or of ``image_ids``."""
    cols = (
        Image.id,
        Image.folder,
        Image.filename,
        Image.recorded_at,
        Image.date_source,
        Image.camera,
        Image.phash,
    )
    if image_ids is None:
        rows = list(s.execute(select(*cols).where(Image.missing.is_(False))))
    else:
        rows = []
        for part in chunked(sorted(set(image_ids))):
            rows += s.execute(select(*cols).where(Image.id.in_(part), Image.missing.is_(False)))
    return [
        Photo(
            id=r.id,
            folder=r.folder or "",
            filename=r.filename or "",
            recorded_at=r.recorded_at,
            date_source=r.date_source or "",
            camera=r.camera or None,
            phash=r.phash or "",
        )
        for r in rows
    ]


def group_photos(photos: Sequence[Photo], vectors: VectorLookup | None = None) -> list[DateGroup]:
    """Groups of photos that belong together (see the module docstring), largest first.

    Photos that belong to no other photo come back as groups of one.
    """
    links = _Links(photos)
    by_folder: dict[str, list[Photo]] = defaultdict(list)
    for p in photos:
        by_folder[p.folder.casefold()].append(p)

    for members in by_folder.values():
        # one event: trustworthy dates close together
        timed = sorted((p for p in members if p.trusted), key=lambda p: p.recorded_at)
        for a, b in pairwise(timed):
            if b.recorded_at - a.recorded_at <= EVENT_GAP:  # type: ignore[operator]
                links.union(a.id, b.id)
        # neighbours in file name order: consecutive numbers or the same scene
        named = sorted(members, key=lambda p: (_name_key(p.filename), p.id))
        for a, b in pairwise(named):
            if _consecutive(a, b) or _look_alike(a, b, vectors):
                links.union(a.id, b.id, keep_together=True)

    # copies, in any folder
    by_hash: dict[str, list[int]] = defaultdict(list)
    for p in photos:
        if p.phash and p.phash.strip("0"):  # a blank image hashes to zeros
            by_hash[p.phash].append(p.id)
    for ids in by_hash.values():
        for other in ids[1:]:
            links.union(ids[0], other, keep_together=True)

    members_of: dict[int, list[Photo]] = defaultdict(list)
    for p in photos:
        members_of[links.find(p.id)].append(p)
    groups = [_summarize(m) for m in members_of.values()]
    groups.sort(key=lambda g: (not g.needs_attention, -g.size, g.start or datetime.max, g.key))
    return groups


def _summarize(members: list[Photo]) -> DateGroup:
    members.sort(
        key=lambda p: (p.recorded_at is None, p.recorded_at or datetime.min, _name_key(p.filename))
    )
    dated = [p.recorded_at for p in members if p.recorded_at is not None]
    trusted = [p for p in members if p.trusted]
    folders = sorted({p.folder for p in members}, key=str.casefold)
    group = DateGroup(
        photos=members,
        folders=folders,
        start=min(dated, default=None),
        end=max(dated, default=None),
        date=None,
    )
    if not trusted:
        return group
    stamps = sorted(p.recorded_at for p in trusted)  # type: ignore[type-var]
    group.date = stamps[(len(stamps) - 1) // 2]  # the lower median: a real photo's date
    # the date of a photo that doesn't fit: its nearest trustworthy neighbour in file name
    # order (in its folder), else the group's date
    named = sorted(members, key=lambda p: (p.folder.casefold(), _name_key(p.filename), p.id))
    for i, p in enumerate(named):
        if p.recorded_at is not None and abs(p.recorded_at - group.date) <= OFF_BY:
            continue
        if p.trusted and p.date_source == "manual":
            continue  # set by hand: never contradicted
        group.off.add(p.id)
        group.suggestions[p.id] = _nearest_trusted(named, i, group) or group.date
    return group


def _nearest_trusted(named: list[Photo], i: int, group: DateGroup) -> datetime | None:
    folder = named[i].folder.casefold()
    for step in range(1, len(named)):
        for j in (i - step, i + step):
            if 0 <= j < len(named):
                q = named[j]
                if (
                    q.folder.casefold() == folder
                    and q.trusted
                    and q.id not in group.off
                    and abs(q.recorded_at - group.date) <= OFF_BY  # type: ignore[operator]
                ):
                    return q.recorded_at
        if i - step < 0 and i + step >= len(named):
            break
    return None


# --------------------------------------------------------------------------- correcting


def set_photo_date(s: Session, img: Image, when: datetime) -> None:
    """Set one photo's date by hand: kept on re-index, trusted by ages and exports."""
    if not MIN_YEAR <= when.year <= date.today().year + 1:
        raise ValueError(f"implausible year {when.year}")
    img.recorded_at = when.replace(microsecond=0, tzinfo=None)
    img.date_source = "manual"
    year = str(when.year)
    props = list(
        s.scalars(
            select(ImageProperty).where(
                ImageProperty.image_id == img.id, ImageProperty.key == "year"
            )
        )
    )
    for prop in props:
        prop.value = year
    if not props:  # the photo had no date before
        s.add(ImageProperty(image_id=img.id, key="year", value=year, source="exif"))


def planned_dates(
    photos: Sequence[Photo | Image],
    day: date,
    at: time | None = None,
    *,
    shift: bool = False,
) -> dict[int, datetime]:
    """The new date of each photo.

    ``shift=False``: every photo gets ``day``; at ``at``, or at its own time of day (noon if it
    had no date), so the order within the day stays. ``shift=True``: the earliest photo gets
    ``day`` (at ``at`` or its own time) and the others move by the same amount, so the time
    between them stays (a camera clock that was set wrong).
    """
    if not photos:
        return {}
    noon = time(12, 0)
    if not shift:
        return {
            p.id: datetime.combine(day, at or (p.recorded_at.time() if p.recorded_at else noon))
            for p in photos
        }
    dated = [p for p in photos if p.recorded_at is not None]
    if not dated:
        return planned_dates(photos, day, at)
    first = min(dated, key=lambda p: p.recorded_at)  # type: ignore[arg-type,return-value]
    target = datetime.combine(day, at or first.recorded_at.time())  # type: ignore[union-attr]
    offset = target - first.recorded_at  # type: ignore[operator]
    out = {p.id: p.recorded_at + offset for p in dated}  # type: ignore[operator]
    for p in photos:
        out.setdefault(p.id, target)
    return out


def apply_dates(s: Session, new_dates: dict[int, datetime]) -> int:
    """Set the dates of several photos by hand (inside a write session); returns how many."""
    from .pipeline.control import mark_pending

    for when in new_dates.values():  # check all first: nothing changes on a bad date
        if not MIN_YEAR <= when.year <= date.today().year + 1:
            raise ValueError(f"implausible year {when.year}")
    changed: list[int] = []
    for part in chunked(sorted(new_dates)):
        for img in s.scalars(select(Image).where(Image.id.in_(part))):
            set_photo_date(s, img, new_dates[img.id])
            changed.append(img.id)
    mark_pending(s, "embed", changed)  # the search document contains the date
    return len(changed)
