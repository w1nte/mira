"""Remove photos whose file is missing from the library (Database → Missing files).

Only photos flagged as missing (``images.missing``, see ``Indexer.check_locations``) can be
removed this way, and each file is looked for once more right before its row is deleted: a photo
whose drive is connected again (or that has a copy that still exists) is skipped, never removed.

What goes with a photo: its faces, animals, properties, vision-model result, search index
entries (text and visual), duplicate records and pipeline status (``ON DELETE CASCADE``), and
its cached thumbnails and crops. What it leaves behind is cleaned up: face clusters and unnamed
animal groups without photos are deleted, the others get new counts, averages and covers; places
without photos are deleted unless the user named them. Named persons and named pets are never
deleted: they keep their names and reference faces and just have fewer photos (reference faces
cut from a removed photo keep their embedding, only the picture is gone).

Rows are deleted in chunks, each in a short write transaction; the disk is checked between
transactions. Removing photos is refused while the pipeline runs or is paused (like an import):
its clustering and place grouping work on a snapshot of all faces and photos and would bring
back clusters of faces that no longer exist. ``check`` is called before every chunk so a
pipeline started meanwhile (e.g. from the command line) stops the removal.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, exists, func, select, update

from .db.models import Animal, DuplicateFile, Face, Image, PersonReference, Place
from .db.util import chunked

if TYPE_CHECKING:
    from .config import Settings
    from .db.session import Database

log = logging.getLogger(__name__)

#: photos per write transaction
CHUNK = 200
#: clusters / pets whose statistics are recomputed per write transaction
STATS_CHUNK = 50

_LEADING_ID = re.compile(r"\d+")


@dataclass
class RemovalResult:
    removed: int = 0
    found: int = 0  # skipped: the file (or a copy of it) is there again
    unknown: int = 0  # skipped: no such photo, or it is not marked as missing
    clusters_removed: int = 0
    pets_removed: int = 0
    places_removed: int = 0
    stopped: bool = False
    removed_ids: list[int] = field(default_factory=list)
    new_library_id: bool = False  # ids may be used again: browsers must drop cached pictures


def missing_ids(db: Database) -> list[int]:
    with db.session() as s:
        return list(s.scalars(select(Image.id).where(Image.missing.is_(True)).order_by(Image.id)))


def _max_ids(db: Database) -> tuple[int, int, int]:
    with db.session() as s:
        return (
            s.scalar(select(func.max(Image.id))) or 0,
            s.scalar(select(func.max(Face.id))) or 0,
            s.scalar(select(func.max(Animal.id))) or 0,
        )


def _still_missing(db: Database, ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Of ``ids``: (gone, back, copies): the files that are still not there, the ones that
    are there again, and photos with a duplicate file that exists (``check_locations`` makes
    that copy the photo's file). The disk is checked outside any transaction."""
    from .indexer import _ExistsCache

    with db.session() as s:
        rows = list(
            s.execute(
                select(Image.id, Image.path, Image.root).where(
                    Image.id.in_(ids), Image.missing.is_(True)
                )
            ).all()
        )
        dups: dict[int, list[str]] = {}
        for image_id, path in s.execute(
            select(DuplicateFile.image_id, DuplicateFile.path).where(
                DuplicateFile.image_id.in_(ids)
            )
        ):
            dups.setdefault(image_id, []).append(path)
    found = _ExistsCache()  # one look per folder: an unplugged drive would make each wait
    gone, back, copies = [], [], []
    for image_id, path, root in rows:
        if found(path, root):
            back.append(image_id)
        elif any(found(p) for p in dups.get(image_id, [])):
            copies.append(image_id)
        else:
            gone.append(image_id)
    return gone, back, copies


@dataclass
class _Touched:
    """What the removed photos leave behind, for the clean-up."""

    clusters: set[int] = field(default_factory=set)
    pets: set[int] = field(default_factory=set)
    places: set[int] = field(default_factory=set)
    faces: list[int] = field(default_factory=list)
    animals: list[int] = field(default_factory=list)


def remove_missing(
    db: Database,
    settings: Settings,
    image_ids: Iterable[int] | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    check: Callable[[], None] | None = None,
) -> RemovalResult:
    """Delete the photos ``image_ids`` (default: every missing photo) whose file is missing.

    Photos that are not marked as missing are left alone (``unknown``). Photos whose file
    exists again are skipped and marked as found; photos with a copy that exists are skipped
    too (the next location check makes the copy their file): both count as ``found``.
    ``check`` may raise to stop before the next chunk; what was removed until then is cleaned
    up all the same.
    """
    result = RemovalResult()
    ids = missing_ids(db) if image_ids is None else sorted({int(i) for i in image_ids})
    before = _max_ids(db)
    touched = _Touched()
    try:
        done = 0
        for part in chunked(ids, CHUNK):
            if should_stop is not None and should_stop():
                result.stopped = True
                break
            if check is not None:
                check()
            _remove_chunk(db, part, result, touched)
            done += len(part)
            if progress is not None:
                progress(done, len(ids))
    finally:
        _clean_up(db, result, touched)
        _remove_cached(settings, result.removed_ids, touched.faces, touched.animals)
        if result.removed:
            from .faces import ages

            ages._CALIBRATION_CACHE.clear()
            # SQLite gives a new row the highest id + 1: removing the newest photo (face,
            # animal) lets a new one get its id, and browsers cache pictures by id for a day
            if any(after < old for after, old in zip(_max_ids(db), before, strict=True)):
                from .db import transfer

                transfer.new_library_id(db)
                result.new_library_id = True
        log.info(
            "removed %d missing photos (%d found again, %d unknown); %d clusters, %d pets and "
            "%d places without photos removed",
            result.removed,
            result.found,
            result.unknown,
            result.clusters_removed,
            result.pets_removed,
            result.places_removed,
        )
    return result


def _remove_chunk(db: Database, part: list[int], result: RemovalResult, touched: _Touched) -> None:
    """Check the files of ``part`` once more, then delete the photos that are still missing
    in one short write transaction."""
    gone, back, copies = _still_missing(db, part)
    result.found += len(back) + len(copies)
    result.unknown += len(part) - len(gone) - len(back) - len(copies)
    with db.write() as s:
        # the flag is checked again: a scan may have found a photo since
        now = list(s.scalars(select(Image.id).where(Image.id.in_(gone), Image.missing.is_(True))))
        result.unknown += len(gone) - len(now)
        faces = s.execute(select(Face.id, Face.cluster_id).where(Face.image_id.in_(now))).all()
        animals = s.execute(select(Animal.id, Animal.pet_id).where(Animal.image_id.in_(now))).all()
        places = s.scalars(select(Image.place_id).where(Image.id.in_(now)).distinct()).all()
        face_ids = [f[0] for f in faces]
        for ref_part in chunked(face_ids):
            # the reference keeps its embedding; its face id could be given to a new face
            s.execute(
                update(PersonReference)
                .where(PersonReference.face_id.in_(ref_part))
                .values(face_id=None)
            )
        if now:
            s.execute(delete(Image).where(Image.id.in_(now)))  # cascades to faces etc.
        if back:  # like the location check does
            s.execute(update(Image).where(Image.id.in_(back)).values(missing=False))
    touched.clusters.update(c for _f, c in faces if c is not None)
    touched.pets.update(p for _a, p in animals if p is not None)
    touched.places.update(p for p in places if p is not None)
    touched.faces.extend(face_ids)
    touched.animals.extend(a[0] for a in animals)
    result.removed += len(now)
    result.removed_ids.extend(now)


def _clean_up(db: Database, result: RemovalResult, touched: _Touched) -> None:
    """New counts, averages and covers for the clusters and pets that lost photos; the empty
    ones go (named pets stay), and so do places without photos that the user did not name."""
    from .animals.grouping import update_pet_stats
    from .faces.clustering import update_cluster_stats

    for part in chunked(sorted(touched.clusters), STATS_CHUNK):
        with db.write() as s:
            result.clusters_removed += update_cluster_stats(s, part)
    for part in chunked(sorted(touched.pets), STATS_CHUNK):
        with db.write() as s:
            result.pets_removed += update_pet_stats(s, part)
    for part in chunked(sorted(touched.places)):
        with db.write() as s:
            res = s.execute(
                delete(Place).where(
                    Place.id.in_(part),
                    Place.user_named.is_(False),
                    ~exists(select(Image.id).where(Image.place_id == Place.id)),
                )
            )
            result.places_removed += res.rowcount or 0


def _remove_files(base: Path, ids: Iterable[int]) -> int:
    """Delete the cached files of ``ids`` below ``base`` (``<id // 1000>/<id>…``: every size
    and version). Each folder is listed once; locked files (Windows) are left behind."""
    by_shard: dict[str, set[int]] = {}
    for item_id in ids:
        by_shard.setdefault(f"{item_id // 1000:04d}", set()).add(item_id)
    removed = 0
    for shard, wanted in by_shard.items():
        try:
            entries = list(os.scandir(base / shard))
        except OSError:
            continue
        for entry in entries:
            match = _LEADING_ID.match(entry.name)
            if match is None or int(match.group()) not in wanted:
                continue
            try:
                Path(entry.path).unlink()
                removed += 1
            except OSError:
                log.debug("could not delete %s", entry.path)
    return removed


def _remove_cached(
    settings: Settings, image_ids: list[int], face_ids: list[int], animal_ids: list[int]
) -> None:
    """Thumbnails of the removed photos, crops and aligned portraits of their faces and crops
    of their animals: their ids may be given to new photos."""
    _remove_files(settings.thumbs_dir, image_ids)
    _remove_files(settings.face_crops_dir, face_ids)
    _remove_files(settings.cache_dir / "aligned", face_ids)
    _remove_files(settings.animal_crops_dir, animal_ids)
