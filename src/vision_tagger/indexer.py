"""Step 0: build the image index.

Walks the image roots recursively, extracts metadata (recording date, EXIF, GPS, ...),
computes a perceptual hash and stores one row per *unique* pHash. Files whose pHash is
already known are recorded in ``duplicate_files`` instead of creating a second image.
"""

from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imagehash
from sqlalchemy import delete, exists, insert, literal, select, update

from .config import Settings
from .db.models import (
    AppSetting,
    DuplicateFile,
    Image,
    ImageProperty,
    StageStatus,
    Status,
    VlmResult,
    utcnow,
)
from .db.session import Database
from .db.util import chunked
from .derived import metadata_properties
from .faces.clustering import delete_image_faces
from .imaging import (
    encode_thumbnail,
    ensure_thumbnail,
    gps_position,
    read_image_info,
    save_thumbnail_bytes,
    thumbnail_path,
)

log = logging.getLogger(__name__)


def path_key(path: str | Path) -> str:
    """Normalised key used for uniqueness (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


ROOTS_KEY = "image_roots"


def stored_roots(db: Database, settings: Settings) -> list[str]:
    """Image roots last used in the web UI, falling back to ``paths.image_roots``.

    In remote mode the library is the upload folder.
    """
    if settings.remote_mode:
        return [str(settings.upload_dir)]
    with db.session() as s:
        stored = s.get(AppSetting, ROOTS_KEY)
        if stored is not None and stored.value is not None:
            return [str(r) for r in stored.value]  # an emptied list stays empty
    return [str(p) for p in settings.paths.image_roots]


def hash_distance(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return (int(a, 16) ^ int(b, 16)).bit_count()


def find_duplicate(
    candidates: Iterable[Image],
    phash: str,
    dhash: str,
    max_dist: int,
    exclude_id: int | None = None,
) -> Image | None:
    """The indexed image this content duplicates: same pHash and dHash within ``max_dist``."""
    for img in candidates:
        if img.id is not None and img.id == exclude_id:
            continue
        if img.phash == phash and hash_distance(img.dhash, dhash) <= max_dist:
            return img
    return None


@dataclass
class IndexStats:
    roots: list[str] = field(default_factory=list)
    found: int = 0
    candidates: int = 0
    processed: int = 0
    new: int = 0
    duplicates: int = 0
    moved: int = 0
    updated: int = 0
    unchanged: int = 0
    errors: int = 0
    missing: int = 0
    thumbnails: int = 0  # thumbnails written
    thumbnails_todo: int = 0  # missing thumbnails of files indexed earlier
    thumbnails_warmed: int = 0  # ... of which created so far
    places: str = ""  # summary of finding places (geo.places) after the scan
    error_samples: list[str] = field(default_factory=list)
    done: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Candidate:
    root: str
    path: str
    key: str
    size: int
    mtime: float


@dataclass
class _Result:
    cand: _Candidate
    phash: str | None = None
    dhash: str = ""
    info: Any = None
    error: str | None = None
    thumb: bytes | None = None  # encoded gallery thumbnail (written once the id is known)


def _is_junction(entry: os.DirEntry[str]) -> bool:
    fn = getattr(entry, "is_junction", None)  # Python >= 3.12
    try:
        return bool(fn()) if fn is not None else False
    except OSError:
        return False


def iter_image_files(
    root: Path,
    extensions: Iterable[str],
    *,
    exclude: Iterable[Path] = (),
    skip_hidden: bool = True,
    follow_symlinks: bool = False,
    skip_dirs: Iterable[str] = (),
) -> Iterator[tuple[str, os.stat_result]]:
    """Yield ``(path, stat)`` for every image file below ``root`` (iteratively, no recursion).

    Directory junctions are only followed together with symlinks, and every real directory is
    visited at most once, so link loops cannot make the scan run forever.
    """
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    excluded = {path_key(p) for p in exclude}
    skipped = {d.casefold() for d in skip_dirs}
    visited: set[str] = set()
    stack = [os.fspath(root)]
    while stack:
        current = stack.pop()
        real = os.path.normcase(os.path.realpath(current))
        if real in visited:
            continue
        visited.add(real)
        try:
            with os.scandir(current) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            log.warning("cannot scan %s: %s", current, exc)
            continue
        subdirs: list[str] = []
        for entry in entries:
            if skip_hidden and entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=follow_symlinks):
                    if entry.name.casefold() in skipped:
                        continue
                    if not follow_symlinks and _is_junction(entry):
                        continue
                    if path_key(entry.path) not in excluded:
                        subdirs.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=follow_symlinks):
                    continue
                if os.path.splitext(entry.name)[1].lower() not in exts:
                    continue
                yield entry.path, entry.stat(follow_symlinks=follow_symlinks)
            except OSError as exc:
                log.warning("cannot stat %s: %s", entry.path, exc)
        stack.extend(reversed(subdirs))


class Indexer:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        *,
        stages: list[str] | None = None,
        progress: Callable[[IndexStats], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.stages = stages if stages is not None else list(settings.pipeline.stages)
        self.progress = progress or (lambda _s: None)
        self.should_stop = should_stop or (lambda: False)

    # ------------------------------------------------------------------ public
    def run(
        self, roots: Iterable[str | Path] | None = None, *, mark_missing: bool = True
    ) -> IndexStats:
        if roots is None:
            roots = stored_roots(self.db, self.settings)
        root_paths = [Path(r).expanduser() for r in roots]
        stats = IndexStats(roots=[os.path.abspath(r) for r in root_paths])
        if not root_paths:
            stats.done = True
            return stats

        known = self._known_files()
        candidates: list[_Candidate] = []
        for root in root_paths:
            if not root.is_dir():
                stats.errors += 1
                stats.error_samples.append(f"not a directory: {root}")
                continue
            root_abs = os.path.abspath(root)
            for path, st in iter_image_files(
                root,
                self.settings.index.extensions,
                exclude=[self.settings.data_dir],
                skip_hidden=self.settings.index.skip_hidden,
                follow_symlinks=self.settings.index.follow_symlinks,
                skip_dirs=self.settings.index.skip_dirs,
            ):
                stats.found += 1
                key = path_key(path)
                prev = known.get(key)
                if prev is not None and prev == (st.st_size, round(st.st_mtime, 3)):
                    stats.unchanged += 1
                    continue
                candidates.append(
                    _Candidate(root_abs, os.path.abspath(path), key, st.st_size, st.st_mtime)
                )
        stats.candidates = len(candidates)
        self.progress(stats)

        chunk = max(1, self.settings.index.commit_every)
        with ThreadPoolExecutor(max_workers=max(1, self.settings.index.workers)) as pool:
            # The workers analyse the next chunk while the previous one is stored, so they do
            # not wait for the database. Chunks are stored in order, as before.
            analysing: list[Future[_Result]] = []
            stopped = False
            for start in range(0, len(candidates), chunk):
                if self.should_stop():
                    stopped = True
                    break
                submitted = [
                    pool.submit(self._analyse, c) for c in candidates[start : start + chunk]
                ]
                self._store_chunk(analysing, stats)
                analysing = submitted
                if self.should_stop():
                    stopped = True
                    break
            if stopped:  # like before: nothing after the stop request is stored
                log.info("indexing stopped by request")
                for future in analysing:
                    future.cancel()
            else:
                self._store_chunk(analysing, stats)
                if mark_missing:
                    stats.missing = self._mark_missing(stats.roots)
                if self.settings.geo.auto:
                    self._find_places(stats)
                if self.settings.index.thumbnails:
                    self._warm_thumbnails(pool, stats)

        stats.done = True
        self.progress(stats)
        return stats

    # ------------------------------------------------------------------ internals
    def _store_chunk(self, futures: list[Future[_Result]], stats: IndexStats) -> None:
        if not futures:
            return
        results = [f.result() for f in futures]
        thumbs = self._store(results, stats)
        # written after the transaction: no file I/O while holding the write lock
        for image_id, data in thumbs:
            try:
                save_thumbnail_bytes(
                    self.settings.thumbs_dir, image_id, self.settings.web.thumb_size, data
                )
                stats.thumbnails += 1
            except OSError as exc:  # a cache: the gallery makes it on demand instead
                log.debug("cannot write thumbnail %s: %s", image_id, exc)
        stats.processed += len(results)
        self.progress(stats)

    def _find_places(self, stats: IndexStats) -> None:
        """Group new GPS positions into named places (a convenience: never fails the scan)."""
        from .geo.places import find_places

        try:
            stats.places = find_places(self.db, self.settings, should_stop=self.should_stop).message
        except Exception as exc:
            log.warning("finding places failed: %s", exc)
            stats.places = f"{type(exc).__name__}: {exc}"
        self.progress(stats)

    def _warm_thumbnails(self, pool: ThreadPoolExecutor, stats: IndexStats) -> None:
        """Create the gallery thumbnails that are still missing (files indexed earlier)."""
        size = self.settings.web.thumb_size
        thumbs_dir = self.settings.thumbs_dir
        with self.db.session() as s:
            rows = s.execute(
                select(Image.id, Image.path).where(Image.missing.is_(False)).order_by(Image.id)
            ).all()
        todo = [(i, p) for i, p in rows if not thumbnail_path(thumbs_dir, i, size).exists()]
        stats.thumbnails_todo = len(todo)
        if not todo:
            return
        log.info("creating %d missing thumbnails", len(todo))
        self.progress(stats)

        def make(item: tuple[int, str]) -> bool:
            try:
                ensure_thumbnail(thumbs_dir, item[0], item[1], size)
                return True
            except Exception as exc:  # unreadable file: the gallery shows a placeholder
                log.debug("thumbnail for %s failed: %s", item[1], exc)
                return False

        step = 64
        for start in range(0, len(todo), step):
            if self.should_stop():
                log.info("thumbnails stopped by request")
                break
            made = sum(pool.map(make, todo[start : start + step]))
            stats.thumbnails += made
            stats.thumbnails_warmed += made
            self.progress(stats)

    def _known_files(self) -> dict[str, tuple[int, float]]:
        known: dict[str, tuple[int, float]] = {}
        with self.db.session() as s:
            for key, size, mtime in s.execute(
                select(Image.path_key, Image.size_bytes, Image.mtime)
            ):
                known[key] = (size, round(mtime, 3))
            for key, size, mtime in s.execute(
                select(DuplicateFile.path_key, DuplicateFile.size_bytes, DuplicateFile.mtime)
            ):
                known[key] = (size, round(mtime, 3))
        return known

    def _analyse(self, cand: _Candidate) -> _Result:
        try:
            info = read_image_info(cand.path, cand.mtime)
            assert info.image is not None
            ph = str(imagehash.phash(info.image))
            dh = str(imagehash.dhash(info.image))
            thumb = self._thumbnail_from(info)
            info.image = None  # free memory early
            return _Result(cand, phash=ph, dhash=dh, info=info, thumb=thumb)
        except Exception as exc:
            return _Result(cand, error=f"{type(exc).__name__}: {exc}")

    def _thumbnail_from(self, info: Any) -> bytes | None:
        """The gallery thumbnail, from the small image decoded for hashing (if big enough)."""
        if not self.settings.index.thumbnails:
            return None
        size = self.settings.web.thumb_size
        im = info.image
        needed = min(size, max(info.width or 0, info.height or 0))
        if max(im.size) < needed:
            return None  # thumb_size larger than the hashing image: made on demand instead
        try:
            return encode_thumbnail(im, size)
        except OSError:
            return None

    def _store(self, results: list[_Result], stats: IndexStats) -> list[tuple[int, bytes]]:
        """Store a chunk; returns (image id, thumbnail) pairs to write after the commit."""
        thumbs: list[tuple[int, bytes]] = []
        ok = [r for r in results if r.error is None]
        for r in results:
            if r.error is not None:
                stats.errors += 1
                if len(stats.error_samples) < 50:
                    stats.error_samples.append(f"{r.cand.path}: {r.error}")
                log.warning("index error %s: %s", r.cand.path, r.error)
        if not ok:
            return thumbs

        max_dist = self.settings.index.dhash_max_distance
        with self.db.write() as s:
            by_hash: dict[str, list[Image]] = {}
            for part in chunked(sorted({r.phash for r in ok if r.phash})):
                for img in s.scalars(select(Image).where(Image.phash.in_(part))):
                    by_hash.setdefault(img.phash, []).append(img)
            keys = sorted({r.cand.key for r in ok})
            by_key: dict[str, Image] = {}
            dup_by_key: dict[str, DuplicateFile] = {}
            for part in chunked(keys):
                by_key.update(
                    (img.path_key, img)
                    for img in s.scalars(select(Image).where(Image.path_key.in_(part)))
                )
                dup_by_key.update(
                    (d.path_key, d)
                    for d in s.scalars(
                        select(DuplicateFile).where(DuplicateFile.path_key.in_(part))
                    )
                )

            for r in ok:
                c, info = r.cand, r.info
                assert r.phash is not None
                existing_at_path = by_key.get(c.key)
                if existing_at_path is not None:
                    # Same file path, but content / timestamp changed.
                    self._update_existing(s, existing_at_path, r, by_hash)
                    stats.updated += 1
                    if r.thumb:
                        thumbs.append((existing_at_path.id, r.thumb))
                    continue

                owner = find_duplicate(by_hash.get(r.phash, []), r.phash, r.dhash, max_dist)
                if owner is not None:
                    if owner.path_key != c.key and not os.path.exists(owner.path):
                        # The original file moved: re-point the image instead of duplicating.
                        self._apply_location(owner, c)
                        owner.missing = False
                        by_key[c.key] = owner
                        stale = dup_by_key.pop(c.key, None)
                        if stale is not None:
                            s.delete(stale)
                        stats.moved += 1
                        if r.thumb:
                            thumbs.append((owner.id, r.thumb))
                        continue
                    dup = dup_by_key.get(c.key)
                    if dup is None:
                        dup = DuplicateFile(path=c.path, path_key=c.key, image_id=owner.id)
                        s.add(dup)
                        dup_by_key[c.key] = dup
                    dup.image_id = owner.id
                    dup.size_bytes = c.size
                    dup.mtime = c.mtime
                    stats.duplicates += 1
                    continue

                stale_dup = dup_by_key.pop(c.key, None)
                if stale_dup is not None:  # file was a duplicate before, content changed
                    s.delete(stale_dup)
                    s.flush()
                img = Image(phash=r.phash, dhash=r.dhash, missing=False)
                self._apply_location(img, c)
                self._apply_info(img, info)
                img.stages = [
                    StageStatus(stage=name, status=Status.PENDING) for name in self.stages
                ]
                img.properties = _metadata_rows(img)
                s.add(img)
                s.flush()
                by_hash.setdefault(r.phash, []).append(img)
                by_key[c.key] = img
                stats.new += 1
                if r.thumb:
                    thumbs.append((img.id, r.thumb))
        return thumbs

    def _update_existing(
        self, s: Any, img: Image, r: _Result, by_hash: dict[str, list[Image]]
    ) -> None:
        assert r.phash is not None
        self._apply_info(img, r.info)
        img.size_bytes = r.cand.size
        img.mtime = r.cand.mtime
        img.missing = False
        max_dist = self.settings.index.dhash_max_distance
        content_changed = r.phash != img.phash or hash_distance(r.dhash, img.dhash) > max_dist
        other = find_duplicate(
            by_hash.get(r.phash, []), r.phash, r.dhash, max_dist, exclude_id=img.id
        )
        if content_changed and other is not None:
            # New content equals another indexed image; keep the row but leave its hashes.
            log.info("%s now duplicates image %s", r.cand.path, other.id)
            content_changed = False
        if content_changed:
            if img in by_hash.get(img.phash, []):
                by_hash[img.phash].remove(img)
            img.phash, img.dhash = r.phash, r.dhash
            by_hash.setdefault(r.phash, []).append(img)
            # Results are stale -> drop them (faces with consistent cluster stats) and re-queue.
            delete_image_faces(s, img.id, self.settings.face_crops_dir)
            s.execute(
                delete(ImageProperty).where(
                    ImageProperty.image_id == img.id,
                    ImageProperty.source.not_in(("exif", "place")),  # place: from the position
                )
            )
            s.execute(delete(VlmResult).where(VlmResult.image_id == img.id))
            s.execute(
                update(StageStatus)
                .where(StageStatus.image_id == img.id)
                .values(status=Status.PENDING, attempts=0, last_error=None, stage_version=None)
            )
        # Metadata properties are rebuilt last, so no bulk delete can remove them again.
        s.execute(
            delete(ImageProperty).where(
                ImageProperty.image_id == img.id, ImageProperty.source == "exif"
            )
        )
        for prop in _metadata_rows(img):
            prop.image_id = img.id
            s.add(prop)

    @staticmethod
    def _apply_location(img: Image, c: _Candidate) -> None:
        # rel_path / folder always use "/" so folder filters work the same on every OS
        rel = os.path.relpath(c.path, c.root).replace(os.sep, "/")
        img.path = c.path
        img.path_key = c.key
        img.root = c.root
        img.rel_path = rel
        img.folder = posixpath.dirname(rel)
        img.filename = os.path.basename(c.path)
        img.ext = os.path.splitext(c.path)[1].lower()
        img.size_bytes = c.size
        img.mtime = c.mtime
        img.indexed_at = utcnow()

    @staticmethod
    def _apply_info(img: Image, info: Any) -> None:
        img.width = info.width
        img.height = info.height
        if img.date_source != "manual":  # dates corrected by the user win over file metadata
            img.recorded_at = info.recorded_at
            img.date_source = info.date_source
        img.camera = info.camera
        img.meta = info.meta
        lat, lon = gps_position(info.meta) or (None, None)
        if (img.lat, img.lon) != (lat, lon):
            img.lat, img.lon = lat, lon
            img.place_id = None  # placed again by geo.places (a user-named place takes it back)

    def _mark_missing(self, roots: list[str]) -> int:
        """Flag vanished files without holding the write lock while touching the disk.

        If an image's file is gone but one of its duplicates still exists, the image is
        re-pointed to that copy. Duplicate records whose file vanished are removed.
        """
        with self.db.session() as s:
            images = s.execute(
                select(Image.id, Image.path, Image.missing).where(Image.root.in_(roots))
            ).all()
            dups = s.execute(
                select(
                    DuplicateFile.id,
                    DuplicateFile.image_id,
                    DuplicateFile.path,
                    DuplicateFile.path_key,
                    DuplicateFile.size_bytes,
                    DuplicateFile.mtime,
                )
            ).all()
        # --- disk checks, no transaction open
        image_exists = {i: os.path.exists(p) for i, p, _m in images}
        dup_exists = {d[0]: os.path.exists(d[2]) for d in dups}
        dups_by_image: dict[int, list[Any]] = {}
        for d in dups:
            if dup_exists[d[0]]:
                dups_by_image.setdefault(d[1], []).append(d)
        promote: dict[int, Any] = {}
        for image_id, _path, _missing in images:
            if not image_exists[image_id] and dups_by_image.get(image_id):
                promote[image_id] = dups_by_image[image_id][0]
        vanished_dups = [d_id for d_id, ok in dup_exists.items() if not ok]
        flag_changes = [
            (i, not image_exists[i])
            for i, _p, was_missing in images
            if i not in promote and was_missing == image_exists[i]
        ]
        # --- apply in short write transactions
        for part in chunked(vanished_dups):
            with self.db.write() as s:
                s.execute(delete(DuplicateFile).where(DuplicateFile.id.in_(part)))
        for part in chunked(flag_changes, 500):
            with self.db.write() as s:
                for image_id, missing in part:
                    s.execute(update(Image).where(Image.id == image_id).values(missing=missing))
        for part in chunked(list(promote.items()), 200):
            with self.db.write() as s:
                for image_id, d in part:
                    img = s.get(Image, image_id)
                    dup = s.get(DuplicateFile, d[0])
                    if img is None or dup is None:
                        continue
                    root = _root_for(d[2], roots)
                    s.delete(dup)
                    s.flush()
                    self._apply_location(img, _Candidate(root, d[2], d[3], d[4], d[5]))
                    img.missing = False
                    log.info("image %s moved to its duplicate %s", image_id, d[2])
        return sum(1 for i, _p, _m in images if not image_exists[i] and i not in promote)


def _root_for(path: str, roots: list[str]) -> str:
    """The scan root that contains ``path`` (the deepest one), else its directory."""
    key = os.path.normcase(path)
    best = ""
    for root in roots:
        r = os.path.normcase(root)
        if key.startswith(r.rstrip("\\/") + os.sep) and len(r) > len(best):
            best = root
    return best or os.path.dirname(path)


def _metadata_rows(img: Image) -> list[ImageProperty]:
    return [
        ImageProperty(key=d.key, value=d.value, confidence=d.confidence, source="exif")
        for d in metadata_properties(img)
    ]


def ensure_stage_rows(db: Database, stages: Iterable[str]) -> int:
    """Create missing ``stage_status`` rows (e.g. after a new stage was enabled)."""
    created = 0
    with db.write() as s:
        for stage in stages:
            missing = select(Image.id, literal(stage), literal(Status.PENDING), literal(0)).where(
                ~exists(
                    select(StageStatus.image_id).where(
                        StageStatus.image_id == Image.id, StageStatus.stage == stage
                    )
                )
            )
            res = s.execute(
                insert(StageStatus).from_select(
                    ["image_id", "stage", "status", "attempts"], missing
                )
            )
            created += res.rowcount or 0
    return created
