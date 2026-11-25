"""Step 0: build the image index.

Walks the image roots recursively, extracts metadata (recording date, EXIF, GPS, ...),
computes a perceptual hash and stores one row per *unique* pHash. Files whose pHash is
already known are recorded in ``duplicate_files`` instead of creating a second image.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imagehash
from sqlalchemy import delete, select, update

from .config import Settings
from .db.models import (
    DuplicateFile,
    Image,
    ImageProperty,
    StageStatus,
    Status,
    VlmResult,
    utcnow,
)
from .db.session import Database
from .derived import metadata_properties
from .faces.clustering import delete_image_faces
from .imaging import read_image_info

log = logging.getLogger(__name__)


def path_key(path: str | Path) -> str:
    """Normalised key used for uniqueness (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


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
    info: Any = None
    error: str | None = None


def iter_image_files(
    root: Path,
    extensions: Iterable[str],
    *,
    exclude: Iterable[Path] = (),
    skip_hidden: bool = True,
    follow_symlinks: bool = False,
) -> Iterator[tuple[str, os.stat_result]]:
    """Yield ``(path, stat)`` for every image file below ``root`` (iteratively, no recursion)."""
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    excluded = {path_key(p) for p in exclude}
    stack = [os.fspath(root)]
    while stack:
        current = stack.pop()
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
        root_paths = [Path(r).expanduser() for r in (roots or self.settings.paths.image_roots)]
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
            for start in range(0, len(candidates), chunk):
                if self.should_stop():
                    log.info("indexing stopped by request")
                    break
                results = list(pool.map(self._analyse, candidates[start : start + chunk]))
                self._store(results, stats)
                stats.processed += len(results)
                self.progress(stats)
            else:
                if mark_missing:
                    stats.missing = self._mark_missing(stats.roots)

        stats.done = True
        self.progress(stats)
        return stats

    # ------------------------------------------------------------------ internals
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

    @staticmethod
    def _analyse(cand: _Candidate) -> _Result:
        try:
            info = read_image_info(cand.path, cand.mtime)
            assert info.image is not None
            ph = str(imagehash.phash(info.image))
            info.image = None  # free memory early
            return _Result(cand, phash=ph, info=info)
        except Exception as exc:
            return _Result(cand, error=f"{type(exc).__name__}: {exc}")

    def _store(self, results: list[_Result], stats: IndexStats) -> None:
        ok = [r for r in results if r.error is None]
        for r in results:
            if r.error is not None:
                stats.errors += 1
                if len(stats.error_samples) < 50:
                    stats.error_samples.append(f"{r.cand.path}: {r.error}")
                log.warning("index error %s: %s", r.cand.path, r.error)
        if not ok:
            return

        with self.db.write() as s:
            hashes = {r.phash for r in ok}
            keys = {r.cand.key for r in ok}
            by_hash: dict[str, Image] = {
                img.phash: img for img in s.scalars(select(Image).where(Image.phash.in_(hashes)))
            }
            by_key: dict[str, Image] = {
                img.path_key: img
                for img in s.scalars(select(Image).where(Image.path_key.in_(keys)))
            }
            dup_by_key: dict[str, DuplicateFile] = {
                d.path_key: d
                for d in s.scalars(select(DuplicateFile).where(DuplicateFile.path_key.in_(keys)))
            }

            for r in ok:
                c, info = r.cand, r.info
                assert r.phash is not None
                existing_at_path = by_key.get(c.key)
                if existing_at_path is not None:
                    # Same file path, but content / timestamp changed.
                    self._update_existing(s, existing_at_path, r, by_hash)
                    stats.updated += 1
                    continue

                owner = by_hash.get(r.phash)
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
                img = Image(phash=r.phash, missing=False)
                self._apply_location(img, c)
                self._apply_info(img, info)
                img.stages = [
                    StageStatus(stage=name, status=Status.PENDING) for name in self.stages
                ]
                img.properties = _metadata_rows(img)
                s.add(img)
                s.flush()
                by_hash[r.phash] = img
                by_key[c.key] = img
                stats.new += 1

    def _update_existing(self, s: Any, img: Image, r: _Result, by_hash: dict[str, Image]) -> None:
        assert r.phash is not None
        self._apply_info(img, r.info)
        img.size_bytes = r.cand.size
        img.mtime = r.cand.mtime
        img.missing = False
        content_changed = r.phash != img.phash
        if content_changed and r.phash in by_hash and by_hash[r.phash].id != img.id:
            # New content equals another indexed image; keep the row but leave its hash.
            log.info("%s now duplicates image %s", r.cand.path, by_hash[r.phash].id)
            content_changed = False
        if content_changed:
            by_hash.pop(img.phash, None)
            img.phash = r.phash
            by_hash[r.phash] = img
            # Results are stale -> drop them (faces with consistent cluster stats) and re-queue.
            delete_image_faces(s, img.id, self.settings.face_crops_dir)
            s.execute(
                delete(ImageProperty).where(
                    ImageProperty.image_id == img.id, ImageProperty.source != "exif"
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
        rel = os.path.relpath(c.path, c.root)
        img.path = c.path
        img.path_key = c.key
        img.root = c.root
        img.rel_path = rel
        img.folder = os.path.dirname(rel)
        img.filename = os.path.basename(c.path)
        img.ext = os.path.splitext(c.path)[1].lower()
        img.size_bytes = c.size
        img.mtime = c.mtime
        img.indexed_at = utcnow()

    @staticmethod
    def _apply_info(img: Image, info: Any) -> None:
        img.width = info.width
        img.height = info.height
        img.recorded_at = info.recorded_at
        img.date_source = info.date_source
        img.camera = info.camera
        img.meta = info.meta

    def _mark_missing(self, roots: list[str]) -> int:
        missing = 0
        with self.db.write() as s:
            for img in s.scalars(select(Image).where(Image.root.in_(roots))):
                exists = os.path.exists(img.path)
                if img.missing == exists:  # state changed
                    img.missing = not exists
                if not exists:
                    missing += 1
        return missing


def _metadata_rows(img: Image) -> list[ImageProperty]:
    return [
        ImageProperty(key=d.key, value=d.value, confidence=d.confidence, source="exif")
        for d in metadata_properties(img)
    ]


def ensure_stage_rows(db: Database, stages: Iterable[str]) -> int:
    """Create missing ``stage_status`` rows (e.g. after a new stage was enabled)."""
    created = 0
    with db.write() as s:
        image_ids = list(s.scalars(select(Image.id)))
        for stage in stages:
            have = set(s.scalars(select(StageStatus.image_id).where(StageStatus.stage == stage)))
            for image_id in image_ids:
                if image_id not in have:
                    s.add(StageStatus(image_id=image_id, stage=stage, status=Status.PENDING))
                    created += 1
    return created
