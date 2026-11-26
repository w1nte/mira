"""Resumable uploads for remote mode.

The browser sends each file in chunks (``PUT`` with the byte offset), so an upload continues
where it stopped after a lost connection, a closed tab or a server restart. A finished upload
is moved into the upload folder (``remote.upload_dir``); a ZIP file is unpacked there (images
only). The indexer picks the new files up from that folder.

Safety: names coming from the browser are reduced to harmless relative paths, nothing is
ever written outside the upload folder (no ``..``, no absolute paths, no drive letters, also
not from inside a ZIP), existing files are never overwritten, and ZIP archives are checked
against decompression bombs (``remote.max_zip_gb``, compression ratio).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath

from .config import Settings

log = logging.getLogger(__name__)

_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = (
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
MAX_RATIO = 200  # a ZIP entry may not expand more than this (photos barely compress at all)


class UploadError(ValueError):
    pass


def safe_relative_path(name: str, max_parts: int = 8) -> PurePosixPath:
    """A harmless relative path from a name sent by a browser or found in a ZIP."""
    parts = []
    for raw in re.split(r"[\\/]+", name or ""):
        part = _BAD_CHARS.sub("_", raw).strip().rstrip(".")
        if not part or part in (".", ".."):
            continue
        if part.split(".")[0].lower() in _RESERVED:
            part = "_" + part
        parts.append(part[:120])
    if not parts:
        raise UploadError("empty file name")
    return PurePosixPath(*parts[-max_parts:])


def unique_path(path: Path) -> Path:
    """``path``, or ``name (2).ext`` etc. if it exists already (never overwrite)."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


@dataclass
class Upload:
    id: str
    name: str  # relative path as sent (sanitised)
    size: int
    received: int = 0
    state: str = "uploading"  # uploading | done | failed
    created: float = 0.0
    files: int = 0  # photos placed in the library when done
    error: str = ""


class UploadStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tmp = settings.uploads_tmp_dir
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.target = settings.upload_dir
        self.extensions = {e.lower() for e in settings.index.extensions}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ bookkeeping
    def _lock(self, upload_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(upload_id, threading.Lock())

    def _meta_path(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise UploadError("unknown upload")
        return self.tmp / f"{upload_id}.json"

    def _part(self, upload_id: str) -> Path:
        return self.tmp / f"{upload_id}.part"

    def _save(self, up: Upload) -> None:
        path = self._meta_path(up.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(up)), encoding="utf-8")
        tmp.replace(path)

    def get(self, upload_id: str) -> Upload:
        path = self._meta_path(upload_id)
        try:
            return Upload(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            raise UploadError("unknown upload") from exc

    # ------------------------------------------------------------------ protocol
    def create(self, name: str, size: int) -> Upload:
        rel = safe_relative_path(name)
        suffix = rel.suffix.lower()
        if suffix != ".zip" and suffix not in self.extensions:
            raise UploadError(f"not a photo or ZIP file: {rel.name}")
        if size < 0 or size > self.settings.remote.max_file_gb * 1024**3:
            raise UploadError(f"file too large: {rel.name}")
        up = Upload(id=uuid.uuid4().hex, name=str(rel), size=size, created=time.time())
        self._part(up.id).touch()
        self._save(up)
        return up

    def write(self, upload_id: str, offset: int, data: bytes) -> Upload:
        """Append ``data`` at ``offset``. A chunk sent twice (after a retry) is ignored; a gap
        is refused, and the browser continues from ``received``."""
        with self._lock(upload_id):
            up = self.get(upload_id)
            if up.state != "uploading":
                return up
            part = self._part(upload_id)
            received = part.stat().st_size if part.exists() else 0
            if offset != received:
                up.received = received
                return up  # the client resyncs from up.received
            if received + len(data) > up.size:
                raise UploadError("more data than announced")
            with part.open("ab") as fh:
                fh.write(data)
            up.received = received + len(data)
            self._save(up)
            return up

    def complete(self, upload_id: str) -> tuple[Upload, list[Path]]:
        with self._lock(upload_id):
            up = self.get(upload_id)
            if up.state == "done":
                return up, []
            part = self._part(upload_id)
            if not part.exists() or part.stat().st_size != up.size:
                raise UploadError("upload incomplete")
            try:
                if up.name.lower().endswith(".zip"):
                    placed = self._extract(part, PurePosixPath(up.name))
                    part.unlink(missing_ok=True)
                else:
                    placed = [self._place(part, PurePosixPath(up.name))]
            except (OSError, zipfile.BadZipFile, UploadError) as exc:
                up.state, up.error = "failed", str(exc)
                self._save(up)
                part.unlink(missing_ok=True)
                raise UploadError(str(exc)) from exc
            up.state, up.received, up.files = "done", up.size, len(placed)
            self._save(up)
            return up, placed

    def cancel(self, upload_id: str) -> None:
        with self._lock(upload_id):
            self._part(upload_id).unlink(missing_ok=True)
            with contextlib.suppress(UploadError):
                self._meta_path(upload_id).unlink(missing_ok=True)

    def cleanup(self, max_age_days: float = 7.0) -> int:
        """Remove unfinished uploads nobody continued for a week and old bookkeeping."""
        removed = 0
        cutoff = time.time() - max_age_days * 86400
        for meta in self.tmp.glob("*.json"):
            try:
                if meta.stat().st_mtime < cutoff:
                    self._part(meta.stem).unlink(missing_ok=True)
                    meta.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed

    # ------------------------------------------------------------------ placing files
    def _dest(self, rel: PurePosixPath) -> Path:
        # single files land in a folder per upload day; folder uploads keep their structure
        base = self.target / date.today().isoformat() if len(rel.parts) == 1 else self.target
        dest = (base / Path(*rel.parts)).resolve()
        if self.target not in dest.parents:
            raise UploadError("invalid path")
        dest.parent.mkdir(parents=True, exist_ok=True)
        return unique_path(dest)

    def _place(self, src: Path, rel: PurePosixPath) -> Path:
        dest = self._dest(rel)
        shutil.move(str(src), dest)
        return dest

    def _extract(self, archive: Path, rel: PurePosixPath) -> list[Path]:
        """Unpack the photos of a ZIP into ``<upload dir>/<zip name>/…``."""
        limit = self.settings.remote.max_zip_gb * 1024**3
        folder = PurePosixPath(*rel.parts[:-1], rel.stem)
        placed: list[Path] = []
        total = 0
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.flag_bits & 0x1:  # folders, encrypted entries
                    continue
                name = info.filename
                if "__MACOSX" in name or PurePosixPath(name).name.startswith("."):
                    continue
                try:
                    entry = safe_relative_path(name)
                except UploadError:
                    continue
                if entry.suffix.lower() not in self.extensions:
                    continue
                total += info.file_size
                if total > limit:
                    raise UploadError("ZIP too large when unpacked")
                if info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
                    raise UploadError("suspicious ZIP (compression ratio too high)")
                dest = self._dest(folder / entry)
                with zf.open(info) as src, dest.open("wb") as out:
                    copied = 0
                    while chunk := src.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > info.file_size:  # lies about its size
                            raise UploadError("corrupt ZIP entry")
                        out.write(chunk)
                placed.append(dest)
        return placed
