"""Export and import of the whole library database (another computer, a backup).

* **Export** writes a consistent snapshot of the live database with ``VACUUM INTO``: a read
  transaction, so it works while the app and the pipeline keep writing (WAL mode), and the
  copy is compact and self-contained (no ``-wal`` file).
* **Import** replaces the library with such a file. The file is checked (SQLite, not damaged,
  a vision-tagger database, a schema this version knows), migrated to the current schema, and
  copied into the live database with SQLite's backup API. That works while other connections
  are open (on Windows an open file cannot be replaced) and readers keep their snapshot until
  the copy is committed. A backup of the previous database is kept next to it
  (``vision_tagger.sqlite3.bak-import-<time>``).

After an import, image ids mean other photos: the caller clears every cache keyed by ids
(:func:`clear_cache_dirs` here; in-memory caches in the web app) and then checks the location of
every photo (``Indexer.check_locations``). Photos that are not where the other computer had them
are flagged as missing; a scan of the folder where they are now re-links them by perceptual hash
and keeps their faces, names and properties.

Messages are :class:`~vision_tagger.i18n.Lazy`: the web UI shows them in the viewer's language,
the command line in English.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.engine import make_url

from ..i18n import Lazy
from . import migrate

if TYPE_CHECKING:
    from ..config import Settings
    from .session import Database

log = logging.getLogger(__name__)

SQLITE_MAGIC = b"SQLite format 3\x00"
#: key of the random id that changes with every import (the service worker's thumbnail cache)
LIBRARY_ID_KEY = "library_id"
#: temporary files older than this are left over from an interrupted transfer
STALE_SECONDS = 24 * 3600
MEDIA_TYPE = "application/vnd.sqlite3"


class TransferError(Exception):
    """The file cannot be imported (or exported); the message says why."""

    def __init__(self, message: Lazy) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return str(self.message)


@dataclass
class FileInfo:
    """What an import file contains (see :func:`inspect_file`)."""

    revision: str  # schema revision; files from before migrations count as the first one
    head: str
    images: int
    page_size: int

    @property
    def needs_upgrade(self) -> bool:
        return self.revision != self.head


@dataclass
class ImportResult:
    images: int
    revision: str  # schema revision of the imported file before it was migrated
    backup: Path | None
    roots: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- paths
def database_file(db: Database) -> Path:
    database = make_url(db.url).database
    if db.engine.dialect.name != "sqlite" or not database or database == ":memory:":
        raise TransferError(Lazy("Only a database file can be exported or imported."))
    return Path(database)


def export_name(today: date | None = None) -> str:
    """``vision-tagger-YYYY-MM-DD.sqlite3`` (technical names stay ``vision-tagger``)."""
    return f"vision-tagger-{(today or date.today()).isoformat()}.sqlite3"


def transfer_dir(settings: Settings) -> Path:
    """Temporary files of exports and imports: in the data folder, next to the database."""
    folder = settings.data_dir / "transfer"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def temp_file(settings: Settings, kind: str) -> Path:
    """A new, unique temporary file name (``kind``: ``export`` or ``import``)."""
    folder = transfer_dir(settings)
    remove_stale(folder)
    return folder / f"{kind}-{uuid.uuid4().hex}.sqlite3"


def remove_stale(folder: Path, max_age: float = STALE_SECONDS) -> None:
    """Delete temporary files an interrupted export or import left behind."""
    now = time.time()
    with contextlib.suppress(OSError):
        for path in folder.iterdir():
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime > max_age:
                    path.unlink()


def remove_file(path: Path) -> None:
    """Delete a temporary database file and its journal files; locked files are left."""
    for suffix in ("", "-journal", "-wal", "-shm"):
        with contextlib.suppress(OSError):
            Path(f"{path}{suffix}").unlink(missing_ok=True)


# --------------------------------------------------------------------------- export
def export_database(db: Database, target: Path) -> Path:
    """Write a consistent, compact copy of the live database to ``target`` (must not exist).

    Safe while the app and the pipeline write: ``VACUUM INTO`` reads one snapshot.
    """
    source = database_file(db)
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    con = sqlite3.connect(source, timeout=60, isolation_level=None)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        con.execute("PRAGMA busy_timeout=60000")
        con.execute("VACUUM INTO ?", (os.fspath(target),))
    except (sqlite3.Error, OSError) as exc:  # e.g. the disk is full
        remove_file(target)
        raise TransferError(Lazy("The export failed: %(error)s", error=str(exc))) from exc
    except BaseException:
        remove_file(target)
        raise
    finally:
        con.close()
    return target


# --------------------------------------------------------------------------- import: checks
def _known_revisions() -> set[str]:
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(migrate.alembic_config("sqlite://"))
    return {rev.revision for rev in script.walk_revisions()}


def inspect_file(path: Path) -> FileInfo:
    """Check that ``path`` can be imported; raises :class:`TransferError` if not.

    It must be an SQLite file, pass ``PRAGMA quick_check``, contain the vision-tagger tables
    and have a schema revision this version knows (a file without ``alembic_version`` is from
    before migrations existed: revision ``0001``). The file is switched to rollback-journal
    mode, so it is one self-contained file.
    """
    try:
        with Path(path).open("rb") as fh:
            header = fh.read(100)
    except OSError as exc:
        raise TransferError(Lazy("The file cannot be read: %(error)s", error=str(exc))) from exc
    if len(header) < 100 or not header.startswith(SQLITE_MAGIC):
        raise TransferError(Lazy("This is not a database file. Choose a file Mira exported."))
    try:
        con = sqlite3.connect(path, timeout=60, isolation_level=None)
    except sqlite3.Error as exc:
        raise TransferError(Lazy("The file cannot be read: %(error)s", error=str(exc))) from exc
    try:
        try:
            con.execute("PRAGMA journal_mode=DELETE")
            check = [row[0] for row in con.execute("PRAGMA quick_check")]
        except sqlite3.DatabaseError as exc:
            raise TransferError(
                Lazy("The database file is damaged: %(error)s", error=str(exc))
            ) from exc
        if check != ["ok"]:
            raise TransferError(
                Lazy("The database file is damaged: %(error)s", error="; ".join(check[:3]))
            )
        objects = con.execute("SELECT type, name FROM sqlite_master").fetchall()
        tables = {name for kind, name in objects if kind == "table"}
        if not {"images", "faces", "stage_status"} <= tables:
            raise TransferError(
                Lazy("This database is not a Mira library (it has no photo index).")
            )
        # Mira's schema has no triggers or views; they would run with every change
        foreign = sorted(name for kind, name in objects if kind in ("trigger", "view"))
        if foreign:
            raise TransferError(
                Lazy(
                    "This database contains triggers or views Mira does not create "
                    "(%(names)s). It cannot be imported.",
                    names=", ".join(foreign[:5]),
                )
            )
        revision = migrate.LEGACY_BASELINE
        if "alembic_version" in tables:
            rows = [row[0] for row in con.execute("SELECT version_num FROM alembic_version")]
            if len(rows) != 1:
                raise TransferError(
                    Lazy("The database has an unknown schema version (%(rev)s).", rev=rows)
                )
            revision = str(rows[0])
        head = migrate.head_revision()
        if revision not in _known_revisions():
            raise TransferError(
                Lazy(
                    "This database comes from a newer version of Mira (schema %(rev)s, this "
                    "version knows up to %(head)s). Update Mira first.",
                    rev=revision,
                    head=head,
                )
            )
        images = int(con.execute("SELECT count(*) FROM images").fetchone()[0])
        page_size = int(con.execute("PRAGMA page_size").fetchone()[0])
    finally:
        con.close()
    return FileInfo(revision=revision, head=head, images=images, page_size=page_size)


def _page_size(path: Path) -> int:
    con = sqlite3.connect(path, timeout=60)
    try:
        return int(con.execute("PRAGMA page_size").fetchone()[0])
    finally:
        con.close()


def prepare_file(path: Path, page_size: int) -> FileInfo:
    """Check ``path``, migrate it to the current schema and give it ``page_size``.

    ``path`` is changed in place: always pass a copy, never the user's own file. The backup
    API cannot copy into a database in WAL mode with another page size, hence the
    ``VACUUM`` with the live database's page size.
    """
    info = inspect_file(path)
    url = f"sqlite:///{path.as_posix()}"
    try:
        migrate.upgrade(url, backup=False)
    except Exception as exc:
        raise TransferError(
            Lazy("The database could not be upgraded: %(error)s", error=str(exc))
        ) from exc
    con = sqlite3.connect(path, timeout=60, isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode=DELETE")
        if int(con.execute("PRAGMA page_size").fetchone()[0]) != page_size:
            con.execute(f"PRAGMA page_size={int(page_size)}")
            con.execute("VACUUM")
    finally:
        con.close()
    return info


# --------------------------------------------------------------------------- import: replace
def _read_setting(path: Path, key: str) -> Any:
    import json

    con = sqlite3.connect(path, timeout=60)
    try:
        row = con.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def merge_roots(imported: Any, previous: list[str]) -> list[str]:
    """The imported roots that exist on this computer, then the previous ones that exist.

    Without duplicates (compared like the index compares paths: case-insensitive on Windows).
    """
    out: list[str] = []
    seen: set[str] = set()
    candidates = [str(r) for r in imported] if isinstance(imported, list) else []
    for root in [*candidates, *previous]:
        key = os.path.normcase(str(Path(root).absolute()))
        if key in seen or not Path(root).is_dir():
            continue
        seen.add(key)
        out.append(root)
    return out


def replace_database(db: Database, source: Path, *, backup: bool = True) -> Path | None:
    """Copy ``source`` (prepared, same page size) into the live database of ``db``.

    Uses SQLite's backup API: other connections may stay open, readers see the old data until
    the copy is committed in one transaction. Returns the backup of the previous database.
    """
    live = database_file(db)
    backup_path = migrate.backup_database(live, "import") if backup and live.exists() else None
    db.dispose()
    src = sqlite3.connect(source, timeout=60)
    dst = sqlite3.connect(live, timeout=60)
    try:
        dst.execute("PRAGMA busy_timeout=60000")
        src.backup(dst)
        with contextlib.suppress(sqlite3.Error):  # best effort, readers may hold the WAL
            dst.execute("PRAGMA wal_checkpoint(PASSIVE)")
    finally:
        dst.close()
        src.close()
    db.dispose()  # no pooled connection keeps state from before the copy
    return backup_path


def _clear_dir(folder: Path) -> None:
    """Empty a cache folder. Renaming it first is instant, so no request can find an old file
    in it while the files are deleted; if that fails (Windows: a file is open) the files are
    deleted in place and locked ones are left."""
    for leftover in folder.parent.glob(f"{folder.name}.old-*"):
        shutil.rmtree(leftover, ignore_errors=True)
    if not folder.exists():
        return
    trash = folder.with_name(f"{folder.name}.old-{uuid.uuid4().hex[:8]}")
    try:
        folder.rename(trash)
    except OSError:
        trash = folder
    shutil.rmtree(trash, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    left = sum(1 for p in folder.rglob("*") if p.is_file())
    if left:
        log.warning("%d cached files in %s could not be deleted (in use?)", left, folder)


def clear_cache_dirs(settings: Settings) -> None:
    """Delete the caches keyed by image, face or animal id: thumbnails, face and animal crops,
    aligned portraits and grow-up animations. They are made again on demand."""
    for folder in (
        settings.thumbs_dir,
        settings.face_crops_dir,
        settings.animal_crops_dir,
        settings.cache_dir / "aligned",
    ):
        _clear_dir(folder)


def import_database(
    db: Database,
    settings: Settings,
    source: Path,
    *,
    before_replace: Callable[[], None] | None = None,
    progress: Callable[[Lazy], None] | None = None,
) -> ImportResult:
    """Replace the library of ``db`` with the database file ``source``.

    ``source`` is changed (migrated): pass a temporary copy. ``before_replace`` runs right
    before the live data is replaced and may raise :class:`TransferError` (e.g. a pipeline was
    started meanwhile). Afterwards the pipeline state of the other computer is reset, the
    library folders are merged, and the id-keyed caches on disk are cleared. The caller still
    has to clear in-memory caches and check the photo locations.
    """
    from ..indexer import ROOTS_KEY, ensure_stage_rows, stored_roots
    from ..pipeline.stages import STAGE_NAMES
    from .models import AppSetting, Control

    report = progress or (lambda _m: None)
    live = database_file(db)
    report(Lazy("Checking and upgrading the database file…"))
    info = prepare_file(source, _page_size(live))
    imported_roots = _read_setting(source, ROOTS_KEY)
    previous_roots = [] if settings.remote_mode else stored_roots(db, settings)
    if before_replace is not None:
        before_replace()
    report(Lazy("Replacing the library…"))
    backup_path = replace_database(db, source)
    db.create_all()  # schema check; the control row if the file had none
    from ..pipeline import control

    # no "running" items, runs or runner from the other computer
    control.recover_stale(db)
    with db.write() as s:
        ctrl = s.get(Control, 1)
        if ctrl is not None:
            ctrl.heartbeat_at = None
            ctrl.message = None
            ctrl.progress = {}
    ensure_stage_rows(db, settings.pipeline.stages or list(STAGE_NAMES))
    roots = merge_roots(imported_roots, previous_roots)
    with db.write() as s:
        if roots:
            s.merge(AppSetting(key=ROOTS_KEY, value=roots))
        else:  # nothing usable: back to the configured folders
            stored = s.get(AppSetting, ROOTS_KEY)
            if stored is not None:
                s.delete(stored)
        s.merge(AppSetting(key=LIBRARY_ID_KEY, value=uuid.uuid4().hex))
    report(Lazy("Clearing thumbnails…"))
    clear_cache_dirs(settings)
    from ..faces import ages

    ages._CALIBRATION_CACHE.clear()
    return ImportResult(images=info.images, revision=info.revision, backup=backup_path, roots=roots)


def library_id(db: Database) -> str:
    """Changes with every import (``"0"`` before the first one)."""
    from .models import AppSetting

    with db.session() as s:
        stored = s.get(AppSetting, LIBRARY_ID_KEY)
        return str(stored.value) if stored is not None and stored.value else "0"


def busy_reason(
    db: Database, *, runner_running: bool = False, jobs_busy: bool = False
) -> Lazy | None:
    """Why an import must wait, or ``None``: a job, or a pipeline (running or paused, in this
    or another process)."""
    from ..pipeline import control

    if jobs_busy:
        return Lazy("Another job is running. Wait until it has finished.")
    if runner_running or control.is_runner_alive(db):
        return Lazy("The pipeline is running or paused. Stop the pipeline first.")
    return None
