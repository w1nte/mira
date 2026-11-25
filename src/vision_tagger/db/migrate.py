"""Database schema migrations (Alembic).

* New databases are created from the models and stamped with the newest revision.
* Existing databases must be upgraded explicitly with ``vision-tagger migrate``. It makes a
  backup copy first and runs all pending revisions in one transaction, so a failing migration
  leaves the database untouched.
* Databases created before migrations existed (no ``alembic_version`` table) are treated as
  revision ``0001`` (the first released schema); later revisions check what is already there.

Rule for developers (see CLAUDE.md): every change to ``db/models.py`` needs a new revision in
``src/vision_tagger/migrations/versions``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.engine import make_url

if TYPE_CHECKING:
    from .session import Database

log = logging.getLogger(__name__)

#: databases without ``alembic_version`` are assumed to be at this revision
LEGACY_BASELINE = "0001"


class MigrationRequired(RuntimeError):
    """The database schema is older than this version of vision-tagger."""


class MigrationError(RuntimeError):
    pass


def alembic_config(url: str) -> Any:
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", "vision_tagger:migrations")
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def migration_engine(url: str) -> Engine:
    """Engine for migrations: transactional DDL and foreign keys *off*.

    SQLite rebuilds tables for most ALTERs (copy, drop, rename). With foreign keys on, dropping
    the old table would cascade-delete child rows, so they are disabled for the migration and
    checked afterwards with ``PRAGMA foreign_key_check``.
    """
    engine = create_engine(url, connect_args={"timeout": 60})
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _connect(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.isolation_level = None  # let SQLAlchemy emit BEGIN (DDL in transaction)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=OFF")
            cur.execute("PRAGMA busy_timeout=60000")
            cur.close()

        @event.listens_for(engine, "begin")
        def _begin(conn: Any) -> None:
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


def head_revision() -> str:
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(alembic_config("sqlite://")).get_current_head()
    assert head is not None
    return head


def current_revision(engine: Engine) -> str | None:
    from alembic.migration import MigrationContext

    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


@dataclass
class SchemaStatus:
    state: str  # "new" | "current" | "outdated" | "unversioned"
    current: str | None
    head: str

    @property
    def needs_upgrade(self) -> bool:
        return self.state in ("outdated", "unversioned")


def schema_status(engine: Engine) -> SchemaStatus:
    head = head_revision()
    tables = set(inspect(engine).get_table_names()) - {"sqlite_sequence"}
    if not tables:
        return SchemaStatus("new", None, head)
    rev = current_revision(engine)
    if rev is None:
        return SchemaStatus("unversioned", None, head)
    return SchemaStatus("current" if rev == head else "outdated", rev, head)


def _sqlite_path(url: str) -> Path | None:
    u = make_url(url)
    if u.get_backend_name() != "sqlite" or not u.database or u.database == ":memory:":
        return None
    return Path(u.database)


def backup_database(path: Path, label: str) -> Path:
    """Consistent copy via SQLite's backup API (safe with WAL and open readers)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.bak-{label}-{stamp}")
    src = sqlite3.connect(path)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


@dataclass
class UpgradeResult:
    before: str | None
    after: str
    state_before: str
    backup: Path | None = None

    @property
    def changed(self) -> bool:
        return self.state_before != "current"


def upgrade(url: str, *, backup: bool = True, revision: str = "head") -> UpgradeResult:
    """Bring the database to ``revision`` (default: newest). Backs up the file first."""
    from alembic import command

    engine = migration_engine(url)
    try:
        status = schema_status(engine)
        target = status.head if revision == "head" else revision
        if status.state == "current" and target == status.head:
            return UpgradeResult(status.current, status.head, status.state)
        path = _sqlite_path(url)
        backup_path = None
        if backup and path is not None and path.exists() and status.state != "new":
            backup_path = backup_database(path, status.current or "pre-migrations")
            log.info("backup written to %s", backup_path)
        cfg = alembic_config(url)
        with engine.begin() as conn:  # one transaction: all revisions or nothing
            cfg.attributes["connection"] = conn
            if status.state == "unversioned":
                command.stamp(cfg, LEGACY_BASELINE)
            command.upgrade(cfg, revision)
            problems = conn.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
            if problems:
                raise MigrationError(f"foreign key violations after migration: {problems[:5]}")
        return UpgradeResult(
            status.current, current_revision(engine) or "", status.state, backup_path
        )
    finally:
        engine.dispose()


def downgrade(url: str, revision: str, *, backup: bool = True) -> UpgradeResult:
    from alembic import command

    engine = migration_engine(url)
    try:
        status = schema_status(engine)
        path = _sqlite_path(url)
        backup_path = (
            backup_database(path, status.current or "unknown")
            if backup and path is not None and path.exists()
            else None
        )
        cfg = alembic_config(url)
        with engine.begin() as conn:
            cfg.attributes["connection"] = conn
            command.downgrade(cfg, revision)
        return UpgradeResult(
            status.current, current_revision(engine) or "", status.state, backup_path
        )
    finally:
        engine.dispose()


def ensure_schema(db: Database) -> None:
    """Create a new database, or check that an existing one is up to date.

    Raises :class:`MigrationRequired` for older databases (run ``vision-tagger migrate``).
    """
    from alembic import command

    from .models import Base

    status = schema_status(db.engine)
    if status.state == "current":
        return
    if status.state == "new":
        Base.metadata.create_all(db.engine)
        engine = migration_engine(db.url)
        try:
            cfg = alembic_config(db.url)
            with engine.begin() as conn:
                cfg.attributes["connection"] = conn
                command.stamp(cfg, "head")
        finally:
            engine.dispose()
        return
    where = _sqlite_path(db.url) or db.url
    raise MigrationRequired(
        f"The database {where} has schema version {status.current or 'pre-migrations'}, "
        f"this version of vision-tagger needs {status.head}. "
        "Run `vision-tagger migrate` to upgrade it (a backup copy is made first)."
    )
