"""Engine and session handling.

SQLite is used in WAL mode so the web UI can read while the pipeline writes. Write
transactions are opened with ``BEGIN IMMEDIATE`` (see :meth:`Database.write`) so that
concurrent writers wait on ``busy_timeout`` instead of failing with ``database is locked``
when a deferred read transaction tries to upgrade to a write.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Control


class Database:
    def __init__(self, url_or_path: str | Path, *, echo: bool = False) -> None:
        if isinstance(url_or_path, Path) or "://" not in str(url_or_path):
            path = Path(url_or_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path.as_posix()}"
        else:
            url = str(url_or_path)
        self.url = url
        self.engine: Engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False, "timeout": 60},
        )
        if self.engine.dialect.name == "sqlite":
            _install_sqlite_hooks(self.engine)
        self._immediate_engine = self.engine.execution_options(sqlite_immediate=True)
        self._read = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._write = sessionmaker(bind=self._immediate_engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        self._add_missing_columns()
        with self.write() as s:
            if s.get(Control, 1) is None:
                s.add(Control(id=1, desired_state="stopped", progress={}))

    def _add_missing_columns(self) -> None:
        """Tiny forward migration: add new nullable / defaulted columns and their indexes.

        ``create_all`` only creates missing tables, so databases created by an older version
        would lack columns added later (e.g. ``persons.birth_year``).
        """
        insp = inspect(self.engine)
        existing_tables = set(insp.get_table_names())
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    continue
                have = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name in have or col.primary_key:
                        continue
                    ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" '
                    ddl += col.type.compile(dialect=self.engine.dialect)
                    default = getattr(col.default, "arg", None)
                    if isinstance(default, str):
                        ddl += " DEFAULT '" + default.replace("'", "''") + "'"
                    elif isinstance(default, bool):
                        ddl += f" DEFAULT {int(default)}"
                    elif isinstance(default, int | float):
                        ddl += f" DEFAULT {default}"
                    conn.exec_driver_sql(ddl)
                for index in table.indexes:
                    index.create(conn, checkfirst=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Read-mostly session (deferred transaction). Commits on success."""
        s = self._read()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()

    @contextmanager
    def write(self) -> Iterator[Session]:
        """Write session using ``BEGIN IMMEDIATE``. Commits on success."""
        s = self._write()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()

    def dispose(self) -> None:
        self.engine.dispose()


def _install_sqlite_hooks(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        # Let SQLAlchemy emit BEGIN itself (see _on_begin).
        dbapi_conn.isolation_level = None
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=60000")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Any) -> None:
        if conn.get_execution_options().get("sqlite_immediate"):
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            conn.exec_driver_sql("BEGIN")
