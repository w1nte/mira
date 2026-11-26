"""Cache for expensive read-only results that stay valid until the database changes.

The key is SQLite's ``PRAGMA data_version`` (see :meth:`Database.data_version`): it changes
whenever any connection commits, including the pipeline in another process, so cached values
are never served after the data they were computed from changed. Values are computed in their
own fresh read transaction *after* the version was read, so a stored value is never older than
the version it is stored under.

Cached values are shared between requests: store plain data (numbers, tuples, dataclasses of
primitives), never ORM objects, and treat them as read-only.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .session import Database

T = TypeVar("T")


class VersionedCache:
    def __init__(self, db: Database, maxsize: int = 1024) -> None:
        self.db = db
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._version: int | None = None
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, compute: Callable[[Session], T]) -> T:
        """Cached ``compute(session)``; recomputed after any commit to the database."""
        version = self.db.data_version()
        if version is not None:
            with self._lock:
                if version != self._version:
                    self._data.clear()
                    self._version = version
                elif key in self._data:
                    self._data.move_to_end(key)
                    self.hits += 1
                    return self._data[key]
        self.misses += 1
        with self.db.session() as s:
            value = compute(s)
        if version is not None:
            with self._lock:
                if version == self._version:
                    self._data[key] = value
                    while len(self._data) > self.maxsize:
                        self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._version = None
