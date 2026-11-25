"""Small database helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")

#: SQLite allows at most 32,766 bound parameters per statement; stay far below.
MAX_PARAMS = 900


def chunked(items: Iterable[T], size: int = MAX_PARAMS) -> Iterator[list[T]]:
    """Yield lists of at most ``size`` items (for ``IN (...)`` clauses)."""
    seq: Sequence[T] = items if isinstance(items, Sequence) else list(items)
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


def like_escape(value: str, escape: str = "\\") -> str:
    """Escape ``%`` and ``_`` so user input is matched literally in ``LIKE``."""
    return (
        value.replace(escape, escape + escape).replace("%", escape + "%").replace("_", escape + "_")
    )
