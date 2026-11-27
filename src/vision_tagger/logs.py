"""Mira's log file: what it did, and above all its warnings and errors.

Every process that works on a library writes to ``<data_dir>/logs/vision-tagger.log``: the
desktop app, ``vision-tagger serve`` (also in Docker) and the command line. Each process starts
with a line that names the version, the system and what was started, so the log shows what led
up to an error. The web UI shows it (System → Log, ``web.routes.logs``); it belongs in a bug
report (``bugreport``).

Size: before the file grows beyond ``logging.max_mb``, its content moves to
``vision-tagger.log.1`` (an older ``.1`` to ``.2``, and so on) and what is beyond
``logging.backups`` is deleted. The oldest messages go first; all files together stay below
``(backups + 1) * max_mb``.

Windows first:

* The file is rotated by *copying* and truncating it, never by renaming it: Windows can't rename
  a file that another program has open (a second Mira process, an editor, antivirus), but it can
  truncate it. When rotating fails, the record is written anyway and rotating is tried again
  later; at twice the size limit the old content is dropped without a copy.
* Several processes may write the same file (the app and a ``vision-tagger run``): each opens it
  for appending and checks its real size before every record. A record that another process
  writes between the copy and the truncation can get lost.
* Logging never breaks the app: when the file can't be written (read-only folder, full disk),
  records are dropped and opening it is tried again after a while.

The uvicorn access log (a line per request) stays out of the file; uvicorn's errors (an exception
in a request) go in.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import logging
import os
import platform
import re
import shutil
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from . import APP_NAME, __version__

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

FILE_NAME = "vision-tagger.log"
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
#: ``2026-11-27 10:15:03.123 WARNING  [4242 vision-tagger-runner] vision_tagger.x: message``;
#: the process id and thread tell apart what several processes and threads write
FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s [%(process)d %(threadName)s] %(name)s: %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"
#: libraries that say too much at INFO (httpx: a line per request to Ollama; alembic: its
#: plug-ins on every start)
QUIET = ("httpx", "httpcore", "alembic.runtime.plugins")
#: what the web page reads at most from the end of the log
TAIL_BYTES = 2_000_000

_MARK = "_vision_tagger_log"  # on the handlers and hooks this module installs


def log_dir(settings: Settings) -> Path:
    return settings.data_dir / "logs"


def log_path(settings: Settings) -> Path:
    return log_dir(settings) / FILE_NAME


# --------------------------------------------------------------------------- the file
class LogFile(logging.Handler):
    """Appends records to ``path`` and keeps it below ``max_bytes`` (see the module docstring).

    The file is opened with the first record (nothing is created before something is logged).
    """

    #: seconds before opening or rotating is tried again after it failed
    retry_seconds = 30.0

    def __init__(self, path: Path, max_bytes: int, backups: int = 3) -> None:
        super().__init__()
        self.path = Path(path)
        self.max_bytes = max(int(max_bytes), 4096)
        self.backups = max(int(backups), 0)
        self.clock: Callable[[], float] = time.monotonic
        self._stream: TextIO | None = None
        self._open_failed: float | None = None
        self._rotate_failed: float | None = None

    def backup(self, n: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{n}")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record) + "\n"
        except Exception:
            self.handleError(record)
            return
        stream = self._ensure_stream()
        if stream is None:
            return
        try:
            size = os.fstat(stream.fileno()).st_size  # also what other processes wrote
            if size and size + _bytes(text) > self.max_bytes:
                self._rotate(stream, size)
            stream.write(text)
            stream.flush()
        except OSError:  # full disk, a vanished folder: drop it, open again later
            self._close_stream()
            self._open_failed = self.clock()
        except Exception:
            self.handleError(record)

    def _ensure_stream(self) -> TextIO | None:
        if self._stream is not None:
            return self._stream
        if self._open_failed is not None and self.clock() - self._open_failed < self.retry_seconds:
            return None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # append: every process writes at the end, also after another one truncated it
            self._stream = self.path.open(  # kept open; closed by close()
                "a", encoding="utf-8", errors="backslashreplace"
            )
        except OSError:
            self._open_failed = self.clock()
            return None
        self._open_failed = None
        return self._stream

    def _rotate(self, stream: TextIO, size: int) -> None:
        now = self.clock()
        overdue = size >= 2 * self.max_bytes
        if (
            self._rotate_failed is not None
            and now - self._rotate_failed < self.retry_seconds
            and not overdue
        ):
            return
        try:
            self._keep_copy()
        except OSError:
            if not overdue:
                self._rotate_failed = now
                return
            # the old messages can't be kept: drop them rather than grow without limit
        try:
            stream.truncate(0)
        except OSError:
            self._rotate_failed = now
            return
        self._rotate_failed = None

    def _keep_copy(self) -> None:
        """``.1`` -> ``.2`` ... (the oldest is overwritten), then the file -> ``.1``."""
        if not self.backups:
            return
        for n in range(self.backups - 1, 0, -1):
            older = self.backup(n)
            if older.exists():
                older.replace(self.backup(n + 1))
        shutil.copyfile(self.path, self.backup(1))

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()

    def close(self) -> None:
        self.acquire()
        try:
            self._close_stream()
        finally:
            self.release()
        super().close()


def _bytes(text: str) -> int:
    """What ``text`` takes in the file: UTF-8, and on Windows two bytes per line end."""
    return len(text.encode("utf-8", "backslashreplace")) + text.count("\n") * (len(os.linesep) - 1)


class _Noise(logging.Filter):
    """Kept out of the file: uvicorn's line per HTTP request (on the console of ``serve``), and
    the two lines alembic writes whenever the database is opened. Its migration steps stay."""

    ALEMBIC = ("Context impl ", "Will assume ")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("uvicorn.access"):
            return False
        if record.name == "alembic.runtime.migration":
            return not str(record.msg).startswith(self.ALEMBIC)
        return True


# --------------------------------------------------------------------------- setup
def _ours(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _MARK, False))


def _replace(kind: type[logging.Handler], new: logging.Handler | None) -> None:
    """Remove the handlers of ``kind`` this module installed on the root logger; add ``new``."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if _ours(handler) and type(handler) is kind:
            root.removeHandler(handler)
            handler.close()
    if new is not None:
        setattr(new, _MARK, True)
        root.addHandler(new)
        if root.level != logging.NOTSET and new.level < root.level:  # only ever more
            root.setLevel(new.level)
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)


def console(level: int = logging.INFO) -> None:
    """Log to stderr (the command line, ``serve``, the app started from a terminal).

    Like :func:`logging.basicConfig`: nothing is added when someone else (a program that embeds
    Mira, the test runner) already configured the root logger.
    """
    root = logging.getLogger()
    if any(not _ours(h) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT))
    _replace(logging.StreamHandler, handler)


def setup(settings: Settings, started: str, *, verbose: bool = False) -> Path:
    """Log to the library's log file from now on; returns its path.

    ``started`` says what this process is (``desktop app``, ``server``, ``command line: run``).
    A second call (another library, as ``migrate --desktop`` does) replaces the first.
    """
    cfg = settings.logging
    path = log_path(settings)
    handler = LogFile(path, max_bytes=int(cfg.max_mb * 1_000_000), backups=cfg.backups)
    handler.setLevel(logging.DEBUG if verbose else LEVELS[cfg.level])
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    handler.addFilter(_Noise())
    _replace(LogFile, handler)
    logging.captureWarnings(True)  # warnings.warn() of Mira and its libraries
    _install_hooks()
    log.info(
        "%s %s started (%s): %s; data folder %s",
        APP_NAME,
        __version__,
        started,
        system(settings),
        settings.data_dir,
    )
    return path


def file_handler() -> LogFile | None:
    """The log file this process writes (None before :func:`setup`)."""
    return next(
        (h for h in logging.getLogger().handlers if isinstance(h, LogFile) and _ours(h)), None
    )


def reset() -> None:
    """Undo :func:`console` and :func:`setup` (tests; a process that is done)."""
    _replace(LogFile, None)
    _replace(logging.StreamHandler, None)
    logging.captureWarnings(False)
    hook = sys.excepthook
    if getattr(hook, _MARK, False):
        sys.excepthook = hook.previous  # type: ignore[attr-defined]
    thread_hook = threading.excepthook
    if getattr(thread_hook, _MARK, False):
        threading.excepthook = thread_hook.previous  # type: ignore[attr-defined]


def to_file(
    message: str, level: int = logging.INFO, exc_info: Any = None, name: str = __name__
) -> None:
    """Write a record to the log file only, for what is printed on the console anyway (the
    report of ``migrate``, an uncaught error)."""
    handler = file_handler()
    if handler is None or level < handler.level:
        return
    handler.handle(logging.getLogger(name).makeRecord(name, level, "", 0, message, (), exc_info))


def _install_hooks() -> None:
    """Uncaught exceptions (in the main thread and in other threads) go into the log too."""
    if not getattr(sys.excepthook, _MARK, False):
        previous = sys.excepthook

        def excepthook(kind: Any, value: Any, tb: Any) -> None:
            if not issubclass(kind, KeyboardInterrupt):
                to_file("uncaught exception", logging.CRITICAL, (kind, value, tb))
            previous(kind, value, tb)

        setattr(excepthook, _MARK, True)
        excepthook.previous = previous  # type: ignore[attr-defined]
        sys.excepthook = excepthook

    if not getattr(threading.excepthook, _MARK, False):
        previous_thread_hook = threading.excepthook

        def thread_excepthook(args: Any) -> None:
            if args.exc_type is not SystemExit:
                name = args.thread.name if args.thread is not None else "?"
                to_file(
                    f"uncaught exception in thread {name}",
                    logging.ERROR,
                    (args.exc_type, args.exc_value, args.exc_traceback),
                )
            previous_thread_hook(args)

        setattr(thread_excepthook, _MARK, True)
        thread_excepthook.previous = previous_thread_hook  # type: ignore[attr-defined]
        threading.excepthook = thread_excepthook


# --------------------------------------------------------------------------- this computer
def _installation() -> str:
    if getattr(sys, "frozen", False):
        return "desktop app"
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return "container"
    return "Python package"  # a source checkout or pip


@functools.cache
def _computer() -> str:
    from .updates import installed_flavour

    return (
        f"{_installation()} ({installed_flavour()}), {platform.platform()} "
        f"({platform.machine() or '?'}), Python {platform.python_version()}"
    )


def system(settings: Settings | None = None) -> str:
    """How Mira runs, for the log and bug reports: no paths, no names.

    ``desktop app (gpu), Windows-11-10.0.26100-SP0 (AMD64), Python 3.12.7, local mode``
    """
    text = _computer()
    return f"{text}, {settings.mode} mode" if settings is not None else text


# --------------------------------------------------------------------------- reading
_HEADER = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:[.,]\d+)? +"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) +"
    r"(?:\[(?P<where>[^\]]*)\] +)?(?P<logger>[^\s:]+): (?P<message>.*)$"
)


@dataclass
class Entry:
    """One record of the log; ``details`` are its further lines (a traceback)."""

    time: str
    level: str
    logger: str
    message: str
    where: str = ""
    details: list[str] = field(default_factory=list)

    @property
    def levelno(self) -> int:
        return logging.getLevelNamesMapping().get(self.level, logging.INFO)

    @property
    def severity(self) -> str:
        """``error``, ``warn``, ``info`` or ``debug`` (the page's colours)."""
        if self.levelno >= logging.ERROR:
            return "error"
        if self.levelno >= logging.WARNING:
            return "warn"
        return "info" if self.levelno >= logging.INFO else "debug"

    @property
    def detail_text(self) -> str:
        return "\n".join(self.details).strip("\n")


@dataclass
class LogView:
    """What the web page shows: the newest entries first, and what the read part holds."""

    entries: list[Entry]
    counts: Counter[str]  # by severity, of every entry read
    read: int  # entries read (the end of the log)
    more: bool  # more entries matched than are shown


def files(path: Path) -> list[Path]:
    """The log file and its backups that exist, oldest first."""
    backups = []
    for candidate in path.parent.glob(glob.escape(path.name) + ".*"):
        suffix = candidate.name[len(path.name) + 1 :]
        if suffix.isdigit():
            backups.append((int(suffix), candidate))
    found = [p for _, p in sorted(backups, reverse=True)]
    return [*found, path] if path.is_file() else found


def tail(paths: list[Path], max_bytes: int = TAIL_BYTES) -> str:
    """The last ``max_bytes`` of the files read one after the other, from a line start."""
    chunks: list[bytes] = []
    left = max_bytes
    cut = False
    for path in reversed(paths):
        if left <= 0:
            cut = True
            break
        try:
            with path.open("rb") as f:
                size = f.seek(0, os.SEEK_END)
                start = max(0, size - left)
                f.seek(start)
                data = f.read(size - start)
        except OSError:
            continue
        chunks.append(data)
        left -= len(data)
        if start > 0:
            cut = True
            break
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    if cut:  # the first line is a part of a line
        text = text.partition("\n")[2]
    return text


def parse(text: str) -> list[Entry]:
    """The records in ``text``, oldest first. Lines before the first record are dropped."""
    entries: list[Entry] = []
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            entries.append(
                Entry(
                    time=m["time"],
                    level=m["level"],
                    logger=m["logger"],
                    message=m["message"],
                    where=m["where"] or "",
                )
            )
        elif entries:
            entries[-1].details.append(line)
    return entries


def read(
    path: Path, *, min_level: int = logging.INFO, limit: int = 300, max_bytes: int = TAIL_BYTES
) -> LogView:
    """The newest ``limit`` entries at ``min_level`` or above, newest first."""
    entries = parse(tail(files(path), max_bytes))
    matching = [e for e in entries if e.levelno >= min_level]
    return LogView(
        entries=matching[: -limit - 1 : -1] if limit else [],
        counts=Counter(e.severity for e in entries),
        read=len(entries),
        more=len(matching) > limit,
    )


def iter_bytes(path: Path, chunk: int = 64 * 1024) -> Iterator[bytes]:
    """The log file and its backups as one file, oldest first (the download)."""
    for part in files(path):
        try:
            with part.open("rb") as f:
                while data := f.read(chunk):
                    yield data
        except OSError:
            continue
