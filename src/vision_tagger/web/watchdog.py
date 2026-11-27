"""Slow pages explain themselves in the log.

When a page takes longer than :data:`SLOW_SECONDS` to start its answer, the log gets a warning
with what every thread of Mira was doing at that moment: waiting for the database, for Ollama,
for a lock, or computing. The event loop that receives every request is watched too: while it
does not run (:data:`STALL_SECONDS`), no page loads at all. Each report also says how many of
the worker threads that run the pages are in use (when all are, requests queue).

- Only GET requests are watched (pages and their parts); uploads and form posts may take long
  for good reasons. A request counts until its answer *starts*: a download that streams for
  minutes is fine.
- Paths are logged without the query string (it can hold search words or names).
- Stacks at most every :data:`REPORT_EVERY` seconds (the log is capped); in between, one line
  per check names the slow requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: a page that has not started its answer after this long is reported
SLOW_SECONDS = 10.0
#: the event loop has not run for this long: nothing loads meanwhile
STALL_SECONDS = 5.0
#: thread stacks at most this often
REPORT_EVERY = 60.0
#: how often the watchdog looks
INTERVAL = 1.0
#: innermost frames shown per thread
FRAMES = 14
WATCHED_METHODS = ("GET", "HEAD")


@dataclass
class _Request:
    method: str
    path: str
    started: float
    reported: bool = False


class Watchdog:
    """Tracks requests in flight (:class:`WatchdogMiddleware`) and reports slow ones from a
    thread of its own: a stuck event loop or worker thread can't report itself."""

    def __init__(
        self,
        *,
        slow: float = SLOW_SECONDS,
        stall: float = STALL_SECONDS,
        report_every: float = REPORT_EVERY,
        interval: float = INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.slow = slow
        self.stall = stall
        self.report_every = report_every
        self.interval = interval
        self.clock = clock
        self._lock = threading.Lock()
        self._requests: dict[int, _Request] = {}
        self._ids = itertools.count()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._limiter: Any = None  # anyio's limiter of the worker threads that run the pages
        self._ping_sent: float | None = None
        self._stall_reported = False
        self._stacks_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start watching; call it on the event loop (the app's lifespan)."""
        self._loop = asyncio.get_running_loop()
        try:
            import anyio.to_thread

            self._limiter = anyio.to_thread.current_default_thread_limiter()
        except Exception:  # pragma: no cover - another async backend
            self._limiter = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="vision-tagger-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except Exception:  # the watchdog must never take the server down
                log.exception("watchdog check failed")

    # ------------------------------------------------------------------ requests
    def begin(self, method: str, path: str) -> int:
        key = next(self._ids)
        with self._lock:
            self._requests[key] = _Request(method, path, self.clock())
        return key

    def end(self, key: int) -> None:
        with self._lock:
            request = self._requests.pop(key, None)
        if request is not None and request.reported:
            log.warning(
                "%s %s answered after %.0f s",
                request.method,
                request.path,
                self.clock() - request.started,
            )

    # ------------------------------------------------------------------ checks
    def check(self) -> None:
        """One look: the event loop, then the requests in flight."""
        now = self.clock()
        self._check_loop(now)
        with self._lock:
            due = now - self.slow
            slow = [r for r in self._requests.values() if not r.reported and r.started <= due]
            for r in slow:
                r.reported = True
        if slow:
            listed = ", ".join(f"{r.method} {r.path} ({now - r.started:.0f} s)" for r in slow)
            self._report(f"slow page: {listed} not answered yet", now)

    def _check_loop(self, now: float) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            sent = self._ping_sent
            if sent is None:
                self._ping_sent = now
            elif now - sent >= self.stall and not self._stall_reported:
                self._stall_reported = True
            else:
                return
        if sent is None:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                loop.call_soon_threadsafe(self._pong)
            return
        self._report(
            f"the web server's event loop has not run for {now - sent:.0f} s; no page loads "
            "until it does",
            now,
        )

    def _pong(self) -> None:
        """Runs on the event loop: it is alive."""
        now = self.clock()
        with self._lock:
            sent, stalled = self._ping_sent, self._stall_reported
            self._ping_sent, self._stall_reported = None, False
        if stalled and sent is not None:
            log.warning("the web server's event loop runs again after %.0f s", now - sent)

    def _workers(self) -> str:
        limiter = self._limiter
        if limiter is None:
            return ""
        try:
            return f"; worker threads in use: {limiter.borrowed_tokens} of {limiter.total_tokens}"
        except Exception:  # pragma: no cover - another anyio version
            return ""

    def _report(self, what: str, now: float) -> None:
        if self._stacks_at is not None and now - self._stacks_at < self.report_every:
            log.warning("%s%s", what, self._workers())
            return
        self._stacks_at = now
        log.warning("%s%s. What the threads are doing:\n%s", what, self._workers(), thread_stacks())


class WatchdogMiddleware:
    """ASGI middleware that tells the :class:`Watchdog` when requests start and answer."""

    def __init__(self, app: Any, watchdog: Watchdog) -> None:
        self.app = app
        self.watchdog = watchdog

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") not in WATCHED_METHODS:
            await self.app(scope, receive, send)
            return
        key = self.watchdog.begin(scope["method"], scope.get("path", ""))
        answered = False

        async def watched_send(message: Any) -> None:
            nonlocal answered
            if not answered and message.get("type") == "http.response.start":
                answered = True
                self.watchdog.end(key)
            await send(message)

        try:
            await self.app(scope, receive, watched_send)
        finally:
            if not answered:
                self.watchdog.end(key)


# --------------------------------------------------------------------------- stacks
#: where the module path starts: installed packages, the frozen app, a source checkout
_MARKERS = ("/site-packages/", "/dist-packages/", "/_internal/", "/src/")


def _short(filename: str) -> str:
    """``vision_tagger/web/app.py``, ``httpx/_client.py``, ``Lib/threading.py``: no folders of
    this computer (they can hold the user's name)."""
    path = filename.replace("\\", "/")
    for marker in _MARKERS:
        if marker in path:
            return path.rsplit(marker, 1)[1]
    return "/".join(path.rsplit("/", 2)[-2:])  # the standard library: its folder and file


def _format(frame: Any) -> str:
    lines = []
    for fs in traceback.extract_stack(frame, limit=FRAMES):
        code = f": {fs.line.strip()}" if fs.line else ""
        lines.append(f"    {_short(fs.filename)}:{fs.lineno} in {fs.name}{code}")
    return "\n".join(lines)


def thread_stacks() -> str:
    """The innermost frames of every thread but the calling one; threads doing the same thing
    (idle worker threads) are listed once."""
    names = {t.ident: t.name for t in threading.enumerate()}
    me = threading.get_ident()
    groups: dict[str, list[str]] = {}
    for ident, frame in sys._current_frames().items():
        if ident == me:
            continue
        groups.setdefault(_format(frame), []).append(names.get(ident, f"thread {ident}"))
    parts = []
    for stack, threads in sorted(groups.items(), key=lambda g: (len(g[1]), g[1][0])):
        title = threads[0] if len(threads) == 1 else f"{len(threads)} threads ({threads[0]}, …)"
        parts.append(f"  {title}:\n{stack}")
    return "\n".join(parts)
