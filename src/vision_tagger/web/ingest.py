"""Remote mode: index finished uploads and run the pipeline on them automatically.

Every completed upload calls :meth:`Ingest.notify`. A background thread waits until uploads
have been quiet for a few seconds (so a batch of 500 photos is indexed once, not 500 times),
indexes the upload folder as a normal "index" job (visible on the pipeline page) and then
starts a pipeline run unless one is active. A run the user paused stays paused, and a run the
user stopped is restarted only for photos that were uploaded after the stop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from ..i18n import Lazy
from ..indexer import Indexer
from ..pipeline import control

if TYPE_CHECKING:
    from .context import AppContext, Job

log = logging.getLogger(__name__)


class Ingest:
    def __init__(self, c: AppContext, quiet_seconds: float = 4.0) -> None:
        self.c = c
        self.quiet_seconds = quiet_seconds
        self._event = threading.Event()
        self._last = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="vision-tagger-ingest", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._last = time.monotonic()
        self._event.set()

    def close(self) -> None:
        self._stop.set()
        self._event.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._event.wait()
            if self._stop.is_set():
                return
            # wait until the uploads pause, then index everything that arrived
            while time.monotonic() - self._last < self.quiet_seconds and not self._stop.is_set():
                time.sleep(0.5)
            self._event.clear()
            try:
                self._process()
            except Exception:  # never let the worker die
                log.exception("processing uploads failed")
                self._event.set()
                time.sleep(10)

    def _process(self) -> None:
        c = self.c
        done = threading.Event()

        def job(j: Job) -> Lazy:
            try:
                stats = Indexer(c.db, c.settings, should_stop=lambda: j.stop_requested).run()
                return Lazy(  # made in the job thread: translated when shown
                    "Uploads: %(new)s new photos, %(dup)s duplicates",
                    new=stats.new,
                    dup=stats.duplicates,
                )
            finally:
                done.set()

        while not c.jobs.start("index", job):  # another job (export, scan) is busy
            if self._stop.wait(2.0):
                return
        done.wait()
        if not c.settings.remote.auto_process or c.runner.running:
            return
        state = control.get_control(c.db).desired_state
        if state == control.PAUSED:
            return  # the user paused: keep it that way
        c.runner.start(None)
