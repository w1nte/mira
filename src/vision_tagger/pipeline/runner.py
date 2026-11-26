"""Batched, resumable pipeline runner.

* Work is claimed in batches from ``stage_status`` (``pending`` -> ``running``).
* Every image is committed individually, so an interrupted run loses at most the images that
  were in flight; those are reset to ``pending`` on the next start (crash recovery).
* ``control.desired_state`` is polled between images: ``paused`` waits, ``stopped`` releases
  the remaining claimed rows and ends the run.
* Stages run one after another (all faces first, then all VLM), so only one model occupies
  the GPU at a time.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import or_, select, update

from ..config import Settings
from ..db.models import Control, Image, Run, StageStatus, Status, utcnow
from ..db.session import Database
from . import control
from .control import PAUSED, PULL_PROGRESS, RUNNING, STOPPED, RunnerBusyError
from .stages.base import Stage, StageAbort, StageInterrupted, StageUnavailable

log = logging.getLogger(__name__)

StageFactory = Callable[[str], Stage]


__all__ = ["BackgroundRunner", "PipelineRunner", "RunSummary", "RunnerBusyError"]


class _Heartbeat(threading.Thread):
    """Keeps ``control.heartbeat_at`` fresh for the whole run.

    Without it a single long step (model setup, a slow VLM call, clustering) would make the run
    look dead, and another process could "recover" its images while they are still in work.
    """

    def __init__(self, db: Database, run_id: int, interval: float) -> None:
        super().__init__(name=f"vision-tagger-heartbeat-{run_id}", daemon=True)
        self.db = db
        self.run_id = run_id
        self.interval = max(0.05, interval)
        self.stage: str | None = None
        self._stop_event = threading.Event()

    def beat(self) -> None:
        try:
            with self.db.write() as s:
                now = utcnow()
                ctrl = s.get(Control, 1)
                if ctrl is not None and ctrl.active_run_id == self.run_id:
                    ctrl.heartbeat_at = now
                    ctrl.active_stage = self.stage
                run = s.get(Run, self.run_id)
                if run is not None:
                    run.heartbeat_at = now
        except Exception as exc:  # never let the heartbeat kill the run
            log.warning("heartbeat failed: %s", exc)

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self.beat()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


@dataclass
class RunSummary:
    run_id: int
    status: str = RUNNING
    processed: dict[str, int] = field(default_factory=dict)
    failed: dict[str, int] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


class PipelineRunner:
    def __init__(self, db: Database, settings: Settings, stage_factory: StageFactory) -> None:
        self.db = db
        self.settings = settings
        self.stage_factory = stage_factory
        self.poll_interval = 0.5
        self._lock = threading.Lock()
        self._hb: _Heartbeat
        self._shutdown = threading.Event()
        # desired state as long set-up steps see it (re-read every half second)
        self._state = RUNNING
        self._state_at = 0.0
        self._reported: tuple[str | None, float] = (None, 0.0)

    def shutdown(self) -> None:
        """The application is closing: end the run after the current image, but leave the
        pipeline *paused* (not stopped) so that the next start continues it."""
        self._shutdown.set()

    # ------------------------------------------------------------------ public
    def run(
        self,
        stages: Iterable[str] | None = None,
        *,
        batch_size: int | None = None,
        limit: int | None = None,
    ) -> RunSummary:
        stage_names = list(stages or self.settings.pipeline.stages)
        batch_size = max(1, batch_size or self.settings.pipeline.batch_size)
        stale_after = max(30.0, self.settings.pipeline.heartbeat_seconds * 6)
        if not self._lock.acquire(blocking=False):
            raise RunnerBusyError("a run is already active in this process")
        try:
            run_id, recovered = control.try_acquire(
                self.db, stage_names, batch_size, limit, stale_after
            )
            if recovered:
                log.info("recovered %d interrupted items", recovered)
            from ..indexer import ensure_stage_rows

            # a stage enabled after images were indexed (e.g. "ages") has no rows yet
            created = ensure_stage_rows(self.db, stage_names)
            if created:
                log.info("queued %d images for newly enabled stages", created)
            summary = RunSummary(run_id=run_id)
            self._hb = _Heartbeat(self.db, run_id, self.settings.pipeline.heartbeat_seconds)
            self._hb.start()
            try:
                for name in stage_names:
                    if not self._run_stage(name, run_id, batch_size, limit, summary):
                        break
            except StageAbort as exc:
                summary.status = "failed"
                summary.messages.append(str(exc))
                log.error("run aborted: %s", exc)
            except Exception as exc:
                summary.status = "failed"
                summary.messages.append(f"{type(exc).__name__}: {exc}")
                log.exception("pipeline run failed")
            else:
                if summary.status == RUNNING:
                    summary.status = "completed"
            finally:
                self._hb.stop()
            if self._shutdown.is_set() and summary.status == STOPPED:
                summary.status = PAUSED
                summary.messages.append("paused because the application was closed")
            self._finish_run(summary)
            return summary
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ run bookkeeping
    def _finish_run(self, summary: RunSummary) -> None:
        with self.db.write() as s:
            run = s.get(Run, summary.run_id)
            if run is not None:
                run.status = summary.status
                run.ended_at = utcnow()
                run.processed = sum(summary.processed.values())
                run.failed = sum(summary.failed.values())
                run.message = "\n".join(summary.messages) or None
            ctrl = s.get(Control, 1)
            # Only the owning run may reset the control row (a newer run may hold it).
            if ctrl is not None and ctrl.active_run_id == summary.run_id:
                ctrl.active_run_id = None
                ctrl.active_stage = None
                if (ctrl.progress or {}).get(PULL_PROGRESS) is not None:
                    ctrl.progress = {**ctrl.progress, PULL_PROGRESS: None}
                # closing the application pauses the pipeline; the next start resumes it
                ctrl.desired_state = PAUSED if summary.status == PAUSED else STOPPED
                ctrl.message = f"run {summary.run_id} {summary.status}"
            # Release anything still claimed by this run (stop / failure).
            s.execute(
                update(StageStatus)
                .where(StageStatus.status == Status.RUNNING, StageStatus.run_id == summary.run_id)
                .values(status=Status.PENDING)
            )

    def _set_stage(self, stage: str | None) -> None:
        self._hb.stage = stage
        self._hb.beat()

    def _update_run_counts(self, run_id: int, summary: RunSummary) -> None:
        with self.db.write() as s:
            run = s.get(Run, run_id)
            if run is not None:
                run.processed = sum(summary.processed.values())
                run.failed = sum(summary.failed.values())

    # ------------------------------------------------------------------ control
    def _wait_if_paused(self, run_id: int, stage: str) -> str:
        """Block while paused. Returns the effective state (``running`` or ``stopped``)."""
        if self._shutdown.is_set():
            return STOPPED
        state = control.desired_state(self.db)
        if state == PAUSED:
            self._set_run_status(run_id, PAUSED)
            log.info("pipeline paused")
            while state == PAUSED:
                if self._shutdown.wait(self.poll_interval):
                    return STOPPED
                state = control.desired_state(self.db)
            self._set_run_status(run_id, RUNNING if state == RUNNING else STOPPED)
            log.info("pipeline %s", "resumed" if state == RUNNING else "stopped")
        return RUNNING if state == RUNNING else STOPPED

    def _interrupted(self) -> bool:
        """Should a long set-up step (a model download) give way? Pause, stop or app exit.

        Asked often and from other threads: the state is read at most every half second.
        """
        if self._shutdown.is_set():
            return True
        now = time.monotonic()
        if now - self._state_at >= 0.5:
            try:
                self._state = control.desired_state(self.db)
            except Exception as exc:  # pragma: no cover - a locked database: ask again
                log.debug("reading the pipeline state failed: %s", exc)
            self._state_at = now
        return self._state != RUNNING

    def _report(self, progress: dict[str, Any] | None) -> None:
        """Progress of a long set-up step, for the UI of any process (``control.progress``).

        At most one short write per second; a new phase and the end are written at once.
        """
        phase = progress.get("phase") if progress else None
        now = time.monotonic()
        if progress is not None and phase == self._reported[0] and now - self._reported[1] < 1:
            return
        self._reported = (phase, now)
        try:
            control.set_progress(self.db, PULL_PROGRESS, progress)
        except Exception as exc:  # the UI misses one update, the download goes on
            log.warning("could not store the download progress: %s", exc)

    def _setup(self, stage: Stage, run_id: int, name: str) -> bool:
        """Set a stage up; ``False`` if the run was stopped meanwhile.

        A long set-up step (downloading a model) gives way to pause and stop requests with
        :class:`StageInterrupted`; after a pause the stage is set up again.
        """
        stage.interrupted = self._interrupted
        stage.report = self._report
        self._state_at = 0.0
        while True:
            try:
                stage.setup()
                return True
            except StageInterrupted as exc:
                log.info("%s", exc)
                if self._wait_if_paused(run_id, name) == STOPPED:
                    return False
                self._state_at = 0.0  # resumed: don't trust the cached state

    def _set_run_status(self, run_id: int, status: str) -> None:
        with self.db.write() as s:
            run = s.get(Run, run_id)
            if run is not None:
                run.status = status

    # ------------------------------------------------------------------ stage loop
    def _eligible(self, s: Any, stage: str, run_id: int, n: int) -> list[int]:
        """The next ``n`` images to claim: pending ones first, then retryable failures.

        Two index range scans in image order; one ORDER BY over both statuses would sort the
        whole stage on every claim (while holding the write lock).
        """

        def ids(*where: Any, limit: int) -> list[int]:
            return list(
                s.scalars(
                    select(StageStatus.image_id)
                    .join(Image, Image.id == StageStatus.image_id)
                    .where(StageStatus.stage == stage, *where, Image.missing.is_(False))
                    .order_by(StageStatus.image_id)
                    .limit(limit)
                )
            )

        out = ids(StageStatus.status == Status.PENDING, limit=n)
        if len(out) < n:
            out += ids(
                StageStatus.status == Status.ERROR,
                StageStatus.attempts < self.settings.pipeline.max_attempts,
                or_(StageStatus.run_id.is_(None), StageStatus.run_id != run_id),
                limit=n - len(out),
            )
        return out

    def _claim(self, stage: str, run_id: int, n: int) -> list[int]:
        with self.db.write() as s:
            ids = self._eligible(s, stage, run_id, n)
            if ids:
                s.execute(
                    update(StageStatus)
                    .where(StageStatus.stage == stage, StageStatus.image_id.in_(ids))
                    .values(status=Status.RUNNING, run_id=run_id, started_at=utcnow())
                )
            return ids

    def _upcoming(self, stage: str, run_id: int, n: int) -> list[int]:
        """What the next claim will most likely return (read-only, for prefetching)."""
        with self.db.session() as s:
            return self._eligible(s, stage, run_id, n)

    def _release(self, stage: str, image_ids: list[int]) -> None:
        if not image_ids:
            return
        with self.db.write() as s:
            s.execute(
                update(StageStatus)
                .where(
                    StageStatus.stage == stage,
                    StageStatus.image_id.in_(image_ids),
                    StageStatus.status == Status.RUNNING,
                )
                .values(status=Status.PENDING)
            )

    def _mark(self, stage: Stage, image_id: int, run_id: int, error: BaseException | None) -> None:
        self._mark_many(stage, run_id, [(image_id, error)])

    def _mark_many(
        self, stage: Stage, run_id: int, results: list[tuple[int, BaseException | None]]
    ) -> None:
        """Record results, but only for rows still claimed by this run (one transaction).

        A requeue, skip or content-change reset that happened meanwhile wins.
        """
        stmts = []
        for image_id, error in results:
            guard = update(StageStatus).where(
                StageStatus.image_id == image_id,
                StageStatus.stage == stage.name,
                StageStatus.status == Status.RUNNING,
                StageStatus.run_id == run_id,
            )
            if error is None:
                stmts.append(
                    guard.values(
                        status=Status.DONE,
                        stage_version=stage.version,
                        last_error=None,
                        finished_at=utcnow(),
                    )
                )
            else:
                tb = "".join(traceback.format_exception_only(type(error), error)).strip()
                stmts.append(
                    guard.values(
                        status=Status.ERROR,
                        attempts=StageStatus.attempts + 1,
                        last_error=tb[-2000:],
                        finished_at=utcnow(),
                    )
                )
        with self.db.write() as s:
            for stmt in stmts:
                s.execute(stmt)

    def _process_one(
        self, stage: Stage, image_id: int, run_id: int, prefetch: _Prefetcher | None = None
    ) -> bool:
        try:
            if prefetch is not None:
                stage.process_prepared(image_id, prefetch.take(image_id))
            else:
                stage.process(image_id)
        except StageAbort:
            raise  # row stays claimed and is released by _finish_run (no attempt counted)
        except Exception as exc:
            log.warning("%s failed for image %s: %s", stage.name, image_id, exc)
            log.debug("traceback", exc_info=True)
            self._mark(stage, image_id, run_id, exc)
            return False
        self._mark(stage, image_id, run_id, None)
        return True

    def _process_batch(
        self, stage: Stage, image_ids: list[int], run_id: int, prefetch: _Prefetcher | None = None
    ) -> int:
        """Process images with one ``process_batch`` (or, with prefetched inputs,
        ``process_prepared_batch``) call; returns the number of failures."""
        try:
            if prefetch is None:
                errors = stage.process_batch(image_ids)
            else:
                errors = stage.process_prepared_batch(image_ids, prefetch.take_many(image_ids))
        except StageAbort:
            raise
        except Exception as exc:
            log.debug("traceback", exc_info=True)
            errors = dict.fromkeys(image_ids, exc)
        for image_id, exc in errors.items():
            log.warning("%s failed for image %s: %s", stage.name, image_id, exc)
        self._mark_many(stage, run_id, [(i, errors.get(i)) for i in image_ids])
        return sum(1 for i in image_ids if i in errors)

    def _prefetch_window(
        self,
        prefetch: _Prefetcher,
        ids: list[int],
        pos: int,
        upcoming: list[int] | None,
        name: str,
        run_id: int,
        room: int | None,
        ahead: int | None = None,
    ) -> list[int] | None:
        """Start preparing the images after ``ids[pos]``, into the next batch if needed.

        ``room``: how many more images the run may claim after this batch (``None`` = no
        limit); ``ahead``: how many images after ``ids[pos]`` (default: ``stage.prefetch``).
        Returns the (cached) guess of the next batch.
        """
        k = prefetch.stage.prefetch if ahead is None else ahead
        window = ids[pos : pos + 1 + k]
        more = 1 + k - len(window)
        if more > 0 and (room is None or room > 0):
            if upcoming is None:
                upcoming = self._upcoming(name, run_id, k if room is None else min(k, room))
            window = window + upcoming[:more]
        prefetch.ahead(window)
        return upcoming

    def _run_stage(
        self, name: str, run_id: int, batch_size: int, limit: int | None, summary: RunSummary
    ) -> bool:
        """Run one stage. Returns ``False`` if the run was stopped."""
        if self._wait_if_paused(run_id, name) == STOPPED:
            summary.status = STOPPED
            return False
        stage = self.stage_factory(name)
        summary.processed.setdefault(name, 0)
        summary.failed.setdefault(name, 0)
        self._set_stage(name)
        set_up = False
        stopped = False
        done = 0
        prefetch = _Prefetcher(stage) if stage.prefetch > 0 and stage.concurrency <= 1 else None
        if stage.redo_outdated:  # results of another model version are of no use
            requeued = control.requeue_outdated(self.db, name, stage.version)
            if requeued:
                log.info("%s: %d outdated results queued again", name, requeued)
        try:
            while limit is None or done < limit:
                n = batch_size if limit is None else min(batch_size, limit - done)
                ids = self._claim(name, run_id, n)
                if not ids:
                    break
                if not set_up:
                    log.info("setting up stage %s", name)
                    try:
                        ready = self._setup(stage, run_id, name)
                    except StageUnavailable as exc:  # skip this stage, run the others
                        log.warning("%s skipped: %s", name, exc)
                        summary.messages.append(f"{name} skipped: {exc}")
                        self._release(name, ids)
                        break
                    if not ready:
                        stopped = True
                        self._release(name, ids)
                        break
                    set_up = True
                remaining = list(ids)
                if prefetch is not None:
                    prefetch.retain(ids)  # the guessed next batch may have changed meanwhile
                if stage.concurrency > 1:
                    if self._wait_if_paused(run_id, name) == STOPPED:
                        stopped = True
                        self._release(name, remaining)
                        break
                    with ThreadPoolExecutor(max_workers=stage.concurrency) as pool:
                        results = list(pool.map(lambda i: self._process_one(stage, i, run_id), ids))
                    for ok in results:
                        summary.processed[name] += 1
                        summary.failed[name] += 0 if ok else 1
                    done += len(ids)
                elif stage.batch > 1:
                    room = None if limit is None else limit - done - len(ids)
                    upcoming = None
                    while remaining:
                        if self._wait_if_paused(run_id, name) == STOPPED:
                            stopped = True
                            break
                        chunk = remaining[: stage.batch]
                        if prefetch is not None:  # this chunk and the next one, meanwhile
                            upcoming = self._prefetch_window(
                                prefetch,
                                remaining,
                                0,
                                upcoming,
                                name,
                                run_id,
                                room,
                                ahead=2 * stage.batch - 1,
                            )
                        failed = self._process_batch(stage, chunk, run_id, prefetch)
                        del remaining[: len(chunk)]
                        summary.processed[name] += len(chunk)
                        summary.failed[name] += failed
                        done += len(chunk)
                    if stopped:
                        self._release(name, remaining)
                        break
                else:
                    room = None if limit is None else limit - done - len(ids)
                    upcoming: list[int] | None = None
                    for pos, image_id in enumerate(ids):
                        if self._wait_if_paused(run_id, name) == STOPPED:
                            stopped = True
                            break
                        if prefetch is not None:
                            upcoming = self._prefetch_window(
                                prefetch, ids, pos, upcoming, name, run_id, room
                            )
                        ok = self._process_one(stage, image_id, run_id, prefetch)
                        remaining.remove(image_id)
                        summary.processed[name] += 1
                        if not ok:
                            summary.failed[name] += 1
                        done += 1
                    if stopped:
                        self._release(name, remaining)
                        break
                self._update_run_counts(run_id, summary)
            if not stopped:
                msg = stage.after_stage(summary.processed[name])
                if msg:
                    summary.messages.append(f"{name}: {msg}")
        finally:
            if prefetch is not None:
                prefetch.close()
            if set_up:
                log.info("tearing down stage %s", name)
                stage.teardown()
        if stopped:
            summary.status = STOPPED
            return False
        return True


class _Prefetcher:
    """Runs ``stage.prepare`` for the next images in worker threads.

    The main thread keeps the GPU (or the VLM server) busy while the workers decode the
    following images. Each result is used at most once; leftovers are discarded.
    """

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self.pool = ThreadPoolExecutor(
            max_workers=stage.prefetch, thread_name_prefix=f"vision-tagger-{stage.name}-prepare"
        )
        self.futures: dict[int, Future[Any]] = {}

    def ahead(self, image_ids: Iterable[int]) -> None:
        for image_id in image_ids:
            if image_id not in self.futures:
                self.futures[image_id] = self.pool.submit(self.stage.prepare, image_id)

    def retain(self, image_ids: Iterable[int]) -> None:
        """Drop prepared images that are not in ``image_ids``."""
        keep = set(image_ids)
        for image_id in [i for i in self.futures if i not in keep]:
            self.futures.pop(image_id).cancel()

    def take(self, image_id: int) -> Any:
        """The prepared inputs of an image (raises what ``prepare`` raised)."""
        future = self.futures.pop(image_id, None)
        if future is None:
            return self.stage.prepare(image_id)
        return future.result()

    def take_many(self, image_ids: Iterable[int]) -> dict[int, Any]:
        """The prepared inputs by image id; what ``prepare`` raised is the value instead."""
        out: dict[int, Any] = {}
        for image_id in image_ids:
            try:
                out[image_id] = self.take(image_id)
            except StageAbort:
                raise
            except Exception as exc:
                out[image_id] = exc
        return out

    def close(self) -> None:
        for future in self.futures.values():
            future.cancel()
        self.futures.clear()
        self.pool.shutdown(wait=True)


class BackgroundRunner:
    """Runs :class:`PipelineRunner` in a daemon thread (used by the web app)."""

    def __init__(self, runner: PipelineRunner) -> None:
        self.runner = runner
        self._thread: threading.Thread | None = None
        self.last_summary: RunSummary | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        stages: list[str] | None = None,
        *,
        batch_size: int | None = None,
        limit: int | None = None,
    ) -> bool:
        if self.running:
            return False
        self.last_error = None

        def target() -> None:
            try:
                self.last_summary = self.runner.run(stages, batch_size=batch_size, limit=limit)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("background run failed")

        self._thread = threading.Thread(target=target, name="vision-tagger-runner", daemon=True)
        self._thread.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


def heartbeat_age(db: Database) -> timedelta | None:
    ctrl = control.get_control(db)
    return None if ctrl.heartbeat_at is None else utcnow() - ctrl.heartbeat_at
