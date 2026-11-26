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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import or_, select, update

from ..config import Settings
from ..db.models import Control, Image, Run, StageStatus, Status, utcnow
from ..db.session import Database
from . import control
from .control import PAUSED, RUNNING, STOPPED, RunnerBusyError
from .stages.base import Stage, StageAbort

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
                ctrl.desired_state = STOPPED
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
        state = control.desired_state(self.db)
        if state == PAUSED:
            self._set_run_status(run_id, PAUSED)
            log.info("pipeline paused")
            while state == PAUSED:
                time.sleep(self.poll_interval)
                state = control.desired_state(self.db)
            self._set_run_status(run_id, RUNNING if state == RUNNING else STOPPED)
            log.info("pipeline %s", "resumed" if state == RUNNING else "stopped")
        return RUNNING if state == RUNNING else STOPPED

    def _set_run_status(self, run_id: int, status: str) -> None:
        with self.db.write() as s:
            run = s.get(Run, run_id)
            if run is not None:
                run.status = status

    # ------------------------------------------------------------------ stage loop
    def _claim(self, stage: str, run_id: int, n: int) -> list[int]:
        max_attempts = self.settings.pipeline.max_attempts
        with self.db.write() as s:
            eligible = or_(
                StageStatus.status == Status.PENDING,
                (StageStatus.status == Status.ERROR)
                & (StageStatus.attempts < max_attempts)
                & (or_(StageStatus.run_id.is_(None), StageStatus.run_id != run_id)),
            )
            ids = list(
                s.scalars(
                    select(StageStatus.image_id)
                    .join(Image, Image.id == StageStatus.image_id)
                    .where(StageStatus.stage == stage, eligible, Image.missing.is_(False))
                    .order_by(StageStatus.status.desc(), StageStatus.image_id)  # pending first
                    .limit(n)
                )
            )
            if ids:
                s.execute(
                    update(StageStatus)
                    .where(StageStatus.stage == stage, StageStatus.image_id.in_(ids))
                    .values(status=Status.RUNNING, run_id=run_id, started_at=utcnow())
                )
            return ids

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
        """Record the result, but only if the row is still claimed by this run.

        A requeue, skip or content-change reset that happened meanwhile wins.
        """
        guard = update(StageStatus).where(
            StageStatus.image_id == image_id,
            StageStatus.stage == stage.name,
            StageStatus.status == Status.RUNNING,
            StageStatus.run_id == run_id,
        )
        if error is None:
            stmt = guard.values(
                status=Status.DONE,
                stage_version=stage.version,
                last_error=None,
                finished_at=utcnow(),
            )
        else:
            tb = "".join(traceback.format_exception_only(type(error), error)).strip()
            stmt = guard.values(
                status=Status.ERROR,
                attempts=StageStatus.attempts + 1,
                last_error=tb[-2000:],
                finished_at=utcnow(),
            )
        with self.db.write() as s:
            s.execute(stmt)

    def _process_one(self, stage: Stage, image_id: int, run_id: int) -> bool:
        try:
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
        try:
            while limit is None or done < limit:
                n = batch_size if limit is None else min(batch_size, limit - done)
                ids = self._claim(name, run_id, n)
                if not ids:
                    break
                if not set_up:
                    log.info("setting up stage %s", name)
                    stage.setup()
                    set_up = True
                remaining = list(ids)
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
                else:
                    for image_id in ids:
                        if self._wait_if_paused(run_id, name) == STOPPED:
                            stopped = True
                            break
                        ok = self._process_one(stage, image_id, run_id)
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
            if set_up:
                log.info("tearing down stage %s", name)
                stage.teardown()
        if stopped:
            summary.status = STOPPED
            return False
        return True


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
