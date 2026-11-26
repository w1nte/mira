"""DB-backed pipeline control and status queries (shared by the CLI and the web UI)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.orm import Session

from ..db.models import Control, Run, StageStatus, Status, utcnow
from ..db.session import Database

RUNNING, PAUSED, STOPPED = "running", "paused", "stopped"


def get_control(db: Database) -> Control:
    with db.session() as s:
        ctrl = s.get(Control, 1)
        if ctrl is None:  # pragma: no cover - created by create_all
            ctrl = Control(id=1, desired_state=STOPPED, progress={})
        s.expunge_all()
        return ctrl


def request_state(db: Database, state: str) -> None:
    if state not in (RUNNING, PAUSED, STOPPED):
        raise ValueError(f"invalid state {state!r}")
    with db.write() as s:
        ctrl = s.get(Control, 1)
        assert ctrl is not None
        ctrl.desired_state = state


def desired_state(db: Database) -> str:
    with db.session() as s:
        return s.scalar(select(Control.desired_state).where(Control.id == 1)) or STOPPED


def set_progress(db: Database, key: str, value: Any) -> None:
    with db.write() as s:
        ctrl = s.get(Control, 1)
        assert ctrl is not None
        progress = dict(ctrl.progress or {})
        progress[key] = value
        ctrl.progress = progress


def is_runner_alive(db: Database, stale_after: float = 30.0) -> bool:
    return _alive(get_control(db), stale_after)


# --------------------------------------------------------------------------- maintenance
def retry_errors(db: Database, stage: str | None = None) -> int:
    with db.write() as s:
        stmt = update(StageStatus).where(StageStatus.status == Status.ERROR)
        if stage:
            stmt = stmt.where(StageStatus.stage == stage)
        res = s.execute(stmt.values(status=Status.PENDING, attempts=0, run_id=None))
        return res.rowcount or 0


def skip_errors(db: Database, stage: str | None = None) -> int:
    with db.write() as s:
        stmt = update(StageStatus).where(StageStatus.status == Status.ERROR)
        if stage:
            stmt = stmt.where(StageStatus.stage == stage)
        return s.execute(stmt.values(status=Status.SKIPPED)).rowcount or 0


def requeue(db: Database, stage: str, image_ids: list[int] | None = None) -> int:
    """Set rows back to pending (all images of the stage, or a selection)."""
    with db.write() as s:
        stmt = update(StageStatus).where(StageStatus.stage == stage)
        if image_ids is not None:
            stmt = stmt.where(StageStatus.image_id.in_(image_ids))
        stmt = stmt.where(StageStatus.status != Status.RUNNING)
        return (
            s.execute(stmt.values(status=Status.PENDING, attempts=0, last_error=None)).rowcount or 0
        )


def mark_pending(s: Session, stage: str, image_ids: Iterable[int]) -> None:
    """Re-queue ``stage`` for these images inside an open session (skips rows in work)."""
    from ..db.util import chunked

    for part in chunked(sorted(set(image_ids))):
        s.execute(
            update(StageStatus)
            .where(
                StageStatus.stage == stage,
                StageStatus.image_id.in_(part),
                StageStatus.status != Status.RUNNING,
            )
            .values(status=Status.PENDING, attempts=0, last_error=None)
        )


def mark_done(s: Session, stage: str, image_ids: Iterable[int], version: str) -> None:
    """Record ``stage`` as done for these images (work another stage did on the way)."""
    from ..db.util import chunked

    for part in chunked(sorted(set(image_ids))):
        s.execute(
            update(StageStatus)
            .where(
                StageStatus.stage == stage,
                StageStatus.image_id.in_(part),
                StageStatus.status != Status.RUNNING,
            )
            .values(
                status=Status.DONE,
                attempts=0,
                last_error=None,
                stage_version=version,
                finished_at=utcnow(),
            )
        )


def requeue_outdated(db: Database, stage: str, version: str) -> int:
    with db.write() as s:
        stmt = (
            update(StageStatus)
            .where(
                StageStatus.stage == stage,
                StageStatus.status == Status.DONE,
                (StageStatus.stage_version.is_(None)) | (StageStatus.stage_version != version),
            )
            .values(status=Status.PENDING, attempts=0)
        )
        return s.execute(stmt).rowcount or 0


class RunnerBusyError(RuntimeError):
    """Another runner (this or another process) holds the pipeline."""


def _alive(ctrl: Control | None, stale_after: float) -> bool:
    if ctrl is None or ctrl.active_run_id is None or ctrl.heartbeat_at is None:
        return False
    return utcnow() - ctrl.heartbeat_at < timedelta(seconds=stale_after)


def _recover(s: Session) -> int:
    n = (
        s.execute(
            update(StageStatus)
            .where(StageStatus.status == Status.RUNNING)
            .values(status=Status.PENDING)
        ).rowcount
        or 0
    )
    s.execute(
        update(Run)
        .where(Run.status.in_([RUNNING, PAUSED]))
        .values(status="interrupted", ended_at=utcnow())
    )
    ctrl = s.get(Control, 1)
    if ctrl is not None:
        ctrl.active_run_id = None
        ctrl.active_stage = None
        ctrl.desired_state = STOPPED
    return n


def recover_stale(db: Database, stale_after: float | None = None) -> int:
    """Reset rows stuck in ``running`` (from a crashed or killed run) to ``pending``.

    With ``stale_after`` nothing happens while a runner still sends heartbeats; the check and
    the reset happen in one ``BEGIN IMMEDIATE`` transaction, so they cannot race a starting run.
    """
    with db.write() as s:
        if stale_after is not None and _alive(s.get(Control, 1), stale_after):
            return 0
        return _recover(s)


def try_acquire(
    db: Database, stages: list[str], batch_size: int, limit: int | None, stale_after: float
) -> tuple[int, int]:
    """Atomically check that no runner is alive, recover stale rows and register a new run.

    Returns ``(run_id, recovered_rows)``; raises :class:`RunnerBusyError` if a live runner
    holds the pipeline.
    """
    with db.write() as s:
        ctrl = s.get(Control, 1)
        if _alive(ctrl, stale_after):
            raise RunnerBusyError("another runner is active on this database")
        recovered = _recover(s)
        run = Run(
            stages=",".join(stages),
            batch_size=batch_size,
            limit=limit,
            status=RUNNING,
            heartbeat_at=utcnow(),
        )
        s.add(run)
        s.flush()
        ctrl = s.get(Control, 1)
        assert ctrl is not None
        ctrl.desired_state = RUNNING
        ctrl.active_run_id = run.id
        ctrl.heartbeat_at = utcnow()
        ctrl.active_stage = None
        ctrl.message = None
        return run.id, recovered


# --------------------------------------------------------------------------- status
def stage_counts(db: Database) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    with db.session() as s:
        for stage, status, n in s.execute(
            select(StageStatus.stage, StageStatus.status, func.count()).group_by(
                StageStatus.stage, StageStatus.status
            )
        ):
            counts.setdefault(stage, dict.fromkeys(Status.ALL, 0))[status] = n
    for c in counts.values():
        c["total"] = sum(c[k] for k in Status.ALL)
    return counts


def outdated_counts(db: Database, versions: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    with db.session() as s:
        for stage, version in versions.items():
            out[stage] = (
                s.scalar(
                    select(func.count())
                    .select_from(StageStatus)
                    .where(
                        and_(
                            StageStatus.stage == stage,
                            StageStatus.status == Status.DONE,
                            (StageStatus.stage_version.is_(None))
                            | (StageStatus.stage_version != version),
                        )
                    )
                )
                or 0
            )
    return out


def recent_errors(db: Database, limit: int = 50) -> list[dict[str, Any]]:
    from ..db.models import Image

    with db.session() as s:
        rows = s.execute(
            select(StageStatus, Image.path)
            .join(Image, Image.id == StageStatus.image_id)
            .where(StageStatus.status == Status.ERROR)
            .order_by(StageStatus.finished_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "image_id": st.image_id,
                "stage": st.stage,
                "attempts": st.attempts,
                "error": st.last_error,
                "path": path,
                "at": st.finished_at,
            }
            for st, path in rows
        ]


def latest_runs(db: Database, limit: int = 10) -> list[Run]:
    with db.session() as s:
        runs = list(s.scalars(select(Run).order_by(Run.id.desc()).limit(limit)))
        s.expunge_all()
        return runs
