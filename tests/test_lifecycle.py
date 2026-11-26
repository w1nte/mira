from __future__ import annotations

import time

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from vision_tagger.db.models import Control, Run, StageStatus, Status, utcnow
from vision_tagger.db.session import Database
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline import control
from vision_tagger.web.app import create_app

from .conftest import make_image


def _paused_library(settings, photos, stages="vlm"):
    for i in range(4):
        make_image(photos / f"p{i}.jpg", seed=i)
    db = Database(settings.db_path)
    db.create_all()
    Indexer(db, settings, stages=["vlm"]).run()
    with db.write() as s:  # what an app closed during a run leaves behind
        run = Run(stages=stages, batch_size=2, limit=None, status=control.PAUSED)
        s.add(run)
        s.get(Control, 1).desired_state = control.PAUSED
    db.dispose()


def _vlm_done(app) -> int:
    with app.state.ctx.db.session() as s:
        return (
            s.scalar(
                select(func.count())
                .select_from(StageStatus)
                .where(StageStatus.stage == "vlm", StageStatus.status == Status.DONE)
            )
            or 0
        )


def test_paused_pipeline_continues_on_start(settings, photos):
    _paused_library(settings, photos)
    app = create_app(settings)
    with TestClient(app):
        deadline = time.time() + 30
        while _vlm_done(app) < 4:
            assert time.time() < deadline, "the paused pipeline did not continue"
            time.sleep(0.1)
        runs = control.latest_runs(app.state.ctx.db, limit=1)
        assert runs[0].stages == "vlm" and runs[0].batch_size == 2  # same settings as before


def test_resume_can_be_switched_off(settings, photos):
    _paused_library(settings, photos)
    settings.pipeline.resume_on_start = False
    app = create_app(settings)
    with TestClient(app):
        time.sleep(0.5)
        assert not app.state.ctx.runner.running and _vlm_done(app) == 0


def test_no_resume_while_another_process_runs(settings, photos):
    _paused_library(settings, photos)
    db = Database(settings.db_path)
    with db.write() as s:  # e.g. a CLI run in another terminal, sending heartbeats
        ctrl = s.get(Control, 1)
        ctrl.active_run_id = s.scalar(select(Run.id))
        ctrl.heartbeat_at = utcnow()
    db.dispose()
    app = create_app(settings)
    with TestClient(app):
        assert not app.state.ctx.runner.running
