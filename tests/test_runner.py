from __future__ import annotations

import threading
import time

import pytest
from sqlalchemy import select

from vision_tagger.db.models import Run, StageStatus, Status
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline import control
from vision_tagger.pipeline.runner import BackgroundRunner, PipelineRunner, RunnerBusyError
from vision_tagger.pipeline.stages.base import Stage

from .conftest import make_image


class FakeStage(Stage):
    name = "faces"

    def __init__(self, db, settings, *, fail_ids=(), delay=0.0, on_process=None, name=None):
        super().__init__(db, settings)
        if name:
            self.name = name  # type: ignore[misc]
        self.fail_ids = set(fail_ids)
        self.delay = delay
        self.on_process = on_process
        self.processed: list[int] = []
        self.setup_calls = 0
        self.teardown_calls = 0
        self.after_calls: list[int] = []

    def setup(self):
        self.setup_calls += 1

    def teardown(self):
        self.teardown_calls += 1

    def after_stage(self, processed):
        self.after_calls.append(processed)
        return None

    def process(self, image_id):
        if self.on_process:
            self.on_process(image_id)
        time.sleep(self.delay)
        if image_id in self.fail_ids:
            raise RuntimeError(f"boom {image_id}")
        self.processed.append(image_id)


@pytest.fixture
def indexed(db, settings, photos):
    for i in range(10):
        make_image(photos / f"img{i}.jpg", seed=i)
    Indexer(db, settings, stages=["faces", "vlm"]).run()
    return db


def statuses(db, stage="faces"):
    with db.session() as s:
        return {
            r.image_id: r.status
            for r in s.scalars(select(StageStatus).where(StageStatus.stage == stage))
        }


def test_full_run_marks_done_and_errors(indexed, settings):
    stages = {
        "faces": FakeStage(indexed, settings, fail_ids={3}),
        "vlm": FakeStage(indexed, settings, name="vlm"),
    }
    summary = PipelineRunner(indexed, settings, stages.__getitem__).run()
    assert summary.status == "completed"
    assert summary.processed == {"faces": 10, "vlm": 10}
    assert summary.failed == {"faces": 1, "vlm": 0}
    st = statuses(indexed)
    assert st[3] == Status.ERROR
    assert sum(v == Status.DONE for v in st.values()) == 9
    assert stages["faces"].setup_calls == 1 and stages["faces"].teardown_calls == 1
    assert stages["faces"].after_calls == [10]
    with indexed.session() as s:
        row = s.get(StageStatus, (3, "faces"))
        assert row.attempts == 1 and "boom 3" in row.last_error
        assert s.get(StageStatus, (1, "faces")).stage_version == "1"
    assert control.get_control(indexed).desired_state == control.STOPPED


def test_errors_retried_in_next_run_until_max_attempts(indexed, settings):
    settings.pipeline.max_attempts = 2
    stage = FakeStage(indexed, settings, fail_ids={5})
    runner = PipelineRunner(indexed, settings, lambda _n: stage)
    runner.run(["faces"])
    runner.run(["faces"])
    runner.run(["faces"])
    with indexed.session() as s:
        row = s.get(StageStatus, (5, "faces"))
        assert row.status == Status.ERROR and row.attempts == 2
    assert control.retry_errors(indexed, "faces") == 1
    stage.fail_ids.clear()
    runner.run(["faces"])
    assert statuses(indexed)[5] == Status.DONE


def test_limit_processes_only_n_images(indexed, settings):
    stage = FakeStage(indexed, settings)
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"], limit=3)
    assert summary.processed["faces"] == 3
    st = statuses(indexed)
    assert sum(v == Status.DONE for v in st.values()) == 3
    assert sum(v == Status.PENDING for v in st.values()) == 7


def test_stop_releases_claimed_rows(indexed, settings):
    def on_process(image_id):
        if image_id == 2:
            control.request_state(indexed, control.STOPPED)

    stage = FakeStage(indexed, settings, on_process=on_process)
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"], batch_size=4)
    assert summary.status == "stopped"
    st = statuses(indexed)
    assert [i for i, v in st.items() if v == Status.DONE] == [1, 2]
    assert Status.RUNNING not in st.values()
    assert stage.teardown_calls == 1


def test_pause_and_resume(indexed, settings):
    first = threading.Event()

    def on_process(image_id):
        if image_id == 1:
            control.request_state(indexed, control.PAUSED)
            first.set()

    stage = FakeStage(indexed, settings, on_process=on_process)
    runner = PipelineRunner(indexed, settings, lambda _n: stage)
    runner.poll_interval = 0.05
    bg = BackgroundRunner(runner)
    assert bg.start(["faces"])
    assert first.wait(5)
    time.sleep(0.4)
    assert bg.running
    assert stage.processed == [1]  # paused after the first image
    with indexed.session() as s:
        assert s.scalars(select(Run.status).order_by(Run.id.desc())).first() == control.PAUSED
    control.request_state(indexed, control.RUNNING)
    bg.join(10)
    assert not bg.running
    assert bg.last_summary.status == "completed"
    assert len(stage.processed) == 10


def test_crash_recovery_resets_running_rows(indexed, settings):
    # Simulate a crashed run: rows left in "running", control says a run is active but stale.
    with indexed.write() as s:
        for row in s.scalars(select(StageStatus).where(StageStatus.stage == "faces").limit(3)):
            row.status = Status.RUNNING
            row.run_id = 99
    stage = FakeStage(indexed, settings)
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"])
    assert summary.status == "completed"
    assert set(statuses(indexed).values()) == {Status.DONE}


def test_refuses_second_runner_with_fresh_heartbeat(indexed, settings):
    block = threading.Event()
    stage = FakeStage(indexed, settings, on_process=lambda _i: block.wait(5))
    bg = BackgroundRunner(PipelineRunner(indexed, settings, lambda _n: stage))
    bg.start(["faces"])
    deadline = time.time() + 10
    while control.get_control(indexed).active_run_id is None and time.time() < deadline:
        time.sleep(0.05)
    other = PipelineRunner(indexed, settings, lambda _n: FakeStage(indexed, settings))
    with pytest.raises(RunnerBusyError):
        other.run(["faces"])
    control.request_state(indexed, control.STOPPED)
    block.set()
    bg.join(10)


def test_requeue_outdated(indexed, settings):
    stage = FakeStage(indexed, settings)
    PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"])
    assert control.outdated_counts(indexed, {"faces": "2"}) == {"faces": 10}
    assert control.requeue_outdated(indexed, "faces", "2") == 10
    assert set(statuses(indexed).values()) == {Status.PENDING}


def test_concurrent_stage(indexed, settings):
    stage = FakeStage(indexed, settings, delay=0.05, fail_ids={4})
    stage.concurrency = 4
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"], batch_size=5)
    assert summary.processed["faces"] == 10 and summary.failed["faces"] == 1
    assert statuses(indexed)[4] == Status.ERROR


def test_heartbeat_continues_during_long_image(indexed, settings):
    """A single slow image must not make the run look dead (other processes would recover it)."""
    settings.pipeline.heartbeat_seconds = 0.1
    seen = []

    def on_process(_image_id):
        seen.append(control.get_control(indexed).heartbeat_at)
        time.sleep(0.6)
        seen.append(control.get_control(indexed).heartbeat_at)

    stage = FakeStage(indexed, settings, on_process=on_process)
    PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"], limit=1)
    assert seen[1] > seen[0]


def test_try_acquire_refuses_live_runner(indexed, settings):
    run_id, _ = control.try_acquire(indexed, ["faces"], 4, None, stale_after=30)
    with pytest.raises(RunnerBusyError):
        control.try_acquire(indexed, ["faces"], 4, None, stale_after=30)
    # a stale heartbeat (crashed runner) is taken over
    with indexed.write() as s:
        from datetime import timedelta

        from vision_tagger.db.models import Control, utcnow

        s.get(Control, 1).heartbeat_at = utcnow() - timedelta(minutes=5)
    new_id, _ = control.try_acquire(indexed, ["faces"], 4, None, stale_after=30)
    assert new_id != run_id
    # recover_stale with a fresh heartbeat leaves the live run alone
    assert control.recover_stale(indexed, stale_after=30) == 0
    assert control.get_control(indexed).active_run_id == new_id


def test_finished_old_run_does_not_reset_newer_run(indexed, settings):
    from vision_tagger.pipeline.runner import RunSummary

    runner = PipelineRunner(indexed, settings, lambda _n: FakeStage(indexed, settings))
    old_id, _ = control.try_acquire(indexed, ["faces"], 4, None, stale_after=30)
    with indexed.write() as s:
        from vision_tagger.db.models import Control

        s.get(Control, 1).heartbeat_at = None  # old runner died
    new_id, _ = control.try_acquire(indexed, ["faces"], 4, None, stale_after=30)
    runner._finish_run(RunSummary(run_id=old_id, status="failed"))
    ctrl = control.get_control(indexed)
    assert ctrl.active_run_id == new_id and ctrl.desired_state == control.RUNNING


def test_mark_does_not_overwrite_requeued_row(indexed, settings):
    def on_process(image_id):
        if image_id == 1:  # user re-queues the image while it is being processed
            with indexed.write() as s:
                row = s.get(StageStatus, (1, "faces"))
                row.status = Status.PENDING
                row.run_id = None

    stage = FakeStage(indexed, settings, on_process=on_process)
    PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"], limit=1)
    assert statuses(indexed)[1] == Status.PENDING


def test_missing_images_are_not_claimed(indexed, settings):
    from vision_tagger.db.models import Image

    with indexed.write() as s:
        s.get(Image, 2).missing = True
    stage = FakeStage(indexed, settings)
    PipelineRunner(indexed, settings, lambda _n: stage).run(["faces"])
    assert 2 not in stage.processed
    assert statuses(indexed)[2] == Status.PENDING
