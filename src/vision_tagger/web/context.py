"""Application context shared by all web routes."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import Settings
from ..db.session import Database
from ..indexer import ensure_stage_rows
from ..pipeline import control
from ..pipeline.runner import BackgroundRunner, PipelineRunner
from ..pipeline.stages import STAGE_NAMES, default_stage_factory

log = logging.getLogger(__name__)


@dataclass
class Job:
    name: str
    status: str = "running"
    message: str = ""
    progress: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    stop_requested: bool = False


class JobManager:
    """Runs one maintenance job (index scan, recluster, reference import, export) at a time."""

    def __init__(self) -> None:
        self.current: Job | None = None
        self.history: deque[Job] = deque(maxlen=15)
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def start(self, name: str, fn: Callable[[Job], str | None]) -> bool:
        with self._lock:
            if self.busy:
                return False
            job = Job(name=name)
            self.current = job

        def target() -> None:
            try:
                result = fn(job)
                job.status = "done"
                job.message = result or job.message or "done"
            except Exception as exc:
                job.status = "failed"
                job.message = f"{type(exc).__name__}: {exc}"
                log.error("job %s failed\n%s", name, traceback.format_exc())
            finally:
                job.finished_at = datetime.now()
                self.history.appendleft(job)

        threading.Thread(target=target, name=f"job-{name}", daemon=True).start()
        return True

    def request_stop(self) -> None:
        if self.current is not None:
            self.current.stop_requested = True


class AppContext:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.db = Database(settings.db_path)
        self.db.create_all()
        ensure_stage_rows(self.db, settings.pipeline.stages or list(STAGE_NAMES))
        stale_after = max(30.0, settings.pipeline.heartbeat_seconds * 6)
        # atomic: does nothing while another process (e.g. a CLI run) sends heartbeats
        recovered = control.recover_stale(self.db, stale_after=stale_after)
        if recovered:
            log.info("recovered %d interrupted items from a previous run", recovered)
        self.runner = BackgroundRunner(
            PipelineRunner(self.db, settings, default_stage_factory(self.db, settings))
        )
        self.jobs = JobManager()
        self._health: tuple[float, dict[str, Any]] = (0.0, {})

    def vlm_health(self, max_age: float = 30.0) -> dict[str, Any]:
        ts, cached = self._health
        if cached and time.monotonic() - ts < max_age:
            return cached
        from ..vlm.backend import make_backend
        from ..vlm.schema import Vocabulary

        backend = None
        try:
            backend = make_backend(
                self.settings.vlm, Vocabulary.load(self.settings.vlm.vocabulary_file)
            )
            health = backend.health()
        except Exception as exc:  # pragma: no cover - defensive
            health = {"ok": False, "error": str(exc)}
        finally:
            if backend is not None:
                backend.close()
        self._health = (time.monotonic(), health)
        return health

    def onnx_providers(self) -> list[str]:
        try:
            import onnxruntime as ort

            return list(ort.get_available_providers())
        except Exception:  # pragma: no cover - onnxruntime missing
            return []

    def close(self) -> None:
        self.db.dispose()
