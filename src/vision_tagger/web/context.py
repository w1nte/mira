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
from ..db.cache import VersionedCache
from ..db.session import Database
from ..i18n import Lazy
from ..indexer import ensure_stage_rows
from ..pipeline import control
from ..pipeline.runner import BackgroundRunner, PipelineRunner
from ..pipeline.stages import STAGE_NAMES, default_stage_factory, stage_versions
from ..reveal import FileManager
from ..updates import UpdateChecker
from ..vlm.models import OllamaStatus, ollama_status

log = logging.getLogger(__name__)

#: jobs that replace or remove library data: no pipeline may run meanwhile
LIBRARY_JOBS = ("import", "clear", "remove-missing")


@dataclass
class Job:
    name: str
    status: str = "running"
    # shown to every viewer: messages made in the job thread are ``Lazy`` (translated when shown)
    message: str | Lazy = ""
    progress: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    stop_requested: bool = False


def _failure(exc: Exception) -> str | Lazy:
    """What a failed job shows. Errors meant for users carry a Lazy text, translated when shown:
    ``text`` (vlm.models.PullError) or ``message`` (db.transfer errors)."""
    user_text = getattr(exc, "text", None)
    if not isinstance(user_text, Lazy):
        user_text = getattr(exc, "message", None)
    return user_text if isinstance(user_text, Lazy) else f"{type(exc).__name__}: {exc}"


class JobManager:
    """Runs one maintenance job (index scan, recluster, reference import, export, database
    import, location check, clearing the library, removing missing photos) at a time."""

    def __init__(self) -> None:
        self.current: Job | None = None
        self.history: deque[Job] = deque(maxlen=15)
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def start(self, name: str, fn: Callable[[Job], str | Lazy | None]) -> bool:
        with self._lock:
            if self.busy:
                return False
            job = Job(name=name)
            self.current = job

        def target() -> None:
            try:
                result = fn(job)
                job.status = "done"
                job.message = result or job.message or Lazy("done")
            except Exception as exc:
                job.status = "failed"
                job.message = _failure(exc)
                log.error("job %s failed\n%s", name, traceback.format_exc())
            finally:
                job.finished_at = datetime.now()
                self.history.appendleft(job)

        threading.Thread(target=target, name=f"job-{name}", daemon=True).start()
        return True

    def run_now(self, name: str, fn: Callable[[Job], str | Lazy | None]) -> Job | None:
        """Run a short job in the calling thread (a request), with the same exclusivity as
        :meth:`start`. ``None``: another job runs. Exceptions are re-raised."""
        with self._lock:
            if self.busy:
                return None
            job = Job(name=name)
            self.current = job
        done = False
        try:
            result = fn(job)
            job.message = result or job.message or Lazy("done")
            done = True
        except Exception as exc:
            job.message = _failure(exc)
            raise
        finally:
            job.status = "done" if done else "failed"
            job.finished_at = datetime.now()
            self.history.appendleft(job)
        return job

    def request_stop(self) -> None:
        if self.current is not None:
            self.current.stop_requested = True


class AppContext:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.db = Database(settings.db_path)
        self.db.create_all()
        #: expensive read-only results, valid until the next commit (see ``db.cache``)
        self.cache = VersionedCache(self.db)
        ensure_stage_rows(self.db, settings.pipeline.stages or list(STAGE_NAMES))
        stale_after = max(30.0, settings.pipeline.heartbeat_seconds * 6)
        # read before the recovery below, which resets the desired state
        was_paused = control.desired_state(self.db) == control.PAUSED
        # atomic: does nothing while another process (e.g. a CLI run) sends heartbeats
        recovered = control.recover_stale(self.db, stale_after=stale_after)
        if recovered:
            log.info("recovered %d interrupted items from a previous run", recovered)
        self.runner = BackgroundRunner(
            PipelineRunner(self.db, settings, default_stage_factory(self.db, settings))
        )
        self.jobs = JobManager()
        #: downloads of Ollama models: one at a time, beside the other jobs (an index scan
        #: should not wait for gigabytes of models)
        self.downloads = JobManager()
        if was_paused:
            self._resume_paused()
        self.uploads: Any = None
        self.ingest: Any = None
        if settings.remote_mode:
            from ..uploads import UploadStore
            from .ingest import Ingest

            self.uploads = UploadStore(settings)
            self.uploads.cleanup()
            self.ingest = Ingest(self)
        self._models: tuple[float, OllamaStatus | None] = (0.0, None)
        self._versions: tuple[float, dict[str, str]] = (-1.0, {})
        self._search: Any = None
        self._search_lock = threading.Lock()
        #: the update check (``updates.check``): in the background, about once a day
        self.updates = UpdateChecker(settings)
        self.updates.start()
        #: "Show in folder" (local mode); tests replace its launcher
        self.file_manager = FileManager()

    def _resume_paused(self) -> None:
        """Continue the pipeline that was paused when the app last closed."""
        if not self.settings.pipeline.resume_on_start or self.runner.running:
            return
        if control.is_runner_alive(self.db):
            return  # another process (e.g. a CLI run) holds the pipeline
        last = next(iter(control.latest_runs(self.db, limit=1)), None)
        stages = [st for st in (last.stages or "").split(",") if st] if last else None
        log.info("continuing the paused pipeline")
        self.runner.start(
            stages or None,
            batch_size=last.batch_size if last else None,
            limit=last.limit if last else None,
        )

    def shutdown_pipeline(self, timeout: float = 60.0) -> None:
        """Application exit: pause a running pipeline (it continues on the next start)."""
        if not self.runner.running:
            return
        control.request_state(self.db, control.PAUSED)
        self.runner.runner.shutdown()
        self.runner.join(timeout)  # lets the current image finish

    @property
    def importing(self) -> bool:
        """A database import is replacing the library (no pipeline may start meanwhile)."""
        return self.library_job == "import"

    @property
    def library_job(self) -> str | None:
        """The running job that replaces or removes library data (import, clearing, removing
        missing photos), during which no pipeline may start; else ``None``."""
        job = self.jobs.current
        if job is not None and job.status == "running" and job.name in LIBRARY_JOBS:
            return job.name
        return None

    def forget_library(self) -> None:
        """After a database import, clearing the library or removing photos: drop every
        in-memory cache keyed by ids (they now mean other photos, or none). Disk caches are
        cleared by ``db.transfer.clear_cache_dirs`` (``removal`` deletes single files)."""
        self.cache.clear()
        with self._search_lock:
            self._search = None  # its embedding matrix and value counts
        self._health = (0.0, {})

    @property
    def search(self) -> Any:
        """The (lazily created) natural-language search engine."""
        with self._search_lock:
            if self._search is None:
                from ..search.engine import SearchEngine

                self._search = SearchEngine(self.db, self.settings)
            return self._search

    def stage_versions(self) -> dict[str, str]:
        """Current stage versions; re-read only when the vocabulary file changes."""
        from ..vlm.schema import default_vocabulary_path

        path = self.settings.vlm.vocabulary_file or default_vocabulary_path()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime != self._versions[0] or not self._versions[1]:
            self._versions = (mtime, stage_versions(self.settings))
        return self._versions[1]

    def models_status(self, max_age: float = 10.0) -> OllamaStatus:
        """Ollama and the models this configuration needs (cached for ``max_age`` seconds)."""
        ts, cached = self._models
        if cached is not None and time.monotonic() - ts < max_age:
            return cached
        status = ollama_status(self.settings)
        self._models = (time.monotonic(), status)
        return status

    def forget_models_status(self) -> None:
        self._models = (0.0, None)

    def onnx_providers(self) -> list[str]:
        try:
            import onnxruntime as ort

            return list(ort.get_available_providers())
        except Exception:  # pragma: no cover - onnxruntime missing
            return []

    def close(self) -> None:
        self.updates.stop()
        self.downloads.request_stop()  # Ollama keeps the layers; the next download continues
        if self.ingest is not None:
            self.ingest.close()
        self.db.dispose()
