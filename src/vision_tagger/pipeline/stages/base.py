"""Stage protocol shared by all pipeline stages."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from ...config import Settings
from ...db.session import Database


class StageAbort(RuntimeError):
    """Raised by a stage when continuing makes no sense (e.g. the VLM server is unreachable).

    The runner stops the run, releases the claimed images without counting an attempt and
    reports the message.
    """


class StageInterrupted(RuntimeError):
    """Raised by :meth:`Stage.setup` when a long set-up step (a model download) gave way to a
    pause or stop request (:attr:`Stage.interrupted`).

    The runner waits while the pipeline is paused and then sets the stage up again; a stop ends
    the run. Nothing was processed, so nothing is lost.
    """


class StageUnavailable(RuntimeError):
    """Raised by :meth:`Stage.setup` when the stage can't run now but the others can (e.g. its
    model can't be downloaded without internet).

    The runner releases the claimed images without counting an attempt, notes the message and
    goes on with the next stage; the next run tries again.
    """


class Stage(ABC):
    """A unit of per-image work.

    ``process`` must be idempotent: it may be called again for the same image after a crash,
    so implementations replace earlier results for the image instead of appending to them.
    """

    name: ClassVar[str]
    #: Parallel ``process`` calls inside a batch (e.g. concurrent HTTP requests to Ollama).
    concurrency: int = 1
    #: Images whose :meth:`prepare` runs ahead in worker threads while the current image is
    #: processed, so the GPU or the VLM server does not wait for decoding (0 = off).
    prefetch: int = 0
    #: Images handed to :meth:`process_batch` at once (1 = one :meth:`process` call each).
    #: With ``prefetch`` too, :meth:`prepare` runs ahead for this batch and the next one, and
    #: :meth:`process_prepared_batch` gets the results.
    batch: int = 1
    #: Results of another :attr:`version` are queued again whenever the stage runs (they are
    #: of no use, e.g. vectors of another model), instead of waiting for "re-queue outdated".
    redo_outdated: ClassVar[bool] = False

    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        #: set by the runner: True once the run is to pause or stop. Long set-up steps (model
        #: downloads) ask it, also from other threads, and raise :class:`StageInterrupted`.
        self.interrupted: Callable[[], bool] = lambda: False
        #: set by the runner: progress of a long set-up step for the UI (``None`` clears it)
        self.report: Callable[[dict[str, Any] | None], None] = lambda _progress: None

    @property
    def version(self) -> str:
        """Results produced with a different version are considered outdated."""
        return "1"

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Load models etc. Called once before the first batch of a run.

        May raise :class:`StageInterrupted` (then it is called again after a pause).
        """

    @abstractmethod
    def process(self, image_id: int) -> None:
        """Process one image and persist the results."""

    def prepare(self, image_id: int) -> Any:
        """Read and decode an image's inputs (stages with ``prefetch``).

        Runs in a worker thread, ahead of :meth:`process_prepared`: thread-safe, no database
        writes. Exceptions are raised again when the image is processed.
        """
        raise NotImplementedError

    def process_prepared(self, image_id: int, prepared: Any) -> None:
        """Process one image with the result of :meth:`prepare`."""
        self.process(image_id)

    def process_batch(self, image_ids: list[int]) -> dict[int, Exception]:
        """Process several images (stages with ``batch``); returns the errors by image id."""
        errors: dict[int, Exception] = {}
        for image_id in image_ids:
            try:
                self.process(image_id)
            except StageAbort:
                raise
            except Exception as exc:
                errors[image_id] = exc
        return errors

    def process_prepared_batch(
        self, image_ids: list[int], prepared: dict[int, Any]
    ) -> dict[int, Exception]:
        """Process several images with the results of :meth:`prepare` (an exception that
        ``prepare`` raised is the value instead); returns the errors by image id."""
        errors: dict[int, Exception] = {}
        for image_id in image_ids:
            value = prepared.get(image_id)
            if isinstance(value, Exception):
                errors[image_id] = value
                continue
            try:
                self.process_prepared(image_id, value)
            except StageAbort:
                raise
            except Exception as exc:
                errors[image_id] = exc
        return errors

    def teardown(self) -> None:  # noqa: B027 - optional hook
        """Free models / VRAM. Called after the stage finished (or was stopped)."""

    def after_stage(self, processed: int) -> str | None:
        """Called after all claimed images were processed (e.g. to recluster faces).

        Returns an optional message for the run log.
        """
        return None

    def info(self) -> dict[str, object]:
        """Extra diagnostics shown on the dashboard."""
        return {}


def decode_workers(limit: int = 6) -> int:
    """Threads for decoding images ahead of the GPU: leave a core for the main thread."""
    return max(1, min(limit, (os.cpu_count() or 2) - 1))
