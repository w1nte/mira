"""Stage protocol shared by all pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ...config import Settings
from ...db.session import Database


class StageAbort(RuntimeError):
    """Raised by a stage when continuing makes no sense (e.g. the VLM server is unreachable).

    The runner stops the run, releases the claimed images without counting an attempt and
    reports the message.
    """


class Stage(ABC):
    """A unit of per-image work.

    ``process`` must be idempotent: it may be called again for the same image after a crash,
    so implementations replace earlier results for the image instead of appending to them.
    """

    name: ClassVar[str]
    #: Parallel ``process`` calls inside a batch (e.g. concurrent HTTP requests to Ollama).
    concurrency: int = 1

    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    @property
    def version(self) -> str:
        """Results produced with a different version are considered outdated."""
        return "1"

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Load models etc. Called once before the first batch of a run."""

    @abstractmethod
    def process(self, image_id: int) -> None:
        """Process one image and persist the results."""

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
