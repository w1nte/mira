"""Ollama models in stage set-up: a missing model is downloaded first (``vlm.auto_pull``)."""

from __future__ import annotations

import logging
import time
from typing import Any

from ...vlm.models import PullCancelled, PullError, PullProgress, fmt_size, normalize
from .base import Stage, StageAbort, StageInterrupted

log = logging.getLogger(__name__)


class _ProgressLog:
    """Logs a download every 10 % (or every 30 s) and at each phase: the CLI shows the logs.

    The percentage can drop for a moment when Ollama announces the next layer; only a new
    high is logged.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.phase = ""
        self.step = -1
        self.at = 0.0

    def __call__(self, p: PullProgress) -> None:
        step = (p.percent or 0) // 10
        now = time.monotonic()
        if p.phase == self.phase and step <= self.step and now - self.at < 30:
            return
        self.phase, self.step, self.at = p.phase, max(step, self.step), now
        percent = p.percent or 0
        if p.phase == "download" and p.total:
            log.info(
                "%s: downloading %s: %s of %s (%d %%)",
                self.stage,
                p.model,
                fmt_size(p.completed),
                fmt_size(p.total),
                percent,
            )
        else:
            log.info("%s: downloading %s: %s", self.stage, p.model, p.status or p.phase)


def ensure_model(stage: Stage, backend: Any, name: str) -> bool:
    """True if Ollama has the model ``name``, after downloading it if needed and allowed.

    False: the model is missing and is not downloaded here (``vlm.auto_pull = false``, or a
    backend that cannot download); the stage then raises its usual error. Raises
    :class:`StageAbort` when the download fails (Ollama not reachable, unknown model, disk
    full, …) and :class:`StageInterrupted` when the run pauses or stops meanwhile: Ollama keeps
    what it has downloaded, and the next set-up continues there.

    Runs outside any database transaction (the runner calls ``setup`` between batches).
    """
    if backend.has_model(name):
        return True
    pull = getattr(backend, "pull", None)
    if pull is None or not stage.settings.vlm.auto_pull:
        return False
    name = normalize(name)
    log.info("%s: the Ollama model %s is missing; downloading it (vlm.auto_pull)", stage.name, name)
    progress_log = _ProgressLog(stage.name)

    def on_progress(p: PullProgress) -> None:
        stage.report({"stage": stage.name, **p.as_dict()})
        progress_log(p)

    stage.report({"stage": stage.name, "model": name, "phase": "manifest"})
    try:
        pull(name, on_progress=on_progress, cancel=stage.interrupted)
    except PullCancelled as exc:
        raise StageInterrupted(f"{stage.name}: download of {name} interrupted") from exc
    except PullError as exc:
        raise StageAbort(f"cannot download the Ollama model {name}: {exc}") from exc
    finally:
        stage.report(None)
    if not backend.has_model(name):
        raise StageAbort(f"Ollama still lists no model {name} after downloading it")
    log.info("%s: downloaded %s", stage.name, name)
    return True
