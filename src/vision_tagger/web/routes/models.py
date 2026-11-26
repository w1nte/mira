"""Ollama models: their state (Environment card, Home) and downloading them in the background.

Model names never come from a request: forms name a model by its key in
:func:`vlm.models.required_models` (``vlm``, ``embed``, ``search``), or download every missing
one. The server's Ollama downloads, so remote mode and Docker work the same way.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from ...i18n import Lazy, _
from ...pipeline import control
from ...vlm.models import PullCancelled, PullProgress, pull_message, pull_missing, required_models
from ..common import back, ctx, redirect, render
from ..context import AppContext, Job

router = APIRouter()

JOB_NAME = "models"


def _run_active(c: AppContext) -> bool:
    return c.runner.running or control.is_runner_alive(c.db)


def downloads(c: AppContext) -> dict[str, dict[str, Any]]:
    """Model name -> progress of a download running now (the models job or a pipeline run)."""
    out: dict[str, dict[str, Any]] = {}
    if _run_active(c):
        pull = control.pull_progress(c.db)
        if pull and pull.get("model"):
            out[pull["model"]] = pull
    job = c.downloads.current
    if job is not None and job.status == "running":
        for name in job.progress.get("queued", []):
            out.setdefault(name, {"model": name, "phase": "queued"})
        current = job.progress.get("current")
        if current and current.get("model"):
            out[current["model"]] = current
    return out


def models_context(c: AppContext, *, fresh: bool = False) -> dict[str, Any]:
    """What the model templates need.

    While a download runs, or a pipeline run may start one (``vlm.auto_pull``), the state is
    read anew and the Environment card follows it (``poll``).
    """
    running = downloads(c)
    active = _run_active(c)
    status = c.models_status(max_age=0 if fresh or running or active else 10)
    return {
        "models": status,
        "poll": bool(running) or (active and bool(status.missing)),
        "downloads": running,
        "downloading": c.downloads.busy,
        "download_job": c.downloads.current,
        "remote": c.settings.remote_mode,
    }


def start_download(c: AppContext, keys: list[str] | None = None) -> bool:
    """Download the missing models (or those of ``keys``) one after another in a job.

    False if a download job is running already.
    """
    settings = c.settings

    def job(j: Job) -> Lazy:  # runs in the job thread: messages are translated when shown
        j.message = Lazy("checking which models are missing")

        def show(index: int, todo: list[str], current: dict[str, Any]) -> None:
            current = {**current, "index": index, "count": len(todo)}
            j.progress = {"current": current, "queued": todo[index + 1 :]}
            j.message = pull_message(current)

        def on_start(index: int, todo: list[str]) -> None:
            show(index, todo, {"model": todo[index], "phase": "manifest"})

        def on_progress(index: int, todo: list[str], p: PullProgress) -> None:
            show(index, todo, p.as_dict())

        try:
            pulled = pull_missing(
                settings,
                keys=keys,
                on_start=on_start,
                on_progress=on_progress,
                cancel=lambda: j.stop_requested,
            )
        except PullCancelled:
            return Lazy(
                "Download cancelled. Ollama keeps what it has downloaded; the next download "
                "continues there."
            )
        finally:
            j.progress = {}
            c.forget_models_status()
        if not pulled:
            return Lazy("All AI models are there already.")
        return Lazy("AI models ready: %(models)s", models=", ".join(pulled))

    return c.downloads.start(JOB_NAME, job)


@router.post("/models/download")
def download(request: Request, key: str = Form("")) -> Response:
    """Start downloading the missing models; ``key`` (not a name) picks one of them."""
    c = ctx(request)
    target = back(request, "/pipeline")
    known = {m.key for m in required_models(c.settings)}
    if not known:
        return redirect(request, target, err=_("This configuration needs no models from Ollama."))
    if key and key not in known:
        return redirect(request, target, err=_("Unknown model."))
    if not start_download(c, [key] if key else None):
        return redirect(request, target, err=_("The AI models are being downloaded already."))
    return redirect(request, target, msg=_("Downloading the AI models in the background."))


@router.post("/models/cancel")
def cancel(request: Request) -> Response:
    c = ctx(request)
    target = back(request, "/pipeline")
    if not c.downloads.busy:
        return redirect(request, target, err=_("No download is running."))
    c.downloads.request_stop()
    return redirect(request, target, msg=_("Cancelling the download…"))


@router.get("/partials/models")
def models_partial(request: Request) -> Response:
    """The model list of the Environment card (polled while a download runs)."""
    return render(request, "partials/models.html", **models_context(ctx(request), fresh=True))


@router.get("/partials/models-nudge")
def models_nudge(request: Request, polling: int = 0) -> Response:
    """Home: a card while models are missing or downloading, or Ollama is not reachable.

    Loaded after the page (Ollama may be slow to answer). Nothing when all is ready, unless
    the card itself asks (``polling``): then it says that the download is done.
    """
    c = ctx(request)
    settings = c.settings
    wanted = {"vlm", "embed"} & set(settings.pipeline.stages)
    if not required_models(settings) or not wanted:
        return Response("")  # the fake backend, or properties and search are off in config.toml
    data = models_context(c, fresh=bool(polling))
    status = data["models"]
    busy = data["downloading"] or bool(data["downloads"])
    if not busy and status.reachable and not status.missing and not polling:
        return Response("")
    return render(request, "partials/models_nudge.html", busy=busy, **data)
