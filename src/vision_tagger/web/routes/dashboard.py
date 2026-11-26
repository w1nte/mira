"""Pipeline page: run control, stage status, runs, errors, environment; indexing actions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ...db.models import AppSetting, Cluster, DuplicateFile, Face, Image, Person, Run, utcnow
from ...i18n import Lazy, LazyJoin, _, _n
from ...indexer import ROOTS_KEY, Indexer, stored_roots
from ...pipeline import control
from ...pipeline.stages import STAGE_NAMES
from ..common import ctx, redirect, render
from ..context import Job

router = APIRouter()


def _roots(request: Request) -> list[str]:
    c = ctx(request)
    return stored_roots(c.db, c.settings)


def status_context(request: Request) -> dict[str, Any]:
    c = ctx(request)
    counts = control.stage_counts(c.db)
    versions = c.stage_versions()
    outdated = control.outdated_counts(c.db, versions)
    ctrl = control.get_control(c.db)
    runs = control.latest_runs(c.db, limit=6)
    active = next((r for r in runs if r.id == ctrl.active_run_id), None)
    eta = rate = None
    if active and active.processed and active.started_at:
        elapsed = (utcnow() - active.started_at).total_seconds()
        rate = active.processed / elapsed if elapsed > 0 else None
        stage = ctrl.active_stage
        pending = counts.get(stage or "", {}).get("pending", 0) if stage else 0
        eta = pending / rate if rate else None
    with c.db.session() as s:
        totals = {
            "images": s.scalar(select(func.count()).select_from(Image)) or 0,
            "missing": s.scalar(select(func.count()).select_from(Image).where(Image.missing)) or 0,
            "duplicates": s.scalar(select(func.count()).select_from(DuplicateFile)) or 0,
            "faces": s.scalar(select(func.count()).select_from(Face)) or 0,
            "clusters": s.scalar(select(func.count()).select_from(Cluster)) or 0,
            "persons": s.scalar(select(func.count()).select_from(Person)) or 0,
            "unnamed_clusters": s.scalar(
                select(func.count()).select_from(Cluster).where(Cluster.person_id.is_(None))
            )
            or 0,
        }
    alive = c.runner.running or control.is_runner_alive(c.db)
    return {
        "counts": counts,
        # every stage; the ones config.toml turns off are shown unticked and can still be run
        "stages": list(STAGE_NAMES),
        "enabled_stages": [st for st in STAGE_NAMES if st in c.settings.pipeline.stages]
        or list(STAGE_NAMES),
        "outdated": outdated,
        "ctrl": ctrl,
        "alive": alive,
        "runs": runs,
        "active_run": active,
        "rate": rate,
        "eta": eta,
        "totals": totals,
        "job": c.jobs.current,
        "jobs": list(c.jobs.history)[:5],
        "runner_error": c.runner.last_error,
        "errors": control.recent_errors(c.db, limit=20),
    }


@router.get("/pipeline")
def pipeline_page(request: Request) -> Response:
    c = ctx(request)
    return render(
        request,
        "pipeline.html",
        roots=_roots(request),
        settings=c.settings,
        vlm_health=c.vlm_health(),
        providers=c.onnx_providers(),
        **status_context(request),
    )


@router.get("/partials/status")
def status_partial(request: Request) -> Response:
    return render(request, "partials/status.html", **status_context(request))


@router.get("/partials/activity")
def activity_partial(request: Request) -> Response:
    """The small "what is running" indicator in the sidebar. Polled on every page, so cheap."""
    c = ctx(request)
    ctrl = control.get_control(c.db)
    alive = c.runner.running or control.is_runner_alive(c.db)
    stage = ctrl.active_stage if alive else None
    if not alive:
        state = "idle"
    elif ctrl.desired_state == control.PAUSED:
        state = "paused"
    elif ctrl.desired_state == control.STOPPED:
        state = "stopping"
    else:
        state = "running"
    percent = None
    if stage:
        cnt = control.stage_counts(c.db).get(stage, {})
        total = cnt.get("total", 0)
        percent = round(100 * cnt.get("done", 0) / total) if total else None
    job = c.jobs.current if c.jobs.busy else None
    return render(
        request, "partials/activity.html", state=state, stage=stage, percent=percent, job=job
    )


@router.get("/partials/health")
def health_partial(request: Request) -> Response:
    c = ctx(request)
    return render(
        request,
        "partials/health.html",
        vlm_health=c.vlm_health(max_age=0),
        providers=c.onnx_providers(),
        settings=c.settings,
    )


# --------------------------------------------------------------------------- actions
@router.post("/actions/index")
def start_index(request: Request, roots: str = Form(""), analyse: str = Form("")) -> Response:
    """Scan the library folders (given, or the saved ones); ``analyse`` runs the pipeline after."""
    c = ctx(request)
    root_list = [r.strip().strip('"') for r in roots.splitlines() if r.strip()]
    if c.settings.remote_mode:
        root_list = stored_roots(c.db, c.settings)  # the upload folder, nothing else
    elif root_list:
        with c.db.write() as s:
            s.merge(AppSetting(key=ROOTS_KEY, value=root_list))
    else:
        root_list = stored_roots(c.db, c.settings)
    if not root_list:
        return redirect(request, "/import", err=_("Add at least one folder first."))

    def job(j: Job) -> Lazy:
        def progress(stats: Any) -> None:  # runs in the job thread: translated when shown
            j.progress = stats.as_dict()
            message = Lazy(
                "%(processed)s/%(candidates)s new or changed files analysed (%(found)s found)",
                processed=stats.processed,
                candidates=stats.candidates,
                found=stats.found,
            )
            if stats.thumbnails_todo:
                thumbs = Lazy(
                    "thumbnails %(warmed)s/%(todo)s",
                    warmed=stats.thumbnails_warmed,
                    todo=stats.thumbnails_todo,
                )
                j.message = LazyJoin(" · ", [message, thumbs])
            else:
                j.message = message

        stats = Indexer(
            c.db, c.settings, progress=progress, should_stop=lambda: j.stop_requested
        ).run(root_list)
        if analyse and not j.stop_requested and not c.runner.running:
            c.runner.start(None)
        return Lazy(
            "Index: %(new)s new, %(dup)s duplicates, %(moved)s moved, %(updated)s updated, "
            "%(unchanged)s unchanged, %(errors)s errors",
            new=stats.new,
            dup=stats.duplicates,
            moved=stats.moved,
            updated=stats.updated,
            unchanged=stats.unchanged,
            errors=stats.errors,
        )

    if not c.jobs.start("index", job):
        return redirect(request, "/import", err=_("Another job is running."))
    return redirect(request, "/import", msg=_("Scanning started."))


@router.post("/actions/run")
def start_run(
    request: Request,
    stages: list[str] = Form(default_factory=list),
    batch_size: int = Form(0),
    limit: str = Form(""),
) -> Response:
    c = ctx(request)
    stages = [st for st in STAGE_NAMES if st in stages]
    if not stages:
        return redirect(request, "/pipeline", err=_("Select at least one stage."))
    if control.is_runner_alive(c.db) or c.runner.running:
        return redirect(request, "/pipeline", err=_("A run is already active."))
    if c.importing:
        return redirect(request, "/pipeline", err=_("Wait until the database import has finished."))
    lim = int(limit) if limit.strip().isdigit() and int(limit) > 0 else None
    c.runner.start(stages, batch_size=batch_size or None, limit=lim)
    return redirect(
        request, "/pipeline", msg=_("Run started: %(stages)s.", stages=", ".join(stages))
    )


@router.post("/actions/{state}")
def change_state(request: Request, state: str) -> Response:
    c = ctx(request)
    mapping = {"pause": control.PAUSED, "resume": control.RUNNING, "stop": control.STOPPED}
    requested = {
        "pause": _("Pause requested."),
        "resume": _("Resume requested."),
        "stop": _("Stop requested."),
    }
    if state in mapping:
        control.request_state(c.db, mapping[state])
        if state == "stop":
            c.jobs.request_stop()
        return redirect(request, "/pipeline", msg=requested[state])
    if state == "retry-errors":
        n = control.retry_errors(c.db, request.query_params.get("stage"))
        return redirect(
            request, "/pipeline", msg=_n("%(num)d item re-queued.", "%(num)d items re-queued.", n)
        )
    if state == "skip-errors":
        n = control.skip_errors(c.db, request.query_params.get("stage"))
        return redirect(
            request, "/pipeline", msg=_n("%(num)d item skipped.", "%(num)d items skipped.", n)
        )
    if state == "requeue-outdated":
        stage = request.query_params.get("stage", "")
        versions = c.stage_versions()
        if stage not in versions:
            return redirect(request, "/pipeline", err=_("Unknown stage."))
        n = control.requeue_outdated(c.db, stage, versions[stage])
        return redirect(
            request,
            "/pipeline",
            msg=_n(
                "%(num)d outdated %(stage)s result re-queued.",
                "%(num)d outdated %(stage)s results re-queued.",
                n,
                stage=stage,
            ),
        )
    if state == "recluster":
        full = request.query_params.get("full") == "1"

        def job(_j: Job) -> str:
            from ...pipeline.stages.faces import cluster_and_match

            return cluster_and_match(c.db, c.settings, full=full)

        ok = c.jobs.start("recluster", job)
        return redirect(
            request,
            "/pipeline",
            msg=_("Clustering started.") if ok else None,
            err=None if ok else _("Another job is running."),
        )
    if state == "regroup-animals":
        full = request.query_params.get("full") == "1"

        def regroup_job(_j: Job) -> str:
            from ...pipeline.stages.animals import group_animals

            return group_animals(c.db, c.settings, full=full)

        ok = c.jobs.start("regroup", regroup_job)
        return redirect(
            request,
            "/pipeline",
            msg=_("Grouping of the animals started.") if ok else None,
            err=None if ok else _("Another job is running."),
        )
    if state == "import-references":

        def job(_j: Job) -> Lazy:
            from ...faces.references import import_references
            from ...pipeline.stages.faces import cluster_and_match

            stats = import_references(c.db, c.settings)
            msg = cluster_and_match(c.db, c.settings, full=False)  # technical summary
            return Lazy(
                "References: %(added)s added for %(persons)s persons, %(no_face)s without face, "
                "%(errors)s errors. %(details)s",
                added=stats.added,
                persons=stats.persons,
                no_face=len(stats.no_face),
                errors=len(stats.errors),
                details=msg,
            )

        ok = c.jobs.start("references", job)
        return redirect(
            request,
            "/pipeline",
            msg=_("Reference import started.") if ok else None,
            err=None if ok else _("Another job is running."),
        )
    return redirect(request, "/pipeline", err=_("Unknown action %(action)s.", action=state))


@router.get("/api/status")
def api_status(request: Request) -> dict[str, Any]:
    data = status_context(request)
    ctrl = data["ctrl"]
    active: Run | None = data["active_run"]
    return {
        "counts": data["counts"],
        "outdated": data["outdated"],
        "desired_state": ctrl.desired_state,
        "active_stage": ctrl.active_stage,
        "active_run": None
        if active is None
        else {"id": active.id, "processed": active.processed, "failed": active.failed},
        "totals": data["totals"],
        "job": None
        if data["job"] is None
        else {
            "name": data["job"].name,
            "status": data["job"].status,
            "message": str(data["job"].message),
        },
    }
