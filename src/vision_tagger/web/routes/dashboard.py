"""Dashboard: indexing, pipeline control and status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ...db.models import AppSetting, Cluster, DuplicateFile, Face, Image, Person, Run, utcnow
from ...indexer import ROOTS_KEY, Indexer, stored_roots
from ...pipeline import control
from ...pipeline.stages import STAGE_NAMES, stage_versions
from ..common import ctx, redirect, render
from ..context import Job

router = APIRouter()


def _roots(request: Request) -> list[str]:
    c = ctx(request)
    return stored_roots(c.db, c.settings)


def status_context(request: Request) -> dict[str, Any]:
    c = ctx(request)
    counts = control.stage_counts(c.db)
    versions = stage_versions(c.settings)
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
        "stages": [st for st in STAGE_NAMES if st in c.settings.pipeline.stages]
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


@router.get("/")
def dashboard(request: Request) -> Response:
    c = ctx(request)
    return render(
        request,
        "dashboard.html",
        roots=_roots(request),
        settings=c.settings,
        vlm_health=c.vlm_health(),
        providers=c.onnx_providers(),
        **status_context(request),
    )


@router.get("/partials/status")
def status_partial(request: Request) -> Response:
    return render(request, "partials/status.html", **status_context(request))


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
def start_index(request: Request, roots: str = Form("")) -> Response:
    c = ctx(request)
    root_list = [r.strip().strip('"') for r in roots.splitlines() if r.strip()]
    if not root_list:
        return redirect(request, "/", err="Please enter at least one folder.")
    with c.db.write() as s:
        s.merge(AppSetting(key=ROOTS_KEY, value=root_list))

    def job(j: Job) -> str:
        def progress(stats: Any) -> None:
            j.progress = stats.as_dict()
            j.message = (
                f"{stats.processed}/{stats.candidates} new or changed files analysed "
                f"({stats.found} found)"
            )

        stats = Indexer(
            c.db, c.settings, progress=progress, should_stop=lambda: j.stop_requested
        ).run(root_list)
        return (
            f"Index: {stats.new} new, {stats.duplicates} duplicates, {stats.moved} moved, "
            f"{stats.updated} updated, {stats.unchanged} unchanged, {stats.errors} errors"
        )

    if not c.jobs.start("index", job):
        return redirect(request, "/", err="Another job is running.")
    return redirect(request, "/", msg="Indexing started.")


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
        return redirect(request, "/", err="Select at least one stage.")
    if control.is_runner_alive(c.db) or c.runner.running:
        return redirect(request, "/", err="A run is already active.")
    lim = int(limit) if limit.strip().isdigit() and int(limit) > 0 else None
    c.runner.start(stages, batch_size=batch_size or None, limit=lim)
    return redirect(request, "/", msg=f"Run started: {', '.join(stages)}.")


@router.post("/actions/{state}")
def change_state(request: Request, state: str) -> Response:
    c = ctx(request)
    mapping = {"pause": control.PAUSED, "resume": control.RUNNING, "stop": control.STOPPED}
    if state in mapping:
        control.request_state(c.db, mapping[state])
        if state == "stop":
            c.jobs.request_stop()
        return redirect(request, "/", msg=f"{state.capitalize()} requested.")
    if state == "retry-errors":
        n = control.retry_errors(c.db, request.query_params.get("stage"))
        return redirect(request, "/", msg=f"{n} items re-queued.")
    if state == "skip-errors":
        n = control.skip_errors(c.db, request.query_params.get("stage"))
        return redirect(request, "/", msg=f"{n} items skipped.")
    if state == "requeue-outdated":
        stage = request.query_params.get("stage", "")
        versions = stage_versions(c.settings)
        if stage not in versions:
            return redirect(request, "/", err="Unknown stage.")
        n = control.requeue_outdated(c.db, stage, versions[stage])
        return redirect(request, "/", msg=f"{n} outdated {stage} results re-queued.")
    if state == "recluster":
        full = request.query_params.get("full") == "1"

        def job(_j: Job) -> str:
            from ...pipeline.stages.faces import cluster_and_match

            return cluster_and_match(c.db, c.settings, full=full)

        ok = c.jobs.start("recluster", job)
        return redirect(
            request,
            "/",
            msg="Clustering started." if ok else None,
            err=None if ok else "Another job is running.",
        )
    if state == "import-references":

        def job(_j: Job) -> str:
            from ...faces.references import import_references
            from ...pipeline.stages.faces import cluster_and_match

            stats = import_references(c.db, c.settings)
            msg = cluster_and_match(c.db, c.settings, full=False)
            return (
                f"References: {stats.added} added for {stats.persons} persons, "
                f"{len(stats.no_face)} without face, {len(stats.errors)} errors. {msg}"
            )

        ok = c.jobs.start("references", job)
        return redirect(
            request,
            "/",
            msg="Reference import started." if ok else None,
            err=None if ok else "Another job is running.",
        )
    return redirect(request, "/", err=f"Unknown action {state}.")


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
            "message": data["job"].message,
        },
    }
