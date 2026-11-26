"""Database page: export and import the whole library, list and re-check missing files."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from ...db import transfer
from ...db.models import Cluster, Face, Image, Person
from ...i18n import Lazy, _
from ...indexer import Indexer, stored_roots
from ..common import back, ctx, redirect, render
from ..context import AppContext, Job

try:
    from python_multipart.multipart import MultipartParser, parse_options_header
except ModuleNotFoundError:  # pragma: no cover - python-multipart < 0.0.13
    from multipart.multipart import MultipartParser, parse_options_header  # type: ignore

router = APIRouter()

JOBS = ("import", "locations")
PER_PAGE = 100
#: cookie with the library id a browser last saw (see ``library_cookie``)
LIBRARY_COOKIE = "vt_lib"


def _busy(c: AppContext) -> Lazy | None:
    return transfer.busy_reason(c.db, runner_running=c.runner.running, jobs_busy=c.jobs.busy)


def _job(c: AppContext) -> tuple[Job | None, bool]:
    """The import or location check that runs or ran last (``None`` once another job ran
    after it: its result would be out of date), and whether it is running."""
    job = c.jobs.current
    if job is None or job.name not in JOBS:
        return None, False
    return job, job.status == "running"


def _counts(c: AppContext) -> dict[str, int]:
    with c.db.session() as s:
        return {
            "images": s.scalar(select(func.count()).select_from(Image)) or 0,
            "missing": s.scalar(select(func.count()).select_from(Image).where(Image.missing)) or 0,
            "faces": s.scalar(select(func.count()).select_from(Face)) or 0,
            "persons": s.scalar(select(func.count()).select_from(Person)) or 0,
        }


def _db_size(c: AppContext) -> int | None:
    try:
        return transfer.database_file(c.db).stat().st_size
    except (OSError, transfer.TransferError):
        return None


@router.get("/database")
def database_page(request: Request) -> Response:
    c = ctx(request)
    busy = _busy(c)
    return render(
        request,
        "database.html",
        counts=_counts(c),
        db_size=_db_size(c),
        busy=str(busy) if busy else None,
        remote=c.settings.remote_mode,
        roots=stored_roots(c.db, c.settings),
        **_job_context(c),
    )


def _job_context(c: AppContext) -> dict[str, Any]:
    job, running = _job(c)
    return {"job": job, "running": running}


@router.get("/partials/database-status")
def database_status(request: Request, watch: int = 0) -> Response:
    """Progress of an import or location check; reloads the page once the watched job ends,
    so the page shows the new library (and the browser drops its cached thumbnails)."""
    c = ctx(request)
    job, running = _job(c)
    if watch and not running:
        return Response(status_code=204, headers={"HX-Redirect": back(request, "/database")})
    return render(request, "partials/database_status.html", job=job, running=running)


# --------------------------------------------------------------------------- export
@router.get("/database/export")
def export_database(request: Request) -> Response:
    """Download a consistent snapshot of the live database (also while the pipeline runs)."""
    c = ctx(request)
    target = transfer.temp_file(c.settings, "export")
    try:
        transfer.export_database(c.db, target)
    except transfer.TransferError as exc:
        transfer.remove_file(target)
        return redirect(request, "/database", err=str(exc))
    return FileResponse(
        target,
        media_type=transfer.MEDIA_TYPE,
        filename=transfer.export_name(),
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(transfer.remove_file, target),
    )


# --------------------------------------------------------------------------- import
async def receive_file(request: Request, target: Path, field: str = "file") -> str | None:
    """Stream the file of the multipart field ``field`` straight into ``target``.

    Starlette's form parser would spool it to the system temp folder first; a library
    database can be gigabytes. Returns the file name the browser sent (``None``: no file).
    """
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise HTTPException(400, "multipart/form-data expected")
    wanted = field.encode()
    headers: dict[bytes, bytes] = {}
    current = {"field": b"", "value": b""}
    state: dict[str, Any] = {"writing": False, "name": None}
    pending: list[bytes] = []

    def on_header_field(data: bytes, start: int, end: int) -> None:
        current["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        current["value"] += data[start:end]

    def on_header_end() -> None:
        headers[current["field"].lower()] = current["value"]
        current["field"] = current["value"] = b""

    def on_headers_finished() -> None:
        _disposition, options = parse_options_header(headers.get(b"content-disposition", b""))
        filename = options.get(b"filename")
        take = options.get(b"name") == wanted and bool(filename) and state["name"] is None
        state["writing"] = take
        if take:
            state["name"] = filename.decode("utf-8", "replace")

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if state["writing"]:
            pending.append(data[start:end])

    def on_part_end() -> None:
        state["writing"] = False
        headers.clear()

    parser = MultipartParser(
        boundary,
        {
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )

    async def flush(fh: Any) -> None:
        if pending:
            data = b"".join(pending)
            pending.clear()
            await run_in_threadpool(fh.write, data)

    with target.open("wb") as fh:
        async for chunk in request.stream():
            parser.write(chunk)
            await flush(fh)
        parser.finalize()
        await flush(fh)
    return state["name"]


def _import_job(c: AppContext, source: Path) -> Callable[[Job], Lazy]:
    def job(j: Job) -> Lazy:
        def before_replace() -> None:
            # a pipeline started (here or from the command line) while the file was prepared
            reason = transfer.busy_reason(c.db, runner_running=c.runner.running)
            if reason is not None:
                raise transfer.TransferError(reason)

        def report(message: Lazy) -> None:
            j.message = message

        try:
            result = transfer.import_database(
                c.db, c.settings, source, before_replace=before_replace, progress=report
            )
        finally:
            transfer.remove_file(source)
        c.forget_library()
        missing = _check_locations(c, j)
        return Lazy(
            "Database imported: %(images)s photos, %(missing)s of them not found on this computer.",
            images=result.images,
            missing=missing,
        )

    return job


def _check_locations(c: AppContext, j: Job) -> int:
    def progress(done: int, total: int) -> None:  # job thread: translated when shown
        j.progress = {"processed": done, "candidates": total}
        j.message = Lazy("%(done)s/%(total)s photo locations checked", done=done, total=total)

    j.message = Lazy("Checking where the photos are…")
    return Indexer(c.db, c.settings, should_stop=lambda: j.stop_requested).check_locations(progress)


@router.post("/database/import")
async def import_database(request: Request) -> Response:
    c = ctx(request)
    busy = _busy(c)
    if busy is not None:
        return redirect(request, "/database", err=str(busy))
    target = transfer.temp_file(c.settings, "import")
    try:
        name = await receive_file(request, target)
        if not name or target.stat().st_size == 0:
            transfer.remove_file(target)
            return redirect(request, "/database", err=_("Choose a database file first."))
        info = await run_in_threadpool(transfer.inspect_file, target)
    except transfer.TransferError as exc:
        transfer.remove_file(target)
        return redirect(request, "/database", err=str(exc))
    except BaseException:
        transfer.remove_file(target)
        raise
    busy = _busy(c)  # the upload may have taken a while
    if busy is not None or not c.jobs.start("import", _import_job(c, target)):
        transfer.remove_file(target)
        return redirect(request, "/database", err=str(busy or _("Another job is running.")))
    return redirect(
        request,
        "/database",
        msg=_(
            "Importing %(name)s (%(num)s photos). Mira checks where the photos are next.",
            name=Path(name).name,
            num=info.images,
        ),
    )


# --------------------------------------------------------------------------- missing files
@router.post("/database/check")
def check_locations(request: Request) -> Response:
    c = ctx(request)

    def job(j: Job) -> Lazy:
        missing = _check_locations(c, j)
        with c.db.session() as s:
            total = s.scalar(select(func.count()).select_from(Image)) or 0
        return Lazy(
            "Location check: %(found)s photos found, %(missing)s missing.",
            found=total - missing,
            missing=missing,
        )

    if not c.jobs.start("locations", job):
        return redirect(request, "/database/missing", err=_("Another job is running."))
    return redirect(request, "/database/missing", msg=_("Checking where the photos are…"))


@router.get("/database/missing")
def missing_page(request: Request, page: int = 1) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        total = s.scalar(select(func.count()).select_from(Image).where(Image.missing)) or 0
        pages = max(1, math.ceil(total / PER_PAGE))
        page = min(max(1, page), pages)
        images = list(
            s.execute(
                select(Image.id, Image.path, Image.filename, Image.recorded_at)
                .where(Image.missing)
                .order_by(Image.path_key)
                .offset((page - 1) * PER_PAGE)
                .limit(PER_PAGE)
            ).all()
        )
        ids = [row.id for row in images]
        faces = dict(
            s.execute(
                select(Face.image_id, func.count())
                .where(Face.image_id.in_(ids), Face.ignored.is_(False))
                .group_by(Face.image_id)
            ).all()
        )
        persons: dict[int, list[str]] = {}
        for image_id, name in s.execute(
            select(Face.image_id, Person.name)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .where(Face.image_id.in_(ids), Face.ignored.is_(False))
            .distinct()
            .order_by(Person.name)
        ):
            persons.setdefault(image_id, []).append(name)
    rows = [
        {
            "id": row.id,
            "path": row.path,
            "filename": row.filename,
            "folder": _folder(row.path, row.filename),
            "date": row.recorded_at,
            "faces": faces.get(row.id, 0),
            "persons": persons.get(row.id, []),
        }
        for row in images
    ]
    return render(
        request,
        "missing.html",
        rows=rows,
        total=total,
        page=page,
        pages=pages,
        remote=c.settings.remote_mode,
        **_job_context(c),
    )


def _folder(path: str, filename: str) -> str:
    """The folder part of a stored path, also for paths of another OS (``D:\\Photos\\a.jpg``)."""
    return path[: -len(filename)].rstrip("\\/") if filename and path.endswith(filename) else path


# --------------------------------------------------------------------------- browser caches
def library_cookie() -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Any]:
    """Middleware: a browser that last saw another library (before an import) drops its HTTP
    cache once, on its next page view. Thumbnails and face crops are cached for a day under
    URLs with image and face ids, which now mean other photos.

    ``Clear-Site-Data`` needs a secure context (localhost or HTTPS); the service worker's
    thumbnail cache is renewed separately (its name contains the library id).
    """

    async def middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        response = await call_next(request)
        if (
            request.method != "GET"
            or request.headers.get("HX-Request")
            or request.headers.get("sec-fetch-dest", "document") != "document"
            or not response.headers.get("content-type", "").startswith("text/html")
        ):
            return response
        c: AppContext | None = getattr(request.app.state, "ctx", None)
        if c is None:
            return response
        with contextlib.suppress(Exception):  # never break a page for this
            current = await run_in_threadpool(transfer.library_id, c.db)
            seen = request.cookies.get(LIBRARY_COOKIE)
            if seen != current:
                if seen is not None or current != "0":
                    response.headers["Clear-Site-Data"] = '"cache"'
                response.set_cookie(
                    LIBRARY_COOKIE, current, max_age=400 * 86400, httponly=True, samesite="lax"
                )
        return response

    return middleware
