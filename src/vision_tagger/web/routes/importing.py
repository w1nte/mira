"""Import page: library folders and a folder browser (local mode), uploads (remote mode)."""

from __future__ import annotations

import contextlib
import logging
import os
import string
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import func, select

from ...db.models import AppSetting, Image
from ...i18n import _
from ...indexer import ROOTS_KEY, stored_roots
from ...uploads import UploadError
from ..common import ctx, redirect, render

log = logging.getLogger(__name__)
router = APIRouter()


def _counts(c) -> dict[str, int]:
    with c.db.session() as s:
        return {
            "photos": s.scalar(select(func.count(Image.id)).where(Image.missing.is_(False))) or 0
        }


@router.get("/import")
def import_page(request: Request) -> Response:
    c = ctx(request)
    return render(
        request,
        "import.html",
        remote=c.settings.remote_mode,
        roots=stored_roots(c.db, c.settings),
        job=c.jobs.current,
        counts=_counts(c),
        chunk_bytes=c.settings.remote.chunk_mb * 1024 * 1024,
        auto_process=c.settings.remote.auto_process,
    )


@router.get("/partials/import-status")
def import_status(request: Request) -> Response:
    c = ctx(request)
    return render(
        request,
        "partials/import_status.html",
        job=c.jobs.current,
        counts=_counts(c),
        running=c.runner.running,
    )


# --------------------------------------------------------------------------- local: folders
def _local_only(request: Request):
    c = ctx(request)
    if c.settings.remote_mode:
        raise HTTPException(403, "not available in remote mode")
    return c


def _save_roots(c, roots: list[str]) -> None:
    with c.db.write() as s:
        s.merge(AppSetting(key=ROOTS_KEY, value=roots))


@router.post("/import/roots")
def add_root(request: Request, path: str = Form(...)) -> Response:
    c = _local_only(request)
    folder = Path(path.strip().strip('"')).expanduser()
    if not folder.is_dir():
        return redirect(request, "/import", err=_("Not a folder: %(path)s", path=path))
    roots = stored_roots(c.db, c.settings)
    resolved = str(folder.resolve())
    if resolved not in roots:
        roots.append(resolved)
        _save_roots(c, roots)
        log.info("library folder added: %s", resolved)
    return redirect(request, "/import", msg=_("Added %(path)s.", path=resolved))


@router.post("/import/roots/remove")
def remove_root(request: Request, path: str = Form(...)) -> Response:
    c = _local_only(request)
    _save_roots(c, [r for r in stored_roots(c.db, c.settings) if r != path])
    log.info("library folder removed: %s", path)
    return redirect(
        request,
        "/import",
        msg=_("Removed. Photos already in the library stay until the next scan."),
    )


def _drives() -> list[str]:
    if os.name != "nt":
        return []
    return [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]


@router.get("/api/fs")
def browse(request: Request, path: str = "") -> JSONResponse:
    """Folders (never file contents) for the folder picker; local mode only."""
    c = _local_only(request)
    exts = {e.lower() for e in c.settings.index.extensions}
    skip = {d.lower() for d in c.settings.index.skip_dirs}
    if not path:
        start = Path.home()
        return JSONResponse(
            {
                "path": "",
                "parent": None,
                "folders": [{"name": str(start), "path": str(start)}]
                + [{"name": d, "path": d} for d in _drives()]
                + ([] if os.name == "nt" else [{"name": "/", "path": "/"}]),
                "images": 0,
            }
        )
    folder = Path(path).expanduser()
    try:
        folder = folder.resolve()
        entries = list(os.scandir(folder))
    except OSError as exc:
        raise HTTPException(404, str(exc)) from exc
    folders, images = [], 0
    for entry in entries:
        name = entry.name
        try:
            if entry.is_dir(follow_symlinks=False):
                if not name.startswith(".") and name.lower() not in skip:
                    folders.append({"name": name, "path": entry.path})
            elif Path(name).suffix.lower() in exts:
                images += 1
        except OSError:
            continue
    folders.sort(key=lambda f: f["name"].lower())
    parent = str(folder.parent) if folder.parent != folder else ""
    return JSONResponse(
        {"path": str(folder), "parent": parent, "folders": folders[:2000], "images": images}
    )


# --------------------------------------------------------------------------- remote: uploads
class NewUpload(BaseModel):
    name: str
    size: int


def _remote_only(request: Request):
    c = ctx(request)
    if not c.settings.remote_mode or c.uploads is None:
        raise HTTPException(403, "uploads are only available in remote mode")
    return c


def _state(up) -> dict:
    return {
        "id": up.id,
        "received": up.received,
        "size": up.size,
        "state": up.state,
        "files": up.files,
    }


@router.post("/api/uploads")
def create_upload(request: Request, body: NewUpload) -> JSONResponse:
    c = _remote_only(request)
    try:
        up = c.uploads.create(body.name, body.size)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(_state(up), status_code=201)


@router.get("/api/uploads/{upload_id}")
def upload_status(request: Request, upload_id: str) -> JSONResponse:
    c = _remote_only(request)
    try:
        return JSONResponse(_state(c.uploads.get(upload_id)))
    except UploadError as exc:
        raise HTTPException(404, str(exc)) from exc


#: slack on top of ``remote.chunk_mb`` for one chunk (the browser never sends more)
CHUNK_SLACK = 64 * 1024


async def read_limited(request: Request, limit: int) -> bytes:
    """The request body, refused with 413 as soon as it is larger than ``limit`` bytes (it is
    held in memory, so its size must be bounded before it is read, not afterwards)."""
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > limit):
        raise HTTPException(413, "chunk too large")
    data = bytearray()
    async for part in request.stream():
        data += part
        if len(data) > limit:
            raise HTTPException(413, "chunk too large")
    return bytes(data)


@router.put("/api/uploads/{upload_id}")
async def upload_chunk(request: Request, upload_id: str, offset: int = 0) -> JSONResponse:
    c = _remote_only(request)
    data = await read_limited(request, c.settings.remote.chunk_mb * 1024 * 1024 + CHUNK_SLACK)
    try:
        from starlette.concurrency import run_in_threadpool

        up = await run_in_threadpool(c.uploads.write, upload_id, offset, data)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(_state(up))


@router.post("/api/uploads/{upload_id}/complete")
def complete_upload(request: Request, upload_id: str) -> JSONResponse:
    c = _remote_only(request)
    try:
        up, placed = c.uploads.complete(upload_id)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    if placed and c.ingest is not None:
        c.ingest.notify()
    return JSONResponse(_state(up))


@router.delete("/api/uploads/{upload_id}")
def cancel_upload(request: Request, upload_id: str) -> Response:
    c = _remote_only(request)
    with contextlib.suppress(UploadError):
        c.uploads.cancel(upload_id)
    return Response(status_code=204)
