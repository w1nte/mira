"""System → Log: the newest entries of the log file (``vision_tagger.logs``), download it, and
report a bug. Opening the file in the text editor goes through ``/reveal`` (kind ``log``): only
in local mode, for a request from this computer.
"""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ... import bugreport, logs
from ..common import ctx, render

router = APIRouter()

#: what the page shows: everything, warnings and errors, errors only
FILTERS = {"all": logging.DEBUG, "warnings": logging.WARNING, "errors": logging.ERROR}
LIMIT = 300  # entries on the page; the download has everything


@router.get("/logs")
def log_page(request: Request, show: str = "all") -> Response:
    c = ctx(request)
    if show not in FILTERS:
        show = "all"
    path = logs.log_path(c.settings)
    parts = logs.files(path)
    return render(
        request,
        "logs.html",
        view=logs.read(path, min_level=FILTERS[show], limit=LIMIT),
        show=show,
        limit=LIMIT,
        log_file=path,  # not `path`: base.html uses that name for the URL path
        exists=path.is_file(),
        size=sum(p.stat().st_size for p in parts if p.is_file()),
        downloadable=bool(parts),
        remote=c.settings.remote_mode,
        bug_url=bugreport.issue_url(c.settings),
    )


@router.get("/logs/download")
def download(request: Request) -> Response:
    """The log file with its older parts, oldest first, as one file (for a bug report)."""
    path = logs.log_path(ctx(request).settings)
    if not logs.files(path):
        raise HTTPException(404, "there is no log yet")
    name = f"vision-tagger-{date.today().isoformat()}.log"
    return StreamingResponse(
        logs.iter_bytes(path),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )
