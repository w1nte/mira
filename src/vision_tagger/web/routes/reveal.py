"""Show a photo, a library folder, the database file or the data folder in Explorer, Finder or
the file manager (local mode, requests from this computer only; see ``common.can_reveal``).

The request names *what* to show (a kind and an id or index), never a path: the server looks
the path up itself, so nothing else on the computer can be opened through this endpoint.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response

from ...db.models import Image
from ...i18n import _
from ...indexer import stored_roots
from ...reveal import RevealError
from ..common import back, can_reveal, ctx, redirect

router = APIRouter()

KINDS = ("image", "root", "database", "data")


@router.post("/reveal")
def reveal(request: Request, kind: str = Form(...), item: str = Form("", alias="id")) -> Response:
    c = ctx(request)
    if not can_reveal(request):
        raise HTTPException(403, "only available on the computer Mira runs on (local mode)")
    if kind not in KINDS:
        raise HTTPException(400, "unknown kind")
    url = back(request, "/")
    number = int(item) if item.strip().isdigit() else None
    if kind == "image":
        if number is None:
            raise HTTPException(400, "photo id expected")
        with c.db.session() as s:
            img = s.get(Image, number)
            stored = img.path if img is not None else None
        if stored is None:
            raise HTTPException(404, "no such photo")
        path, select = Path(stored), True
        if not path.is_absolute() or not path.is_file():
            return redirect(request, url, err=_("The file is not there: %(path)s", path=stored))
    elif kind == "root":
        roots = stored_roots(c.db, c.settings)
        if number is None or number >= len(roots):
            raise HTTPException(404, "no such library folder")
        path, select = Path(roots[number]), False
        if not path.is_absolute() or not path.is_dir():
            return redirect(
                request, url, err=_("The folder is not there: %(path)s", path=roots[number])
            )
    elif kind == "database":
        path, select = c.settings.db_path, True
    else:
        path, select = c.settings.data_dir, False
    try:
        c.file_manager.show(path, select=select)
    except RevealError as exc:
        return redirect(
            request, url, err=_("The file manager could not be opened: %(error)s", error=str(exc))
        )
    if request.headers.get("HX-Request"):
        return Response(status_code=204)  # the file manager is the answer; the page stays
    return redirect(request, url, msg=_("Opened in the file manager."))
