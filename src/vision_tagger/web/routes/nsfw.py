"""Marking photos NSFW or not NSFW by hand (image page, gallery selection).

A manual verdict always wins over the classifier and the vision model (see ``nsfw.verdict``);
"automatic" forgets it again.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response

from ...i18n import _n
from ...nsfw import NO, YES
from ...nsfw.verdict import set_manual
from ..common import back, ctx, ints, redirect

router = APIRouter()

#: form value -> stored verdict (``None``: automatic again)
VALUES = {"yes": YES, "no": NO, "auto": None}


def _mark(request: Request, image_ids: list[int], value: str, default: str) -> Response:
    if value not in VALUES:
        raise HTTPException(400, "value must be yes, no or auto")
    c = ctx(request)
    url = back(request, default)
    with c.db.write() as s:
        changed = set_manual(s, image_ids, VALUES[value], c.settings.nsfw.threshold)
    n = len(set(image_ids)) if VALUES[value] is not None else changed
    if value == "yes":
        msg = _n("%(num)d photo marked as NSFW.", "%(num)d photos marked as NSFW.", n)
    elif value == "no":
        msg = _n("%(num)d photo marked as not NSFW.", "%(num)d photos marked as not NSFW.", n)
    else:
        msg = _n(
            "%(num)d photo is checked automatically again.",
            "%(num)d photos are checked automatically again.",
            n,
        )
    return redirect(request, url, msg=msg)


@router.post("/images/nsfw")
def mark_selected(
    request: Request, value: str = Form(...), image_ids: list[str] = Form(default_factory=list)
) -> Response:
    """The gallery selection's action bar."""
    return _mark(request, ints(image_ids), value, "/images")


@router.post("/images/{image_id}/nsfw")
def mark_one(request: Request, image_id: int, value: str = Form(...)) -> Response:
    return _mark(request, [image_id], value, f"/images/{image_id}")
