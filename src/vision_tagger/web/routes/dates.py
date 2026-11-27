"""The "Dates" page: photos that belong together, to correct their dates in one go.

Groups come from :mod:`vision_tagger.dates`. The page shows first the groups whose dates
disagree or only come from file times; the gallery's selection bar posts to the same
``/dates/apply`` endpoint.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, time
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from ... import dates as D
from ...i18n import _, _n
from ...query import ImageFilter, all_image_ids
from ..common import back, ctx, ints, redirect, render

log = logging.getLogger(__name__)
router = APIRouter()

GROUPS_PER_PAGE = 12
TILES_PER_GROUP = 240  # a bigger group shows this many; narrow it with a folder or year
MODES = ("suggested", "set", "shift")

# the last computation, reused while nothing was committed (data_version) and the scope and
# the visual index are the same
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _groups(c: Any, f: ImageFilter) -> list[D.DateGroup]:
    vectors = None
    try:
        vectors = c.search.visual_lookup()
    except Exception:  # pragma: no cover - the search engine could not start (models missing)
        log.warning("date groups: visual vectors unavailable", exc_info=True)
    version = c.db.data_version()
    key = (version, f.query_string(), vectors is not None, id(c.db))
    with _cache_lock:
        if version is not None and _cache.get("key") == key:
            return _cache["groups"]
    with c.db.session() as s:
        ids = None if f.is_empty else all_image_ids(s, f)
        photos = D.load_photos(s, ids)
    groups = D.group_photos(photos, vectors)
    with _cache_lock:
        _cache.update(key=key, groups=groups)
    return groups


@router.get("/dates")
def dates_page(request: Request, show: str = "attention", page: int = 1) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    show = show if show in ("attention", "all") else "attention"
    groups = [g for g in _groups(c, f) if g.size > 1 or g.loose]
    attention = sum(1 for g in groups if g.needs_attention)
    listed = [g for g in groups if g.needs_attention] if show == "attention" else groups
    pages = max(1, -(-len(listed) // GROUPS_PER_PAGE))
    page = min(max(1, page), pages)
    shown = listed[(page - 1) * GROUPS_PER_PAGE : page * GROUPS_PER_PAGE]
    return render(
        request,
        "dates.html",
        f=f,
        show=show,
        page=page,
        pages=pages,
        groups=shown,
        total_groups=len(groups),
        attention=attention,
        listed=len(listed),
        tiles_per_group=TILES_PER_GROUP,
        visual=_visual_available(c),
        today_iso=date.today().isoformat(),
    )


def _visual_available(c: Any) -> bool:
    try:
        return c.search.visual_lookup() is not None
    except Exception:  # pragma: no cover - see _groups
        return False


def _parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_time(value: str) -> time | None:
    value = value.strip()
    if not value:
        return None
    try:
        return time.fromisoformat(value)
    except ValueError:
        return None


@router.post("/dates/apply")
def apply(
    request: Request,
    mode: str = Form("set"),
    image_ids: list[str] = Form(default_factory=list),
    suggested: list[str] = Form(default_factory=list),
    day: str = Form(""),
    at: str = Form(""),
) -> Response:
    """Set the date of the selected photos.

    ``mode``: ``suggested`` (each photo gets the date its group suggests, sent as
    ``suggested=<id>=<ISO date>``), ``set`` (all get ``day``, at ``at`` or their own time of
    day) or ``shift`` (the earliest gets ``day``, the others keep their distance to it).
    """
    c = ctx(request)
    url = back(request, "/dates")
    ids = ints(image_ids)
    if not ids:
        return redirect(request, url, err=_("Select at least one photo."))
    mode = mode if mode in MODES else "set"
    if mode == "suggested":
        wanted = set(ids)
        plan: dict[int, datetime] = {}
        for item in suggested:
            raw_id, _sep, raw_when = item.partition("=")
            try:
                image_id, when = int(raw_id), datetime.fromisoformat(raw_when)
            except ValueError:
                continue
            if image_id in wanted:
                plan[image_id] = when.replace(tzinfo=None)
        if not plan:
            return redirect(request, url, err=_("None of the selected photos has a suggestion."))
    else:
        day_value = _parse_day(day)
        if day_value is None:
            return redirect(request, url, err=_("Enter a date."))
        if at.strip() and _parse_time(at) is None:
            return redirect(request, url, err=_("Enter a time like 14:30, or leave it empty."))
        with c.db.session() as s:
            photos = D.load_photos(s, ids)
        plan = D.planned_dates(photos, day_value, _parse_time(at), shift=mode == "shift")
    try:
        with c.db.write() as s:
            n = D.apply_dates(s, plan)
    except ValueError:
        return redirect(request, url, err=_("That date is not plausible."))
    log.info("dates set by hand (%s) for %d photos", mode, n)
    return redirect(
        request,
        url,
        msg=_n("Date corrected for %(num)d photo.", "Dates corrected for %(num)d photos.", n),
    )
