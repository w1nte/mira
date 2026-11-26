"""Map page (heatmap of GPS-tagged photos, places) and the place actions."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select

from ...db.models import Image
from ...geo.mapdata import gps_counts, heat_points, place_detail, place_summaries
from ...i18n import MAP_JS_KEYS, _, js_strings
from ...query import ImageFilter, count_images
from ..common import back, ctx, redirect, render
from ..context import Job
from .images import _chips

router = APIRouter()

#: strings used by static/map.js


def _places(c, f: ImageFilter) -> list[dict]:
    if f.is_empty:  # the whole library: cached until the next commit
        return c.cache.get(("geo_places",), lambda s: place_summaries(s, ImageFilter()))
    with c.db.session() as s:
        return place_summaries(s, f)


@router.get("/map")
def map_page(request: Request, photo: int | None = None) -> Response:
    """``photo``: centre the map on this photo (the photo page's "Open on map")."""
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    with c.db.session() as s:
        counts = gps_counts(s, f)
        total = count_images(s, f)
        chips = _chips(s, f)
        spot = None
        if photo is not None:
            row = s.execute(select(Image.id, Image.lat, Image.lon).where(Image.id == photo)).first()
            if row is not None and row.lat is not None:
                spot = {"id": row.id, "lat": row.lat, "lon": row.lon}
    geo = c.settings.geo
    return render(
        request,
        "map.html",
        f=f,
        area=f.copy(bbox=None),
        total=total,
        counts=counts,
        places=_places(c, f),
        photo=spot,
        tile_url=geo.tile_url,
        tile_attribution=geo.tile_attribution,
        tile_max_zoom=geo.tile_max_zoom,
        nominatim=geo.nominatim,
        job=c.jobs.current if c.jobs.busy else None,
        map_i18n=js_strings(MAP_JS_KEYS),
        **chips,
    )


@router.get("/api/geo/points")
def api_points(request: Request) -> JSONResponse:
    """Heatmap data of the filtered photos: ``{"points": [[lat, lon, photos], ...], ...}``."""
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    if f.is_empty:  # the whole library: cached until the next commit
        data = c.cache.get(("geo_points",), lambda s: heat_points(s, ImageFilter()))
    else:
        with c.db.session() as s:
            data = heat_points(s, f)
    return JSONResponse(data)


@router.get("/api/geo/places")
def api_places(request: Request) -> JSONResponse:
    c = ctx(request)
    return JSONResponse({"places": _places(c, ImageFilter.from_params(request.query_params))})


@router.get("/api/geo/places/{place_id}")
def api_place(request: Request, place_id: int) -> JSONResponse:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    with c.db.session() as s:
        data = place_detail(s, place_id, f)
    if data is None:
        raise HTTPException(404)
    data["gallery"] = "/images?" + f.copy(place=place_id).query_string()
    return JSONResponse(data)


@router.post("/places/{place_id}/rename")
def rename(request: Request, place_id: int, name: str = Form("")) -> Response:
    """Name a place; the name is kept when places are found again. Empty: automatic name."""
    from ...geo.places import rename_place

    c = ctx(request)
    try:
        place = rename_place(c.db, c.settings, place_id, name)
    except LookupError as exc:
        raise HTTPException(404) from exc
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"id": place.id, "name": place.name, "user_named": place.user_named})
    msg = (
        _("Renamed to “%(name)s”.", name=place.name)
        if place.user_named
        else _("Automatic name restored: %(name)s.", name=place.name)
    )
    return redirect(request, back(request, "/map"), msg=msg)


@router.post("/places/find")
def find(request: Request, refresh: str = Form("")) -> Response:
    """Group the GPS-tagged photos into places and name them (a job)."""
    from ...geo.places import find_places

    c = ctx(request)
    full = refresh == "1"

    def job(j: Job) -> str:
        def progress(message: str) -> None:
            j.message = message

        return find_places(
            c.db,
            c.settings,
            refresh=full,
            progress=progress,
            should_stop=lambda: j.stop_requested,
        ).message

    if not c.jobs.start("places", job):
        return redirect(request, back(request, "/map"), err=_("Another job is running."))
    return redirect(request, back(request, "/map"), msg=_("Finding places…"))
