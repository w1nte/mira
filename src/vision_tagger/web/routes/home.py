"""Home page (statistics, memories, collections) and managing saved collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ... import library
from ...db.models import StageStatus, Status
from ...i18n import _
from ..common import back, ctx, redirect, render

router = APIRouter()


@dataclass
class HomeData:
    stats: library.LibraryStats
    memories: list[library.Memory]
    recent: list[int]
    saved: list[library.CollectionView]
    suggested: list[library.CollectionView]
    pending: int  # photos still waiting for analysis


def _home_data(s) -> HomeData:
    pending = (
        s.scalar(
            select(func.count(func.distinct(StageStatus.image_id))).where(
                StageStatus.status.in_([Status.PENDING, Status.RUNNING])
            )
        )
        or 0
    )
    return HomeData(
        stats=library.stats(s),
        memories=library.on_this_day(s),
        recent=library.recently_added(s),
        saved=library.saved(s),
        suggested=library.suggested(s),
        pending=pending,
    )


@router.get("/")
def home(request: Request) -> Response:
    c = ctx(request)
    data = c.cache.get(("home", date.today().isoformat()), _home_data)
    return render(request, "home.html", d=data, today=date.today())


@router.get("/collections")
def collections_page(request: Request) -> Response:
    c = ctx(request)
    data = c.cache.get(("home", date.today().isoformat()), _home_data)
    return render(request, "collections.html", saved=data.saved, suggested=data.suggested)


@router.post("/collections")
def create(request: Request, name: str = Form(""), query: str = Form("")) -> Response:
    c = ctx(request)
    try:
        with c.db.write() as s:
            library.add(s, name, query)
    except ValueError as exc:
        return redirect(request, back(request, "/collections"), err=str(exc))
    return redirect(
        request, back(request, "/collections"), msg=_("Collection “%(name)s” saved.", name=name)
    )


@router.post("/collections/{collection_id}/rename")
def rename(request: Request, collection_id: int, name: str = Form("")) -> Response:
    c = ctx(request)
    try:
        with c.db.write() as s:
            library.rename(s, collection_id, name)
    except (ValueError, LookupError) as exc:
        return redirect(request, "/collections", err=str(exc))
    return redirect(request, "/collections", msg=_("Collection renamed."))


@router.post("/collections/{collection_id}/delete")
def delete(request: Request, collection_id: int) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        library.delete(s, collection_id)
    return redirect(request, "/collections", msg=_("Collection deleted."))


@router.post("/collections/{collection_id}/move")
def move(request: Request, collection_id: int, delta: int = Form(0)) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        library.move(s, collection_id, max(-1, min(1, delta)))
    return redirect(request, "/collections")
