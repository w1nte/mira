"""Data overview: timeline (year -> month -> day), metadata breakdowns and a metadata table."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ...db.models import Face, Image, StageStatus
from ...query import (
    ImageFilter,
    column_counts,
    conditions,
    date_histogram,
    folder_children,
    property_counts,
)
from ..common import ctx, render

router = APIRouter()

TABLE_SORTS = {
    "date": Image.recorded_at,
    "name": Image.filename,
    "path": Image.rel_path,
    "size": Image.size_bytes,
    "camera": Image.camera,
    "width": Image.width,
}


def _level(f: ImageFilter, requested: str | None) -> str:
    if requested in ("year", "month", "day"):
        return requested
    if f.date_from and f.date_to and f.date_from[:4] == f.date_to[:4]:
        if len(f.date_from) >= 7 and f.date_from[:7] == f.date_to[:7]:
            return "day"
        return "month"
    return "year"


@router.get("/overview")
def overview(request: Request, level: str | None = None) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    lvl = _level(f, level)
    with c.db.session() as s:
        hist = date_histogram(s, f, lvl)
        undated = (
            s.scalar(
                select(func.count())
                .select_from(Image)
                .where(*conditions(f), Image.recorded_at.is_(None))
            )
            or 0
        )
        total = s.scalar(select(func.count()).select_from(Image).where(*conditions(f))) or 0
        folders = folder_children(s, f)
        cameras = column_counts(s, f, Image.camera, limit=20)
        exts = column_counts(s, f, Image.ext, limit=20)
        date_sources = column_counts(s, f, Image.date_source, limit=10)
        size_total = s.scalar(select(func.sum(Image.size_bytes)).where(*conditions(f))) or 0
        props = property_counts(s, f)
        with_faces = (
            s.scalar(
                select(func.count(func.distinct(Face.image_id))).where(
                    Face.image_id.in_(select(Image.id).where(*conditions(f))),
                    Face.ignored.is_(False),
                )
            )
            or 0
        )
    breakdowns = {
        k: props.get(k, [])
        for k in ("work_image", "capture_device", "orientation", "image_type", "setting")
        if props.get(k)
    }
    return render(
        request,
        "overview.html",
        f=f,
        level=lvl,
        hist=hist,
        hist_max=max((n for _, n in hist), default=0),
        undated=undated,
        total=total,
        folders=folders,
        cameras=cameras,
        exts=exts,
        date_sources=date_sources,
        size_total=size_total,
        breakdowns=breakdowns,
        with_faces=with_faces,
    )


@router.get("/overview/table")
def table(request: Request, page: int = 1, order: str = "date", desc: int = 1) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    page = max(page, 1)
    size = 100
    col = TABLE_SORTS.get(order, Image.recorded_at)
    face_count = (
        select(func.count(Face.id))
        .where(Face.image_id == Image.id, Face.ignored.is_(False))
        .scalar_subquery()
    )
    vlm_status = (
        select(StageStatus.status)
        .where(StageStatus.image_id == Image.id, StageStatus.stage == "vlm")
        .scalar_subquery()
    )
    faces_status = (
        select(StageStatus.status)
        .where(StageStatus.image_id == Image.id, StageStatus.stage == "faces")
        .scalar_subquery()
    )
    with c.db.session() as s:
        total = s.scalar(select(func.count()).select_from(Image).where(*conditions(f))) or 0
        rows = s.execute(
            select(Image, face_count, faces_status, vlm_status)
            .where(*conditions(f))
            .order_by(col.desc() if desc else col.asc(), Image.id)
            .offset((page - 1) * size)
            .limit(size)
        ).all()
    return render(
        request,
        "table.html",
        f=f,
        rows=rows,
        total=total,
        page=page,
        pages=max(1, (total + size - 1) // size),
        order=order,
        desc=desc,
    )
