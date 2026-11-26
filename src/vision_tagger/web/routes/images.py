"""Image gallery with filter builder, image detail and export."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ...db.models import Cluster, Face, Image, ImageProperty, Person, VlmResult
from ...export import METADATA, ExportStats, export_images, write_sidecars_in_place
from ...faces.ages import age_label, check_image_date, correct_image_date
from ...pipeline import control
from ...query import (
    ImageFilter,
    all_image_ids,
    count_images,
    date_histogram,
    neighbours,
    page_images,
    person_counts,
    property_keys,
)
from ..common import back, cached_property_counts, ctx, redirect, render
from ..context import Job

router = APIRouter()

FACET_KEYS = [
    "work_image",
    "location_type",
    "image_type",
    "event_type",
    "composition",
    "people",
    "objects",
    "tags",
    "year",
]


def _chips(s, f: ImageFilter) -> dict[str, object]:
    person_ids = set(f.persons) | set(f.exclude_persons)
    names = (
        {p.id: p.name for p in s.scalars(select(Person).where(Person.id.in_(person_ids)))}
        if person_ids
        else {}
    )
    cluster_names = {}
    if f.clusters:
        for cid, name in s.execute(
            select(Cluster.id, Person.name)
            .outerjoin(Person, Person.id == Cluster.person_id)
            .where(Cluster.id.in_(f.clusters))
        ):
            cluster_names[cid] = name or f"Cluster #{cid}"
    return {"person_names": names, "cluster_names": cluster_names}


MAX_TIMELINE_MONTHS = 1200  # 100 years; older dates are almost always wrong camera clocks


def _month(value: str | None, end: bool = False) -> str | None:
    """``2019`` / ``2019-07`` / ``2019-07-14`` -> the month it starts (or ends) in."""
    if not value or len(value) < 4 or not value[:4].isdigit():
        return None
    if len(value) >= 7 and value[5:7].isdigit():
        return value[:7]
    return f"{value[:4]}-{'12' if end else '01'}"


def _month_range(first: str, last: str) -> list[str]:
    y, m = int(first[:4]), int(first[5:7])
    ly, lm = int(last[:4]), int(last[5:7])
    out = []
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out[-MAX_TIMELINE_MONTHS:]


def _timeline(c, s, f: ImageFilter) -> dict | None:
    """Photos per month for the timeline slider: all other filters apply, the dates do not
    (so the selection can be moved anywhere)."""
    base = f.copy(date_from=None, date_to=None)
    if base.is_empty:
        hist = c.cache.get(
            ("timeline", "month"), lambda s2: date_histogram(s2, ImageFilter(), "month")
        )
    else:
        hist = date_histogram(s, base, "month")
    hist = [(b, n) for b, n in hist if b]
    if not hist:
        return None
    counts = dict(hist)
    months = _month_range(hist[0][0], hist[-1][0])
    return {
        "months": months,
        "counts": [counts.get(m, 0) for m in months],
        "from": _month(f.date_from),
        "to": _month(f.date_to, end=True),
        "base": base.query_string(),
    }


@router.get("/images")
def gallery(request: Request) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    size = c.settings.web.page_size
    with c.db.session() as s:
        total = count_images(s, f)
        images = page_images(s, f, 0, size)
        facets_all = cached_property_counts(c, s, f, keys=FACET_KEYS)
        facets = {k: facets_all[k][:12] for k in FACET_KEYS if k in facets_all}
        people = person_counts(s, f)[:30]
        all_persons = list(s.scalars(select(Person).order_by(Person.name)))
        keys = property_keys(s)
        cameras = [
            cam
            for (cam,) in s.execute(
                select(Image.camera)
                .where(Image.camera.is_not(None))
                .group_by(Image.camera)
                .order_by(func.count().desc())
                .limit(50)
            )
        ]
        chips = _chips(s, f)
        timeline = _timeline(c, s, f)
    return render(
        request,
        "images.html",
        f=f,
        total=total,
        timeline=timeline,
        images=images,
        next_page=2 if total > size else None,
        facets=facets,
        people=people,
        all_persons=all_persons,
        keys=keys,
        cameras=cameras,
        export_mode=c.settings.export.mode,
        **chips,
    )


@router.get("/images/page")
def gallery_page(request: Request, page: int = 2) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    size = c.settings.web.page_size
    with c.db.session() as s:
        total = count_images(s, f)
        images = page_images(s, f, (page - 1) * size, size)
    return render(
        request,
        "partials/image_grid.html",
        images=images,
        f=f,
        next_page=page + 1 if page * size < total else None,
    )


@router.get("/images/add")
def add_filter(request: Request) -> Response:
    """Add a person / property chip to the current filter and redirect back to the gallery."""
    params = request.query_params
    f = ImageFilter.from_params(params)
    target = params.get("target", "/images")
    if target not in ("/images", "/overview", "/properties"):
        target = "/images"
    exclude = params.get("mode") == "exclude"
    if params.get("person", "").isdigit():
        pid = int(params["person"])
        if exclude:
            f.exclude_persons.append(pid)
        elif pid not in f.persons:
            f.persons.append(pid)
    key, value = params.get("key", "").strip(), params.get("value", "").strip()
    if key and value:
        (f.exclude_props if exclude else f.props).append((key, value))
    return redirect(request, f"{target}?{f.query_string()}")


@router.get("/images/download.zip")
def download_zip(request: Request) -> Response:
    """The photos of the current filter as a ZIP, with XMP sidecars (remote mode export)."""
    from fastapi.responses import StreamingResponse

    from ...export import stream_zip

    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    with c.db.session() as s:
        ids = all_image_ids(s, f)
    return StreamingResponse(
        stream_zip(c.db, ids),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="vision-tagger-photos.zip"'},
    )


@router.get("/images/{image_id}")
def image_detail(request: Request, image_id: int) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        img = s.scalar(
            select(Image)
            .options(
                selectinload(Image.stages),
                selectinload(Image.duplicates),
                selectinload(Image.properties),
                selectinload(Image.faces).selectinload(Face.cluster).selectinload(Cluster.person),
            )
            .where(Image.id == image_id)
        )
        if img is None:
            raise HTTPException(404)
        vlm = s.get(VlmResult, image_id)
        props: dict[str, list[ImageProperty]] = {}
        for p in sorted(img.properties, key=lambda p: (p.key, -(p.confidence or 0))):
            props.setdefault(p.key, []).append(p)
        f = ImageFilter.from_params(request.query_params)
        # neighbours inside the current filter and sort order (keyset queries)
        prev_id, next_id = neighbours(s, f, image_id)
        date_check = check_image_date(s, image_id)
    face_ages = {}
    if date_check is not None:
        face_ages = {
            fid: age_label(info, date_check.suspicious) for fid, info in date_check.faces.items()
        }
    return render(
        request,
        "image.html",
        img=img,
        vlm=vlm,
        props=props,
        f=f,
        prev_id=prev_id,
        next_id=next_id,
        date_check=date_check,
        face_ages=face_ages,
    )


@router.post("/images/{image_id}/date")
def set_date(
    request: Request, image_id: int, year: str = Form(...), month: str = Form("")
) -> Response:
    c = ctx(request)
    try:
        with c.db.write() as s:
            correct_image_date(
                s, image_id, int(year), int(month) if month.strip().isdigit() else None
            )
    except (ValueError, LookupError) as exc:
        return redirect(request, back(request, f"/images/{image_id}"), err=str(exc))
    return redirect(request, back(request, f"/images/{image_id}"), msg=f"Date set to {year}.")


@router.post("/images/{image_id}/requeue")
def requeue(request: Request, image_id: int, stage: str = Form(...)) -> Response:
    c = ctx(request)
    n = control.requeue(c.db, stage, [image_id])
    return redirect(
        request,
        back(request, f"/images/{image_id}"),
        msg=f"Re-queued for {stage}." if n else "Could not re-queue (running?).",
    )


@router.post("/images/export")
def export(
    request: Request,
    target: str = Form(...),
    mode: str = Form("copy"),
    layout: str = Form("flat"),
    query: str = Form(""),
    metadata: list[str] = Form([]),
) -> Response:
    from starlette.datastructures import QueryParams

    c = ctx(request)
    if c.settings.remote_mode:  # remote users may not write to folders on the server
        raise HTTPException(403, "use the ZIP download in remote mode")
    f = ImageFilter.from_params(QueryParams(query))
    if not target.strip():
        return redirect(request, f"/images?{query}", err="Enter a target folder.")
    mode = mode if mode in ("copy", "hardlink", "symlink") else "copy"
    layout = layout if layout in ("flat", "date", "folder") else "flat"

    def job(j: Job) -> str:
        j.message = f"exporting to {target}"
        stats = export_images(
            c.db,
            f,
            target.strip(),
            mode=mode,  # type: ignore[arg-type]
            layout=layout,  # type: ignore[arg-type]
            metadata=[m for m in metadata if m in METADATA],
            sidecar_naming=c.settings.export.sidecar_naming,
        )
        return export_summary(stats)

    if not c.jobs.start("export", job):
        return redirect(request, f"/images?{query}", err="Another job is running.")
    return redirect(request, f"/images?{query}", msg="Export started – see dashboard for progress.")


def export_summary(stats: ExportStats, what: str = "Export") -> str:
    parts = [f"{what} to {stats.target}: {stats.exported} files"]
    if stats.sidecars:
        parts.append(f"{stats.sidecars} XMP sidecars")
    if stats.embedded:
        parts.append(f"metadata written into {stats.embedded} copies")
    if stats.fallbacks:
        parts.append(f"{stats.fallbacks} copied instead of linked")
    parts.append(f"{stats.skipped} skipped, {len(stats.errors)} errors")
    return ", ".join(parts)


@router.post("/images/sidecars")
def sidecars(request: Request, query: str = Form("")) -> Response:
    """XMP sidecars next to the originals, for libraries managed in place by other apps."""
    from starlette.datastructures import QueryParams

    c = ctx(request)
    f = ImageFilter.from_params(QueryParams(query))

    def job(j: Job) -> str:
        j.message = "writing XMP sidecars next to the originals"
        stats = write_sidecars_in_place(c.db, f, naming=c.settings.export.sidecar_naming)
        kept = f", {stats.kept} sidecars of other apps kept" if stats.kept else ""
        return (
            f"XMP sidecars: {stats.sidecars} written{kept}, {stats.skipped} photos missing, "
            f"{len(stats.errors)} errors"
        )

    if not c.jobs.start("sidecars", job):
        return redirect(request, f"/images?{query}", err="Another job is running.")
    return redirect(
        request, f"/images?{query}", msg="Writing XMP sidecars – see dashboard for progress."
    )
