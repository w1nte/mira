"""Image gallery with filter builder, image detail and export."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ...db import transfer
from ...db.models import Animal, Cluster, Face, Image, ImageProperty, Person, Pet, Place, VlmResult
from ...db.util import chunked
from ...export import METADATA, ExportStats, export_images, write_sidecars_in_place
from ...faces.ages import age_label, check_image_date, correct_image_date
from ...i18n import Lazy, LazyJoin, _
from ...nsfw import KEY as NSFW_KEY
from ...nsfw import YES
from ...pipeline import control
from ...query import (
    ImageFilter,
    _pairs,
    all_image_ids,
    count_images,
    date_histogram,
    neighbours,
    page_images,
    person_counts,
    pet_counts,
    property_keys,
)
from ..common import back, cached_property_counts, ctx, pet_label, redirect, render
from ..context import Job
from ..nsfw import shown as nsfw_shown

router = APIRouter()

FACET_KEYS = [
    "work_image",
    "place",
    "location_type",
    "image_type",
    "event_type",
    "composition",
    "people",
    "animal",
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
            cluster_names[cid] = name or _("Cluster #%(id)s", id=cid)
    place_names: dict[int, str] = {}
    for part in chunked(f.places):
        place_names.update(s.execute(select(Place.id, Place.name).where(Place.id.in_(part))).all())
    pet_ids = set(f.pets) | set(f.exclude_pets)
    pet_names = (
        {p.id: pet_label(p) for p in s.scalars(select(Pet).where(Pet.id.in_(pet_ids)))}
        if pet_ids
        else {}
    )
    return {
        "person_names": names,
        "cluster_names": cluster_names,
        "place_names": place_names,
        "pet_names": pet_names,
    }


def _alternatives(c, s, f: ImageFilter) -> dict[str, object]:
    """What could be added with "or": the facets without their own condition.

    A facet of a key the filter has a condition on counts the photos matching everything else
    (one more count per such key); persons and pets likewise without the persons / pets.
    """
    facets = {}
    for key in FACET_KEYS:
        if key in f.prop_keys:
            counts = cached_property_counts(c, s, f.without_key(key), key=key)
            facets[key] = counts.get(key, [])[:12]
    people: list = []
    pets: list = []
    if f.who_alternatives:
        others = f.copy(persons=[], clusters=[], pets=[])
        people = person_counts(s, others)[:14]
        pets = pet_counts(s, others)[:14]
    return {"facets": facets, "people": people, "pets": pets}


def _nsfw_filter(c, s, request: Request, f: ImageFilter) -> dict | None:
    """The gallery's NSFW section: hide them / only them (``xt`` / ``t=nsfw:yes``, so "or"
    and collections work as with any property). ``None`` while the library has none."""
    pair = (NSFW_KEY, YES)
    hidden = pair in f.exclude_props
    only = f.has_prop(*pair)
    library = dict(cached_property_counts(c, s, None, key=NSFW_KEY).get(NSFW_KEY, []))
    if not (library.get(YES) or hidden or only or nsfw_shown(request)):
        return None
    base = f.copy(
        props=[p for p in f.props if p != pair],
        prop_groups=[[a for a in g if a != pair] for g in f.prop_groups],
        exclude_props=[p for p in f.exclude_props if p != pair],
    )
    counts = dict(cached_property_counts(c, s, base, key=NSFW_KEY).get(NSFW_KEY, []))
    return {
        "hidden": hidden,
        "only": only,
        "count": counts.get(YES, 0),
        "hide_qs": base.query_string(exclude_props=[*base.exclude_props, pair])
        if not hidden
        else base.query_string(),
        "only_qs": base.query_string(props=[*base.props, pair])
        if not only
        else base.query_string(),
    }


def _people(s, images: list[Image]) -> dict[int, list[str]]:
    """The named persons on each photo of a gallery page (the lightbox's info bar): one
    indexed query per page."""
    out: dict[int, list[str]] = {}
    for part in chunked([img.id for img in images]):
        for image_id, name in s.execute(
            select(Face.image_id, Person.name)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .where(Face.image_id.in_(part), Face.ignored.is_(False))
            .distinct()
            .order_by(Person.name)
        ):
            out.setdefault(image_id, []).append(name)
    return out


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
    base = f.copy(date_from=None, date_to=None, dates=[])
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
        people_on = _people(s, images)
        facets_all = cached_property_counts(c, s, f, keys=FACET_KEYS)
        facets = {k: facets_all[k][:12] for k in FACET_KEYS if k in facets_all}
        people = person_counts(s, f)[:30]
        all_persons = list(s.scalars(select(Person).order_by(Person.name)))
        pets = pet_counts(s, f)[:30]
        alternatives = _alternatives(c, s, f)
        all_pets = list(s.scalars(select(Pet).where(Pet.name.is_not(None)).order_by(Pet.name)))
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
        nsfw = _nsfw_filter(c, s, request, f)
    return render(
        request,
        "images.html",
        f=f,
        total=total,
        timeline=timeline,
        images=images,
        people_on=people_on,
        next_page=2 if total > size else None,
        facets=facets,
        facet_keys=[k for k in FACET_KEYS if k in facets or k in alternatives["facets"]],
        alternatives=alternatives,
        people=people,
        all_persons=all_persons,
        pets=pets,
        all_pets=all_pets,
        keys=keys,
        cameras=cameras,
        export_mode=c.settings.export.mode,
        nsfw=nsfw,
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
        people_on = _people(s, images)
    return render(
        request,
        "partials/image_grid.html",
        images=images,
        people_on=people_on,
        f=f,
        next_page=page + 1 if page * size < total else None,
    )


@router.get("/images/add")
def add_filter(request: Request) -> Response:
    """Add persons, pets or properties to the current filter and redirect back to the gallery."""
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
    if params.get("pet", "").isdigit():
        pet_id = int(params["pet"])
        if exclude:
            if pet_id not in f.exclude_pets:
                f.exclude_pets.append(pet_id)
        elif pet_id not in f.pets:
            f.pets.append(pet_id)
    key, value = params.get("key", "").strip(), params.get("value", "").strip()
    if key and value:
        (f.exclude_props if exclude else f.props).append((key, value))
    # values picked on the properties page: all required, or one OR condition ("any")
    picked = [p for p in _pairs(params.getlist("pick")) if not f.has_prop(*p)]
    if params.get("combine") == "any" and len(picked) > 1:
        f = f.copy(prop_groups=[*f.prop_groups, picked])
    else:
        f = f.copy(props=[*f.props, *picked])
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
        headers={"Content-Disposition": 'attachment; filename="mira-photos.zip"'},
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
                selectinload(Image.animals).selectinload(Animal.pet),
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
        place = s.get(Place, img.place_id) if img.place_id is not None else None
        animals = [a for a in img.animals if not a.ignored]
    face_ages = {}
    if date_check is not None:
        face_ages = {
            fid: age_label(info, date_check.suspicious) for fid, info in date_check.faces.items()
        }
    busy = None
    if img.missing:  # why "Remove from library" has to wait (like the Missing files page)
        busy = transfer.busy_reason(c.db, runner_running=c.runner.running, jobs_busy=c.jobs.busy)
    return render(
        request,
        "image.html",
        busy=str(busy) if busy else None,
        img=img,
        vlm=vlm,
        props=props,
        f=f,
        prev_id=prev_id,
        next_id=next_id,
        date_check=date_check,
        face_ages=face_ages,
        place=place,
        animals=animals,
    )


@router.post("/images/{image_id}/date")
def set_date(
    request: Request, image_id: int, year: str = Form(...), month: str = Form("")
) -> Response:
    c = ctx(request)
    url = back(request, f"/images/{image_id}")
    try:
        with c.db.write() as s:
            correct_image_date(
                s, image_id, int(year), int(month) if month.strip().isdigit() else None
            )
    except LookupError:
        return redirect(request, url, err=_("Photo not found."))
    except ValueError:
        return redirect(request, url, err=_("Implausible year: %(year)s", year=year))
    return redirect(request, url, msg=_("Date set to %(year)s.", year=year))


@router.post("/images/{image_id}/requeue")
def requeue(request: Request, image_id: int, stage: str = Form(...)) -> Response:
    c = ctx(request)
    n = control.requeue(c.db, stage, [image_id])
    return redirect(
        request,
        back(request, f"/images/{image_id}"),
        msg=_("Re-queued for %(stage)s.", stage=stage)
        if n
        else _("Could not re-queue (running?)."),
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
        return redirect(request, f"/images?{query}", err=_("Enter a target folder."))
    mode = mode if mode in ("copy", "hardlink", "symlink") else "copy"
    layout = layout if layout in ("flat", "date", "folder") else "flat"

    def job(j: Job) -> Lazy:
        j.message = Lazy("exporting to %(target)s", target=target)
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
        return redirect(request, f"/images?{query}", err=_("Another job is running."))
    return redirect(
        request, f"/images?{query}", msg=_("Export started – see dashboard for progress.")
    )


def export_summary(stats: ExportStats, search: bool = False) -> Lazy:
    """Result of an export job (made in the job thread: translated when shown)."""
    if search:
        first = Lazy(
            "Search export to %(target)s: %(n)s files", target=stats.target, n=stats.exported
        )
    else:
        first = Lazy("Export to %(target)s: %(n)s files", target=stats.target, n=stats.exported)
    parts: list[Lazy] = [first]
    if stats.sidecars:
        parts.append(Lazy("%(n)s XMP sidecars", n=stats.sidecars))
    if stats.embedded:
        parts.append(Lazy("metadata written into %(n)s copies", n=stats.embedded))
    if stats.fallbacks:
        parts.append(Lazy("%(n)s copied instead of linked", n=stats.fallbacks))
    parts.append(
        Lazy(
            "%(skipped)s skipped, %(errors)s errors",
            skipped=stats.skipped,
            errors=len(stats.errors),
        )
    )
    return LazyJoin(", ", parts)


@router.post("/images/sidecars")
def sidecars(request: Request, query: str = Form("")) -> Response:
    """XMP sidecars next to the originals, for libraries managed in place by other apps."""
    from starlette.datastructures import QueryParams

    c = ctx(request)
    f = ImageFilter.from_params(QueryParams(query))

    def job(j: Job) -> Lazy:
        j.message = Lazy("writing XMP sidecars next to the originals")
        stats = write_sidecars_in_place(c.db, f, naming=c.settings.export.sidecar_naming)
        variables = {
            "written": stats.sidecars,
            "kept": stats.kept,
            "missing": stats.skipped,
            "errors": len(stats.errors),
        }
        if stats.kept:
            return Lazy(
                "XMP sidecars: %(written)s written, %(kept)s sidecars of other apps kept, "
                "%(missing)s photos missing, %(errors)s errors",
                **variables,
            )
        return Lazy(
            "XMP sidecars: %(written)s written, %(missing)s photos missing, %(errors)s errors",
            **variables,
        )

    if not c.jobs.start("sidecars", job):
        return redirect(request, f"/images?{query}", err=_("Another job is running."))
    return redirect(
        request, f"/images?{query}", msg=_("Writing XMP sidecars – see dashboard for progress.")
    )
