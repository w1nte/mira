"""Generations: grow-up timelapse per person and same-age family comparisons."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from sqlalchemy import select

from ...access import checker
from ...db.models import Person
from ...faces.generations import (
    FramesOf,
    aligned_portrait,
    growup_animation,
    growup_frames,
    pair_frames,
    resemblance,
)
from ...i18n import _
from ..common import ctx, redirect, render
from ..nsfw import covered_ids

router = APIRouter()
MEANINGFUL_SIMILARITY = 0.2


def _cache(request: Request):
    return ctx(request).settings.cache_dir / "aligned"


def _frames_of(request: Request) -> FramesOf:
    """Grow-up frames per person, cached until the database changes.

    The overview and the look-alike ranking need the frames of *every* person.
    """
    cache = ctx(request).cache
    return lambda pid: cache.get(("growup_frames", pid), lambda s: growup_frames(s, pid))


@router.get("/generations")
def overview(request: Request) -> Response:
    c = ctx(request)
    frames_of = _frames_of(request)
    with c.db.session() as s:
        persons = list(s.scalars(select(Person).order_by(Person.name)))
    cards = []
    for person in persons:  # (cached) frames use their own sessions, none is held meanwhile
        frames = frames_of(person.id)
        if not frames:
            continue
        ages = [f.age for f in frames if f.age is not None]
        cards.append(
            {
                "person": person,
                "frames": len(frames),
                "first": frames[0],
                "last": frames[-1],
                "span": (min(ages), max(ages)) if ages else None,
            }
        )
    cards.sort(key=lambda c: -(c["span"][1] - c["span"][0] if c["span"] else 0))
    return render(request, "generations.html", cards=cards)


@router.get("/generations/{person_id}")
def person_view(request: Request, person_id: int, compare: int | None = None) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        person = s.get(Person, person_id)
        others = [p for p in s.scalars(select(Person).order_by(Person.name)) if p.id != person_id]
    if person is None:
        return redirect(request, "/generations", err=_("Person not found."))
    frames_of = _frames_of(request)
    frames = frames_of(person_id)
    covered = set(covered_ids(request, [f.image_id for f in frames]))
    if covered:  # faces from NSFW photos stay out of the grow-up viewer
        frames = [f for f in frames if f.image_id not in covered]
    looks_like = c.cache.get(
        ("resemblance", person_id), lambda s: resemblance(s, person_id, frames_of)
    )[:5]
    payload = [
        {"src": f"/media/aligned/{f.face_id}", "label": f.label, "image": f.image_id}
        for f in frames
    ]
    return render(
        request,
        "generations_person.html",
        person=person,
        frames=frames,
        payload=payload,
        looks_like=looks_like,
        others=others,
    )


@router.get("/generations/{person_id}/compare")
def compare_pick(request: Request, person_id: int, other: int) -> Response:
    """Target of the "compare with" select (a plain GET form, no inline JavaScript)."""
    return RedirectResponse(f"/generations/{person_id}/compare/{other}", status_code=303)


@router.get("/generations/{person_id}/compare/{other_id}")
def compare_view(request: Request, person_id: int, other_id: int) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        a, b = s.get(Person, person_id), s.get(Person, other_id)
        others = [p for p in s.scalars(select(Person).order_by(Person.name)) if p.id != person_id]
    if a is None or b is None:
        return redirect(request, "/generations", err=_("Person not found."))
    frames_of = _frames_of(request)
    frames_a, frames_b = frames_of(person_id), frames_of(other_id)
    with c.db.session() as s:
        pairs = pair_frames(s, frames_a, frames_b)
    best = max(pairs, key=lambda p: p.similarity) if pairs else None
    if best is not None and best.similarity < MEANINGFUL_SIMILARITY:
        best = None  # no "most alike" badge between strangers
    avg = sum(p.similarity for p in pairs) / len(pairs) if pairs else None
    return render(
        request,
        "generations_compare.html",
        a=a,
        b=b,
        pairs=pairs,
        best=best,
        avg=avg,
        others=others,
    )


@router.get("/generations/{person_id}/animation.webp")
def animation(request: Request, person_id: int, download: int = 0) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        person = s.get(Person, person_id)
        if person is None:
            raise HTTPException(404)
        result = growup_animation(s, _cache(request), person_id, allowed=checker(c.settings))
    if result is None:
        raise HTTPException(404, "not enough photos at different ages")
    data, _frames = result
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        safe = re.sub(r"[^\w\- ]", "", person.name).strip() or "person"
        headers["Content-Disposition"] = f'attachment; filename="{safe} growing up.webp"'
    return Response(data, media_type="image/webp", headers=headers)


@router.get("/media/aligned/{face_id}")
def aligned(request: Request, face_id: int, size: int = 512) -> Response:
    size = 512 if size not in (256, 384, 512) else size
    c = ctx(request)
    with c.db.session() as s:
        path = aligned_portrait(s, _cache(request), face_id, size, allowed=checker(c.settings))
    if path is None:
        raise HTTPException(404)
    return FileResponse(
        path, media_type="image/webp", headers={"Cache-Control": "private, max-age=86400"}
    )
