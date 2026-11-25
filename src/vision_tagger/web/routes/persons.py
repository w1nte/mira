"""Named persons: overview, rename, merge, delete, references."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ...db.models import Cluster, Face, Image, Person, PersonReference
from ...faces import clustering as C
from ...faces.ages import analyse_person, estimate_birth_year, person_faces
from ..common import back, ctx, ints, redirect, render

router = APIRouter()


@router.get("/persons")
def persons_overview(request: Request) -> Response:
    c = ctx(request)
    today = date.today()
    with c.db.session() as s:
        rows = []
        for person in s.scalars(select(Person).order_by(Person.name)):
            clusters = list(
                s.scalars(
                    select(Cluster)
                    .where(Cluster.person_id == person.id)
                    .order_by(Cluster.face_count.desc())
                )
            )
            n_images = (
                s.scalar(
                    select(func.count(func.distinct(Face.image_id)))
                    .join(Cluster, Cluster.id == Face.cluster_id)
                    .where(Cluster.person_id == person.id, Face.ignored.is_(False))
                )
                or 0
            )
            n_refs = (
                s.scalar(
                    select(func.count())
                    .select_from(PersonReference)
                    .where(PersonReference.person_id == person.id)
                )
                or 0
            )
            cover = clusters[0].cover_face_id if clusters else None
            estimate = (
                None if person.birth_year else estimate_birth_year(person_faces(s, person.id))
            )
            rows.append(
                {
                    "person": person,
                    "age": (today.year - person.birth_year) if person.birth_year else None,
                    "estimate": estimate if estimate and estimate.confidence != "low" else None,
                    "clusters": clusters,
                    "faces": sum(cl.face_count for cl in clusters),
                    "images": n_images,
                    "refs": n_refs,
                    "cover": cover,
                }
            )
    return render(request, "persons.html", rows=rows)


@router.get("/persons/{person_id}")
def person_detail(request: Request, person_id: int) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        person = s.get(Person, person_id)
        if person is None:
            return redirect(request, "/persons", err="Person not found.")
        clusters = list(
            s.scalars(
                select(Cluster)
                .where(Cluster.person_id == person_id)
                .order_by(Cluster.face_count.desc())
            )
        )
        refs = list(
            s.scalars(select(PersonReference).where(PersonReference.person_id == person_id))
        )
        suggested = list(
            s.scalars(
                select(Cluster)
                .where(Cluster.suggested_person_id == person_id, Cluster.person_id.is_(None))
                .order_by(Cluster.suggestion_score.desc())
            )
        )
        ages = analyse_person(s, person)
        image_dates = {
            i: d
            for i, d in s.execute(
                select(Image.id, Image.date_source).where(
                    Image.id.in_({f.image_id for f in ages.faces[:2000]})
                )
            )
        }
    timeline = _timeline(ages.faces)
    return render(
        request,
        "person.html",
        person=person,
        clusters=clusters,
        refs=refs,
        suggested=suggested,
        ages=ages,
        timeline=timeline,
        image_dates=image_dates,
    )


def _timeline(faces, per_year: int = 4) -> list[tuple[str, list]]:
    """Faces grouped by year ("through the years"), a few per year, best-dated first."""
    by_year: dict[str, list] = {}
    for f in faces:
        key = str(f.recorded_at.year) if f.recorded_at else "undated"
        by_year.setdefault(key, []).append(f)
    out = []
    for year in sorted(by_year, key=lambda y: (y == "undated", y)):
        items = sorted(by_year[year], key=lambda f: (f.suspicious, -f.weight))[:per_year]
        out.append((year, items))
    return out


@router.post("/persons/{person_id}/birth-year")
def set_birth_year(request: Request, person_id: int, birth_year: str = Form("")) -> Response:
    c = ctx(request)
    value = birth_year.strip()
    with c.db.write() as s:
        person = s.get(Person, person_id)
        if person is None:
            return redirect(request, "/persons", err="Person not found.")
        if not value:
            person.birth_year = None
            msg = "Birth year removed."
        elif value.isdigit() and 1850 <= int(value) <= date.today().year:
            person.birth_year = int(value)
            msg = f"Birth year set to {value}."
        else:
            return redirect(request, f"/persons/{person_id}", err="Enter a year like 1988.")
    return redirect(request, back(request, f"/persons/{person_id}"), msg=msg)


@router.post("/persons/{person_id}/rename")
def rename(request: Request, person_id: int, name: str = Form(...)) -> Response:
    c = ctx(request)
    try:
        with c.db.write() as s:
            person = C.rename_person(s, person_id, name)
            target = person.id
    except (ValueError, LookupError) as exc:
        return redirect(request, back(request, "/persons"), err=str(exc))
    url = f"/persons/{target}" if target != person_id else back(request, "/persons")
    return redirect(request, url, msg="Person renamed.")


@router.post("/persons/merge")
def merge(
    request: Request, person_ids: list[str] = Form(default_factory=list), target: str = Form("")
) -> Response:
    c = ctx(request)
    ids = ints(person_ids)
    if len(ids) < 2:
        return redirect(request, "/persons", err="Select at least two persons.")
    target_id = int(target) if target.isdigit() and int(target) in ids else ids[0]
    with c.db.write() as s:
        C.merge_persons(s, target_id, ids)
    return redirect(request, "/persons", msg=f"Merged {len(ids)} persons.")


@router.post("/persons/{person_id}/delete")
def delete(request: Request, person_id: int) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        C.delete_person(s, person_id)
    return redirect(request, "/persons", msg="Person deleted (clusters kept, now unnamed).")


@router.post("/persons/{person_id}/merge-clusters")
def merge_person_clusters(request: Request, person_id: int) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        ids = list(
            s.scalars(
                select(Cluster.id)
                .where(Cluster.person_id == person_id)
                .order_by(Cluster.face_count.desc())
            )
        )
        if len(ids) > 1:
            C.merge_clusters(s, ids[0], ids[1:])
    return redirect(request, f"/persons/{person_id}", msg="Clusters merged.")


@router.post("/references/{ref_id}/delete")
def delete_reference(request: Request, ref_id: int) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        ref = s.get(PersonReference, ref_id)
        if ref is not None:
            s.delete(ref)
    return redirect(request, back(request, "/persons"), msg="Reference removed.")
