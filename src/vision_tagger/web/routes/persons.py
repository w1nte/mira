"""Named persons: overview, rename, merge, delete, references."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ...db.models import Cluster, Face, Person, PersonReference
from ...faces import clustering as C
from ..common import back, ctx, ints, redirect, render

router = APIRouter()


@router.get("/persons")
def persons_overview(request: Request) -> Response:
    c = ctx(request)
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
            rows.append(
                {
                    "person": person,
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
    return render(
        request,
        "person.html",
        person=person,
        clusters=clusters,
        refs=refs,
        suggested=suggested,
    )


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
