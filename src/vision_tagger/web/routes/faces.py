"""Face cluster overview and cluster editing (rename, merge, split, move, ignore)."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from ...db.models import Cluster, Face, Image, Person
from ...db.util import like_escape
from ...faces import clustering as C
from ...i18n import _, _n
from ..common import back, ctx, ints, redirect, render

router = APIRouter()

PAGE = 120


def _persons(s) -> list[Person]:
    return list(s.scalars(select(Person).order_by(Person.name)))


@router.get("/faces")
def clusters_overview(
    request: Request, sort: str = "size", q: str = "", page: int = 1, show: str = "all"
) -> Response:
    c = ctx(request)
    page = max(page, 1)
    with c.db.session() as s:
        stmt = select(Cluster).options(
            selectinload(Cluster.person), selectinload(Cluster.suggested_person)
        )
        if q:
            stmt = stmt.outerjoin(Person, Person.id == Cluster.person_id).where(
                Person.name.ilike(f"%{like_escape(q)}%", escape="\\")
            )
        if show == "unnamed":
            stmt = stmt.where(Cluster.person_id.is_(None))
        elif show == "named":
            stmt = stmt.where(Cluster.person_id.is_not(None))
        elif show == "suggested":
            stmt = stmt.where(Cluster.person_id.is_(None), Cluster.suggested_person_id.is_not(None))
        if sort == "unnamed":
            stmt = stmt.order_by(Cluster.person_id.is_not(None), Cluster.face_count.desc())
        elif sort == "newest":
            stmt = stmt.order_by(Cluster.id.desc())
        else:
            stmt = stmt.order_by(Cluster.face_count.desc(), Cluster.id)
        total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        clusters = list(s.scalars(stmt.offset((page - 1) * PAGE).limit(PAGE)))
        unclustered = (
            s.scalar(
                select(func.count())
                .select_from(Face)
                .where(Face.cluster_id.is_(None), Face.ignored.is_(False))
            )
            or 0
        )
        ignored = s.scalar(select(func.count()).select_from(Face).where(Face.ignored)) or 0
        persons = _persons(s)
    return render(
        request,
        "faces.html",
        clusters=clusters,
        total=total,
        page=page,
        pages=max(1, (total + PAGE - 1) // PAGE),
        sort=sort,
        q=q,
        show=show,
        unclustered=unclustered,
        ignored=ignored,
        persons=persons,
    )


def _render_card(request: Request, cluster_id: int) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        cluster = s.scalar(
            select(Cluster)
            .options(selectinload(Cluster.person), selectinload(Cluster.suggested_person))
            .where(Cluster.id == cluster_id)
        )
        if cluster is None:
            return Response("")
        persons = _persons(s)
    return render(request, "partials/cluster_card.html", cl=cluster, persons=persons)


@router.post("/clusters/{cluster_id}/name")
def name_cluster(
    request: Request, cluster_id: int, name: str = Form(""), card: str = Form("")
) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        if s.get(Cluster, cluster_id) is None:
            raise HTTPException(404)
        C.name_cluster(
            s, cluster_id, name, references=c.settings.faces.references_per_named_cluster
        )
    if card and request.headers.get("HX-Request"):
        return _render_card(request, cluster_id)
    return redirect(request, back(request, f"/clusters/{cluster_id}"), msg=_("Name saved."))


@router.post("/clusters/{cluster_id}/accept")
def accept(request: Request, cluster_id: int, card: str = Form("")) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        C.accept_suggestion(s, cluster_id, references=c.settings.faces.references_per_named_cluster)
    if card and request.headers.get("HX-Request"):
        return _render_card(request, cluster_id)
    return redirect(
        request, back(request, f"/clusters/{cluster_id}"), msg=_("Suggestion accepted.")
    )


@router.post("/clusters/merge")
def merge(
    request: Request, cluster_ids: list[str] = Form(default_factory=list), target: str = Form("")
) -> Response:
    c = ctx(request)
    ids = ints(cluster_ids)
    if len(ids) < 2:
        return redirect(request, back(request, "/faces"), err=_("Select at least two clusters."))
    with c.db.write() as s:
        clusters = list(s.scalars(select(Cluster).where(Cluster.id.in_(ids))))
        if target.isdigit() and int(target) in ids:
            target_id = int(target)
        else:  # prefer a named cluster, then the largest
            target_id = max(clusters, key=lambda x: (x.person_id is not None, x.face_count)).id
        C.merge_clusters(s, target_id, ids)
    return redirect(request, back(request, "/faces"), msg=_("Merged %(n)s clusters.", n=len(ids)))


@router.get("/clusters/{cluster_id}")
def cluster_detail(request: Request, cluster_id: int, page: int = 1) -> Response:
    c = ctx(request)
    page = max(page, 1)
    with c.db.session() as s:
        cluster = s.scalar(
            select(Cluster)
            .options(selectinload(Cluster.person), selectinload(Cluster.suggested_person))
            .where(Cluster.id == cluster_id)
        )
        if cluster is None:
            return redirect(
                request, "/faces", err=_("Cluster %(id)s no longer exists.", id=cluster_id)
            )
        faces = list(
            s.scalars(
                select(Face)
                .where(Face.cluster_id == cluster_id, Face.ignored.is_(False))
                .order_by((Face.det_score * Face.face_px).desc(), Face.id)
                .offset((page - 1) * PAGE)
                .limit(PAGE)
            )
        )
        similar_ids = C.similar_clusters(s, cluster_id, limit=8)
        sim_map = dict(similar_ids)
        similar = [
            (cl, sim_map[cl.id])
            for cl in s.scalars(
                select(Cluster)
                .options(selectinload(Cluster.person))
                .where(Cluster.id.in_(list(sim_map)))
            )
        ]
        similar.sort(key=lambda x: -x[1])
        n_images = (
            s.scalar(
                select(func.count(func.distinct(Face.image_id))).where(
                    Face.cluster_id == cluster_id
                )
            )
            or 0
        )
        persons = _persons(s)
        other_clusters = (
            list(
                s.scalars(
                    select(Cluster)
                    .options(selectinload(Cluster.person))
                    .where(Cluster.person_id == cluster.person_id, Cluster.id != cluster_id)
                )
            )
            if cluster.person_id
            else []
        )
    return render(
        request,
        "cluster.html",
        cl=cluster,
        faces=faces,
        page=page,
        pages=max(1, (cluster.face_count + PAGE - 1) // PAGE),
        similar=similar,
        n_images=n_images,
        persons=persons,
        other_clusters=other_clusters,
    )


@router.post("/clusters/{cluster_id}/auto-split")
def auto_split(request: Request, cluster_id: int) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        new = C.auto_split(s, cluster_id, c.settings.faces.split_eps)
    if not new:
        return redirect(request, f"/clusters/{cluster_id}", msg=_("No clear sub-groups found."))
    return redirect(
        request,
        f"/clusters/{cluster_id}",
        msg=_n("Split off %(num)d new cluster.", "Split off %(num)d new clusters.", len(new)),
    )


# --------------------------------------------------------------------------- face selections
@router.post("/faces/action")
def faces_action(
    request: Request,
    action: str = Form(...),
    face_ids: list[str] = Form(default_factory=list),
    name: str = Form(""),
    target_cluster: str = Form(""),
) -> Response:
    c = ctx(request)
    ids = ints(face_ids)
    url = back(request, "/faces")
    if not ids:
        return redirect(request, url, err=_("No faces selected."))
    n = len(ids)
    with c.db.write() as s:
        if action == "split":
            new_id = C.split_faces(s, ids)
            if name.strip() and new_id:
                C.name_cluster(s, new_id, name)
            msg = _n(
                "%(num)d face moved to a new cluster.", "%(num)d faces moved to a new cluster.", n
            )
            return redirect(request, url, msg=msg)
        if action == "person":
            if not name.strip():
                return redirect(request, url, err=_("Enter a person name."))
            C.move_faces_to_person(s, ids, name)
            msg = _n(
                "%(num)d face assigned to %(name)s.",
                "%(num)d faces assigned to %(name)s.",
                n,
                name=name.strip(),
            )
            return redirect(request, url, msg=msg)
        if action == "cluster":
            if not target_cluster.isdigit() or s.get(Cluster, int(target_cluster)) is None:
                return redirect(request, url, err=_("Unknown target cluster."))
            C.move_faces(s, ids, int(target_cluster))
            return redirect(request, url, msg=_n("%(num)d face moved.", "%(num)d faces moved.", n))
        if action == "ignore":
            C.ignore_faces(s, ids, True)
            msg = _n("%(num)d face removed (ignored).", "%(num)d faces removed (ignored).", n)
            return redirect(request, url, msg=msg)
        if action == "unignore":
            C.ignore_faces(s, ids, False)
            return redirect(
                request, url, msg=_n("%(num)d face restored.", "%(num)d faces restored.", n)
            )
    return redirect(request, url, err=_("Unknown action %(action)s.", action=action))


def _face_list(request: Request, title: str, where, page: int, mode: str) -> Response:
    c = ctx(request)
    page = max(page, 1)
    with c.db.session() as s:
        total = s.scalar(select(func.count()).select_from(Face).where(where)) or 0
        faces = list(
            s.scalars(
                select(Face)
                .where(where)
                .order_by((Face.det_score * Face.face_px).desc(), Face.id)
                .offset((page - 1) * PAGE)
                .limit(PAGE)
            )
        )
        persons = _persons(s)
    return render(
        request,
        "face_list.html",
        title=title,
        faces=faces,
        total=total,
        page=page,
        pages=max(1, (total + PAGE - 1) // PAGE),
        persons=persons,
        mode=mode,
    )


@router.get("/faces/unclustered")
def unclustered(request: Request, page: int = 1, quality: str = "good") -> Response:
    where = (Face.cluster_id.is_(None)) & (Face.ignored.is_(False))
    if quality == "good":
        where = where & Face.clusterable.is_(True)
    return _face_list(request, _("Unclustered faces"), where, page, "unclustered")


@router.get("/faces/ignored")
def ignored(request: Request, page: int = 1) -> Response:
    return _face_list(request, _("Ignored faces"), Face.ignored.is_(True), page, "ignored")


@router.get("/api/clusters/search")
def cluster_search(request: Request, q: str = "") -> list[dict[str, object]]:
    c = ctx(request)
    with c.db.session() as s:
        stmt = (
            select(Cluster.id, Person.name, Cluster.face_count)
            .outerjoin(Person, Person.id == Cluster.person_id)
            .order_by(Cluster.face_count.desc())
            .limit(30)
        )
        if q:
            cond = [Person.name.ilike(f"%{like_escape(q)}%", escape="\\")]
            if q.isdigit():
                cond.append(Cluster.id == int(q))
            stmt = stmt.where(or_(*cond))
        return [{"id": i, "name": n, "faces": f} for i, n, f in s.execute(stmt)]


@router.get("/api/faces/{face_id}")
def face_info(request: Request, face_id: int) -> dict[str, object]:
    c = ctx(request)
    with c.db.session() as s:
        face = s.get(Face, face_id)
        if face is None:
            raise HTTPException(404)
        img = s.get(Image, face.image_id)
        return {
            "id": face.id,
            "image_id": face.image_id,
            "path": img.path if img else None,
            "cluster_id": face.cluster_id,
            "det_score": face.det_score,
            "age": face.age,
            "gender": face.gender,
        }
