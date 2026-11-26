"""Animals and pets: overview of the groups, naming, merging, moving and removing detections."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import func, select

from ...animals import SPECIES
from ...animals import grouping as G
from ...db.models import Animal, Pet
from ...db.util import chunked, like_escape
from ...i18n import _, _n
from ..common import back, ctx, ints, pet_label, redirect, render

router = APIRouter()

PAGE = 120
SORTS = ("size", "unnamed", "newest")


def _pet_names(s) -> list[str]:
    return list(s.scalars(select(Pet.name).where(Pet.name.is_not(None)).order_by(Pet.name)))


def _photo_counts(s, pet_ids: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in chunked(pet_ids):
        out.update(
            s.execute(
                select(Animal.pet_id, func.count(func.distinct(Animal.image_id)))
                .where(Animal.pet_id.in_(part), Animal.ignored.is_(False))
                .group_by(Animal.pet_id)
            ).all()
        )
    return out


@router.get("/animals")
def overview(
    request: Request,
    show: str = "all",
    species: str = "",
    q: str = "",
    sort: str = "size",
    page: int = 1,
) -> Response:
    c = ctx(request)
    page = max(page, 1)
    sort = sort if sort in SORTS else "size"
    with c.db.session() as s:
        stmt = select(Pet)
        if q.strip():
            stmt = stmt.where(Pet.name.ilike(f"%{like_escape(q.strip())}%", escape="\\"))
        if show == "named":
            stmt = stmt.where(Pet.name.is_not(None))
        elif show == "unnamed":
            stmt = stmt.where(Pet.name.is_(None))
        if species in SPECIES:
            stmt = stmt.where(Pet.species == species)
        if sort == "unnamed":
            stmt = stmt.order_by(Pet.name.is_not(None), Pet.animal_count.desc(), Pet.id)
        elif sort == "newest":
            stmt = stmt.order_by(Pet.id.desc())
        else:
            stmt = stmt.order_by(Pet.animal_count.desc(), Pet.id)
        total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        pets = list(s.scalars(stmt.offset((page - 1) * PAGE).limit(PAGE)))
        photos = _photo_counts(s, [p.id for p in pets])
        ungrouped = (
            s.scalar(
                select(func.count())
                .select_from(Animal)
                .where(Animal.pet_id.is_(None), Animal.ignored.is_(False))
            )
            or 0
        )
        ignored = s.scalar(select(func.count()).select_from(Animal).where(Animal.ignored)) or 0
        present = list(s.scalars(select(Pet.species).distinct().order_by(Pet.species)))
        names = _pet_names(s)
    return render(
        request,
        "animals.html",
        pets=pets,
        photos=photos,
        total=total,
        page=page,
        pages=max(1, (total + PAGE - 1) // PAGE),
        show=show,
        species=species,
        species_present=present,
        q=q,
        sort=sort,
        ungrouped=ungrouped,
        ignored=ignored,
        pet_names=names,
    )


def _render_card(request: Request, pet_id: int) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        pet = s.get(Pet, pet_id)
        if pet is None:
            return Response("")
        photos = _photo_counts(s, [pet.id])
    return render(request, "partials/pet_card.html", p=pet, photos=photos)


@router.post("/pets/{pet_id}/name")
def name_pet(request: Request, pet_id: int, name: str = Form(""), card: str = Form("")) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        if s.get(Pet, pet_id) is None:
            raise HTTPException(404)
        pet = G.name_pet(s, pet_id, name)
        target = pet.id
    if target != pet_id:  # the name belonged to another pet: merged into it
        msg = _("Merged into %(name)s.", name=G.clean_name(name))
        if card and request.headers.get("HX-Request"):
            return redirect(request, back(request, "/animals"), msg=msg)
        return redirect(request, f"/pets/{target}", msg=msg)
    if card and request.headers.get("HX-Request"):
        return _render_card(request, pet_id)
    return redirect(request, back(request, f"/pets/{pet_id}"), msg=_("Name saved."))


@router.post("/pets/{pet_id}/species")
def set_species(request: Request, pet_id: int, species: str = Form("")) -> Response:
    c = ctx(request)
    url = back(request, f"/pets/{pet_id}")
    try:
        with c.db.write() as s:
            G.set_pet_species(s, pet_id, species)
    except LookupError:
        return redirect(request, "/animals", err=_("This group no longer exists."))
    except ValueError:
        return redirect(request, url, err=_("Unknown species."))
    return redirect(request, url, msg=_("Species saved."))


@router.post("/pets/merge")
def merge(
    request: Request, pet_ids: list[str] = Form(default_factory=list), target: str = Form("")
) -> Response:
    c = ctx(request)
    ids = ints(pet_ids)
    url = back(request, "/animals")
    if len(ids) < 2:
        return redirect(request, url, err=_("Select at least two groups."))
    with c.db.write() as s:
        pets = list(s.scalars(select(Pet).where(Pet.id.in_(ids))))
        if not pets:
            return redirect(request, url, err=_("This group no longer exists."))
        if target.isdigit() and int(target) in {p.id for p in pets}:
            target_id = int(target)
        else:  # prefer a named pet, then the largest
            target_id = max(pets, key=lambda p: (p.name is not None, p.animal_count, -p.id)).id
        G.merge_pets(s, target_id, [p.id for p in pets])
    return redirect(
        request, url, msg=_n("Merged %(num)d group.", "Merged %(num)d groups.", len(ids))
    )


@router.get("/pets/{pet_id}")
def pet_detail(request: Request, pet_id: int, page: int = 1) -> Response:
    c = ctx(request)
    page = max(page, 1)
    with c.db.session() as s:
        pet = s.get(Pet, pet_id)
        if pet is None:
            return redirect(request, "/animals", err=_("This group no longer exists."))
        animals = list(
            s.scalars(
                select(Animal)
                .where(Animal.pet_id == pet_id, Animal.ignored.is_(False))
                .order_by((Animal.score * Animal.box_px).desc(), Animal.id)
                .offset((page - 1) * PAGE)
                .limit(PAGE)
            )
        )
        sims = dict(G.similar_pets(s, pet_id, limit=8))
        similar = sorted(
            ((p, sims[p.id]) for p in s.scalars(select(Pet).where(Pet.id.in_(list(sims))))),
            key=lambda t: -t[1],
        )
        photos = _photo_counts(s, [pet_id, *sims])
        names = _pet_names(s)
    return render(
        request,
        "pet.html",
        pet=pet,
        animals=animals,
        n_photos=photos.get(pet_id, 0),
        photos=photos,
        similar=similar,
        page=page,
        pages=max(1, (pet.animal_count + PAGE - 1) // PAGE),
        species_list=SPECIES,
        pet_names=names,
        mode="pet",
    )


# --------------------------------------------------------------------------- detections
@router.post("/animals/action")
def animals_action(
    request: Request,
    action: str = Form(...),
    animal_ids: list[str] = Form(default_factory=list),
    name: str = Form(""),
) -> Response:
    c = ctx(request)
    ids = ints(animal_ids)
    url = back(request, "/animals")
    if not ids:
        return redirect(request, url, err=_("No animals selected."))
    n = len(ids)
    with c.db.write() as s:
        if action == "pet":
            if not G.clean_name(name):
                return redirect(request, url, err=_("Enter a name."))
            G.move_animals_to_name(s, ids, name)
            msg = _n(
                "%(num)d photo moved to %(name)s.",
                "%(num)d photos moved to %(name)s.",
                n,
                name=G.clean_name(name),
            )
            return redirect(request, url, msg=msg)
        if action == "new":
            pet_id = G.move_animals(s, ids, None)
            if G.clean_name(name):
                G.name_pet(s, pet_id, name)
            msg = _n(
                "%(num)d photo moved to a new group.", "%(num)d photos moved to a new group.", n
            )
            return redirect(request, url, msg=msg)
        if action == "unassign":
            G.unassign_animals(s, ids)
            msg = _n(
                "%(num)d photo taken out of the group.",
                "%(num)d photos taken out of the group.",
                n,
            )
            return redirect(request, url, msg=msg)
        if action == "ignore":
            G.ignore_animals(s, ids, True)
            msg = _n("%(num)d animal removed.", "%(num)d animals removed.", n)
            return redirect(request, url, msg=msg)
        if action == "unignore":
            G.ignore_animals(s, ids, False)
            msg = _n("%(num)d animal restored.", "%(num)d animals restored.", n)
            return redirect(request, url, msg=msg)
    return redirect(request, url, err=_("Unknown action %(action)s.", action=action))


def _animal_list(request: Request, title: str, where, page: int, mode: str) -> Response:
    c = ctx(request)
    page = max(page, 1)
    with c.db.session() as s:
        total = s.scalar(select(func.count()).select_from(Animal).where(where)) or 0
        animals = list(
            s.scalars(
                select(Animal)
                .where(where)
                .order_by((Animal.score * Animal.box_px).desc(), Animal.id)
                .offset((page - 1) * PAGE)
                .limit(PAGE)
            )
        )
        names = _pet_names(s)
    return render(
        request,
        "animal_list.html",
        title=title,
        animals=animals,
        total=total,
        page=page,
        pages=max(1, (total + PAGE - 1) // PAGE),
        pet_names=names,
        mode=mode,
    )


@router.get("/animals/ungrouped")
def ungrouped(request: Request, page: int = 1) -> Response:
    where = Animal.pet_id.is_(None) & Animal.ignored.is_(False)
    return _animal_list(request, _("Ungrouped animals"), where, page, "ungrouped")


@router.get("/animals/ignored")
def removed(request: Request, page: int = 1) -> Response:
    return _animal_list(request, _("Removed animals"), Animal.ignored.is_(True), page, "ignored")


@router.get("/api/pets")
def pets_api(request: Request, q: str = "") -> list[dict[str, object]]:
    """Pets for pickers: id, display name, species, number of detections."""
    c = ctx(request)
    with c.db.session() as s:
        stmt = select(Pet).order_by(Pet.animal_count.desc(), Pet.id).limit(30)
        if q.strip():
            stmt = stmt.where(Pet.name.ilike(f"%{like_escape(q.strip())}%", escape="\\"))
        return [
            {"id": p.id, "name": pet_label(p), "species": p.species, "animals": p.animal_count}
            for p in s.scalars(stmt)
        ]
