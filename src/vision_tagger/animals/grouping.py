"""Pets: animal detections grouped by who they are, and the user's edits of those groups.

A *pet* is simply a group of detections of one animal; naming it ("Bello") makes it findable by
that name. Grouping works like face clustering (``faces.clustering``), with the same rules that
keep the user's work:

* pets with ``locked=True`` (named, merged, edited by the user) are never dissolved; a named pet
  stays even when it has no photos left (re-detected photos can join it again),
* detections with ``assignment="manual"`` are never moved automatically,
* ignored detections never take part in grouping,
* re-detecting the animals of a photo (``replace_animals``) carries pet, assignment and ignore
  state over to the new detections.

Grouping: free detections first join an existing pet whose average embedding is at least
``animals.assign_threshold`` similar (named and locked pets only in a full run); the rest are
grouped with DBSCAN (cosine distance, ``animals.cluster_eps``). Both happen per species, where
the species the detector mixes up on pets (``animals.species_pools``: cat, dog, bear, sheep)
count as one; the pet's species is the one most of its detections show. DBSCAN runs without
holding the write lock (see :func:`regroup_db`).

After every change the ``pet`` / ``animal`` properties of the affected photos are updated
(``animals.sync``).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ..db.models import Animal, Assignment, Pet
from ..db.util import chunked
from ..faces.clustering import _dbscan, _iou, _synced, best_matches
from . import SPECIES
from .sync import sync_all, sync_images

if TYPE_CHECKING:
    from ..config import AnimalsConfig
    from ..db.session import Database

log = logging.getLogger(__name__)

MAX_NAME = 200


# --------------------------------------------------------------------------- vectors
def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32) if b else np.zeros(0, dtype=np.float32)


def stack(blobs: Sequence[bytes]) -> np.ndarray:
    """Rows of L2-normalised vectors (an empty 0 x 0 matrix for no blobs)."""
    if not blobs:
        return np.zeros((0, 0), dtype=np.float32)
    m = np.vstack([from_blob(b) for b in blobs]).astype(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def centroid(m: np.ndarray) -> np.ndarray:
    c = m.mean(axis=0)
    n = float(np.linalg.norm(c))
    return c / n if n > 0 else c


def pool_of(species: str, pools: Iterable[Iterable[str]]) -> str:
    """The grouping pool of a species: species that are often confused share one."""
    for pool in pools:
        members = list(pool)
        if species in members:
            return "+".join(sorted(members))
    return species


def majority_species(species: Iterable[str]) -> str:
    counts = Counter(species)
    if not counts:
        return "dog"
    return min(counts, key=lambda sp: (-counts[sp], sp))


# --------------------------------------------------------------------------- bookkeeping
@dataclass
class GroupStats:
    joined: int = 0
    new_pets: int = 0
    grouped: int = 0
    noise: int = 0
    removed: int = 0

    def __str__(self) -> str:
        return (
            f"{self.joined} animals joined existing pets, {self.new_pets} new groups "
            f"({self.grouped} animals), {self.noise} not grouped"
        )


@dataclass
class GroupPlan:
    """Result of the (lock-free) computation phase of :func:`regroup_db`."""

    to_existing: dict[int, list[int]] = field(default_factory=dict)  # pet id -> animal ids
    new_groups: list[list[int]] = field(default_factory=list)  # animal ids per new pet
    noise: int = 0


def _by_id(rows: Iterable[Any]) -> list[Any]:
    return sorted(rows, key=lambda r: r[0])


def _update_animals(s: Session, animal_ids: Sequence[int], **values: object) -> None:
    for part in chunked([int(i) for i in animal_ids]):
        s.execute(update(Animal).where(Animal.id.in_(part)).values(**values))


def _pets_of(s: Session, animal_ids: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for part in chunked(list(animal_ids)):
        out.update(
            p
            for p in s.scalars(select(Animal.pet_id).where(Animal.id.in_(part)).distinct())
            if p is not None
        )
    return out


def _images_of(s: Session, *, animal_ids: Sequence[int] = (), pet_ids: Iterable[int] = ()):
    out: set[int] = set()
    for part in chunked(list(animal_ids)):
        out.update(s.scalars(select(Animal.image_id).where(Animal.id.in_(part))))
    for part in chunked([p for p in pet_ids if p is not None]):
        out.update(s.scalars(select(Animal.image_id).where(Animal.pet_id.in_(part))))
    return out


def update_pet_stats(s: Session, pet_ids: Iterable[int | None] | None = None) -> int:
    """Recompute count, average, picture and species; delete unnamed pets that became empty.

    A named pet without detections is kept (with its last average), so its animal can find
    its way back when photos are detected again.
    """
    if pet_ids is None:
        pets = list(s.scalars(select(Pet)))
    else:
        ids = sorted({p for p in pet_ids if p is not None})
        pets = []
        for part in chunked(ids):
            pets.extend(s.scalars(select(Pet).where(Pet.id.in_(part))))
    removed = 0
    for pet in pets:
        rows = _by_id(
            s.execute(
                select(
                    Animal.id, Animal.embedding, Animal.score, Animal.box_px, Animal.species
                ).where(Animal.pet_id == pet.id, Animal.ignored.is_(False))
            )
        )
        if not rows:
            if pet.name:
                pet.animal_count = 0
                pet.cover_animal_id = None
            else:
                s.delete(pet)
                removed += 1
            continue
        pet.animal_count = len(rows)
        pet.centroid = to_blob(centroid(stack([r[1] for r in rows])))
        pet.cover_animal_id = max(rows, key=lambda r: (r[2] or 0) * min(r[3] or 1, 400))[0]
        pet.species = majority_species(r[4] for r in rows)
    s.flush()
    return removed


def _lock(s: Session, pet_ids: Iterable[int | None]) -> None:
    ids = [p for p in pet_ids if p is not None]
    for part in chunked(ids):
        s.execute(update(Pet).where(Pet.id.in_(part)).values(locked=True))


# --------------------------------------------------------------------------- grouping
def _free_stmt():
    return select(Animal.id, Animal.species, Animal.embedding).where(
        Animal.pet_id.is_(None),
        Animal.ignored.is_(False),
        Animal.groupable.is_(True),
        Animal.assignment != Assignment.MANUAL,
    )


def _dissolve_unlocked(s: Session) -> tuple[int, set[int]]:
    """Free the detections of automatic (unnamed, unlocked) pets; returns (removed, images)."""
    loose = list(s.scalars(select(Pet.id).where(Pet.locked.is_(False), Pet.name.is_(None))))
    if not loose:
        return 0, set()
    images = _images_of(s, pet_ids=loose)
    for part in chunked(loose):
        s.execute(
            update(Animal)
            .where(Animal.pet_id.in_(part), Animal.assignment != Assignment.MANUAL)
            .values(pet_id=None, assignment=Assignment.AUTO)
        )
    still_used: set[int] = set()
    for part in chunked(loose):
        still_used.update(
            s.scalars(select(Animal.pet_id).where(Animal.pet_id.in_(part)).distinct())
        )
    gone = [p for p in loose if p not in still_used]
    for part in chunked(gone):
        s.execute(delete(Pet).where(Pet.id.in_(part)))
    s.flush()
    return len(gone), images


def _load_inputs(s: Session, full: bool):
    free = _by_id(s.execute(_free_stmt()))
    ids = np.array([r[0] for r in free], dtype=np.int64)
    species = [r[1] for r in free]
    m = stack([r[2] for r in free])
    stmt = select(Pet.id, Pet.species, Pet.centroid).where(Pet.centroid.is_not(None))
    if full:
        stmt = stmt.where((Pet.locked.is_(True)) | (Pet.name.is_not(None)))
    pets = _by_id(s.execute(stmt))
    if m.size:  # a pet average from another model version cannot be compared
        pets = [p for p in pets if len(from_blob(p[2])) == m.shape[1]]
    pet_ids = [p[0] for p in pets]
    pet_species = [p[1] for p in pets]
    pmat = stack([p[2] for p in pets])
    return ids, species, m, pet_ids, pet_species, pmat


def plan_groups(
    ids: np.ndarray,
    species: Sequence[str],
    m: np.ndarray,
    pet_ids: Sequence[int],
    pet_species: Sequence[str],
    pmat: np.ndarray,
    cfg: AnimalsConfig,
) -> GroupPlan:
    """Pure computation: join existing pets, DBSCAN the rest, per species pool."""
    plan = GroupPlan()
    pools = np.array([pool_of(sp, cfg.species_pools) for sp in species])
    pet_pools = np.array([pool_of(sp, cfg.species_pools) for sp in pet_species])
    joined: dict[int, list[int]] = defaultdict(list)
    for pool in sorted(set(pools.tolist())):
        idx = np.flatnonzero(pools == pool)
        remaining = np.ones(len(idx), dtype=bool)
        candidates = np.flatnonzero(pet_pools == pool) if len(pet_pools) else np.zeros(0, int)
        if len(candidates):
            best, sim = best_matches(m[idx], pmat[candidates])
            hits = sim >= cfg.assign_threshold
            for k in np.flatnonzero(hits):
                joined[pet_ids[int(candidates[int(best[k])])]].append(int(ids[idx[k]]))
            remaining &= ~hits
        rest = idx[remaining]
        labels = _dbscan(m[rest], cfg.cluster_eps, cfg.cluster_min_samples)
        for label in sorted(set(labels.tolist()) - {-1}):
            plan.new_groups.append([int(i) for i in ids[rest[labels == label]]])
        plan.noise += int((labels == -1).sum())
    plan.to_existing = dict(joined)
    return plan


def apply_plan(s: Session, plan: GroupPlan, stats: GroupStats) -> set[int]:
    """Write a plan; only detections that are *still* free are touched (user edits win).

    Returns the images whose detections changed.
    """
    all_ids = [a for ids in plan.to_existing.values() for a in ids] + [
        a for g in plan.new_groups for a in g
    ]
    free_now: dict[int, str] = {}
    for part in chunked(all_ids):
        free_now.update(
            s.execute(
                select(Animal.id, Animal.species).where(
                    Animal.id.in_(part),
                    Animal.pet_id.is_(None),
                    Animal.ignored.is_(False),
                    Animal.assignment != Assignment.MANUAL,
                )
            ).all()
        )
    existing: set[int] = set()
    for part in chunked(list(plan.to_existing)):
        existing.update(s.scalars(select(Pet.id).where(Pet.id.in_(part))))
    touched: set[int] = set()
    changed: list[int] = []
    for pet_id, ids in plan.to_existing.items():
        ids = [i for i in ids if i in free_now]
        if not ids or pet_id not in existing:
            continue
        _update_animals(s, ids, pet_id=pet_id, assignment=Assignment.AUTO)
        touched.add(pet_id)
        changed += ids
        stats.joined += len(ids)
    for group in plan.new_groups:
        ids = [i for i in group if i in free_now]
        if not ids:
            continue
        pet = Pet(locked=False, species=majority_species(free_now[i] for i in ids))
        s.add(pet)
        s.flush()
        _update_animals(s, ids, pet_id=pet.id, assignment=Assignment.AUTO)
        touched.add(pet.id)
        changed += ids
        stats.new_pets += 1
        stats.grouped += len(ids)
    stats.noise = plan.noise
    update_pet_stats(s, touched)
    return _images_of(s, animal_ids=changed)


@_synced
def regroup(s: Session, cfg: AnimalsConfig, *, full: bool = False) -> GroupStats:
    """Group all free detections inside one session (simple API for tests and small jobs).

    ``full=True`` dissolves the automatic (unnamed, unlocked) pets first.
    """
    stats = GroupStats()
    images: set[int] = set()
    if full:
        stats.removed, images = _dissolve_unlocked(s)
    ids, species, m, pet_ids, pet_species, pmat = _load_inputs(s, full)
    if len(ids):
        plan = plan_groups(ids, species, m, pet_ids, pet_species, pmat, cfg)
        images |= apply_plan(s, plan, stats)
    sync_images(s, images)
    return stats


def regroup_db(db: Database, cfg: AnimalsConfig, *, full: bool = False) -> GroupStats:
    """Like :func:`regroup`, with short write transactions.

    1. dissolve automatic pets (short write), 2. read the free detections, 3. DBSCAN without any
    transaction, 4. apply the result (short write, guarded against concurrent edits),
    5. update the properties of every photo whose animals changed (chunked writes).
    """
    stats = GroupStats()
    if full:
        with db.write() as s:
            stats.removed, _images = _dissolve_unlocked(s)
    with db.session() as s:
        ids, species, m, pet_ids, pet_species, pmat = _load_inputs(s, full)
    if len(ids):
        plan = plan_groups(ids, species, m, pet_ids, pet_species, pmat, cfg)
        with db.write() as s:
            apply_plan(s, plan, stats)
    sync_all(db)
    return stats


# --------------------------------------------------------------------------- re-detection
@_synced
def replace_animals(
    s: Session, image_id: int, new: list[Animal], iou_threshold: float = 0.5
) -> list[int]:
    """Replace the animals of a photo with fresh detections, keeping the user's work.

    Old and new detections are matched by box overlap; matched new ones inherit pet, assignment
    and ignore state. New rows are inserted before the old ones are deleted, so no pet becomes
    empty in between. Updates the photo's properties; returns the removed ids (crop cleanup).
    """
    old = list(s.scalars(select(Animal).where(Animal.image_id == image_id).order_by(Animal.id)))
    pairs = sorted(
        (
            (_iou((o.x1, o.y1, o.x2, o.y2), (n.x1, n.y1, n.x2, n.y2)), oi, ni)
            for oi, o in enumerate(old)
            for ni, n in enumerate(new)
        ),
        reverse=True,
    )
    used_old: set[int] = set()
    matched: set[int] = set()
    for score, oi, ni in pairs:
        if score < iou_threshold:
            break
        if oi in used_old or ni in matched:
            continue
        used_old.add(oi)
        matched.add(ni)
        o, n = old[oi], new[ni]
        n.pet_id, n.assignment, n.ignored = o.pet_id, o.assignment, o.ignored
    for n in new:
        n.image_id = image_id
        s.add(n)
    s.flush()
    affected = {o.pet_id for o in old} | {n.pet_id for n in new}
    old_ids = [o.id for o in old]
    for part in chunked(old_ids):
        s.execute(delete(Animal).where(Animal.id.in_(part)))
    s.flush()
    update_pet_stats(s, affected)
    sync_images(s, [image_id])
    return old_ids


@_synced
def delete_image_animals(s: Session, image_id: int, crops_dir: Path | None = None) -> list[int]:
    """Delete all animals of a photo (its content changed), keeping pet stats consistent."""
    rows = s.execute(select(Animal.id, Animal.pet_id).where(Animal.image_id == image_id)).all()
    ids = [r[0] for r in rows]
    if not ids:
        return []
    for part in chunked(ids):
        s.execute(delete(Animal).where(Animal.id.in_(part)))
    s.flush()
    update_pet_stats(s, [r[1] for r in rows])
    if crops_dir is not None:
        from ..imaging import remove_face_crops

        remove_face_crops(crops_dir, ids)
    return ids


# --------------------------------------------------------------------------- the user's edits
def clean_name(name: str | None) -> str:
    return " ".join((name or "").split())[:MAX_NAME]


def find_pet(s: Session, name: str, exclude_id: int | None = None) -> Pet | None:
    """Case-insensitive lookup that also works for non-ASCII names (compared in Python)."""
    key = clean_name(name).casefold()
    if not key:
        return None
    for pet in s.scalars(select(Pet).where(Pet.name.is_not(None))):
        if pet.id != exclude_id and (pet.name or "").casefold() == key:
            return pet
    return None


@_synced
def name_pet(s: Session, pet_id: int, name: str | None) -> Pet:
    """Name a pet (kept from now on). A name another pet has merges this one into it; an empty
    name removes the name (the group stays, as edited by the user)."""
    pet = s.get(Pet, pet_id)
    if pet is None:
        raise LookupError(f"pet {pet_id} not found")
    name = clean_name(name)
    if name:
        other = find_pet(s, name, exclude_id=pet_id)
        if other is not None:
            return merge_pets(s, other.id, [pet_id])
    pet.name = name or None
    pet.locked = True
    s.flush()
    sync_images(s, _images_of(s, pet_ids=[pet_id]))
    if pet.animal_count == 0 and not pet.name:
        s.delete(pet)
        s.flush()
    return pet


@_synced
def merge_pets(s: Session, target_id: int, source_ids: Iterable[int]) -> Pet:
    """Move every detection of the sources into the target; the target keeps its name (or takes
    the first name of a source)."""
    target = s.get(Pet, target_id)
    if target is None:
        raise LookupError(f"pet {target_id} not found")
    sources = [p for p in source_ids if p != target_id]
    for pid in sources:
        src = s.get(Pet, pid)
        if src is None:
            continue
        if not target.name and src.name:
            target.name = src.name
        s.execute(update(Animal).where(Animal.pet_id == pid).values(pet_id=target_id))
        s.delete(src)
    target.locked = True
    s.flush()
    update_pet_stats(s, [target_id])
    sync_images(s, _images_of(s, pet_ids=[target_id]))
    return target


@_synced
def move_animals(s: Session, animal_ids: Iterable[int], target_pet_id: int | None) -> int:
    """Move detections to a pet (``None``: a new pet). They count as the user's choice."""
    ids = [int(i) for i in animal_ids]
    if not ids:
        raise ValueError("no animals given")
    old = _pets_of(s, ids)
    images = _images_of(s, animal_ids=ids)
    if target_pet_id is None:
        species = []
        for part in chunked(ids):
            species += list(s.scalars(select(Animal.species).where(Animal.id.in_(part))))
        pet = Pet(locked=True, species=majority_species(species))
        s.add(pet)
        s.flush()
        target_pet_id = pet.id
    elif s.get(Pet, target_pet_id) is None:
        raise LookupError(f"pet {target_pet_id} not found")
    _update_animals(s, ids, pet_id=target_pet_id, assignment=Assignment.MANUAL, ignored=False)
    _lock(s, [*old, target_pet_id])
    s.flush()
    update_pet_stats(s, [*old, target_pet_id])
    sync_images(s, images)
    return target_pet_id


@_synced
def move_animals_to_name(s: Session, animal_ids: Iterable[int], name: str) -> int:
    """Move detections to the pet with this name (a new pet if there is none)."""
    name = clean_name(name)
    if not name:
        raise ValueError("a pet needs a name")
    pet = find_pet(s, name)
    pet_id = move_animals(s, animal_ids, pet.id if pet else None)
    if pet is None:
        created = s.get(Pet, pet_id)
        assert created is not None
        created.name = name
        s.flush()
        sync_images(s, _images_of(s, pet_ids=[pet_id]))
    return pet_id


@_synced
def unassign_animals(s: Session, animal_ids: Iterable[int]) -> None:
    """Take detections out of their pet ("that's not Bello"); grouping won't put them back."""
    ids = [int(i) for i in animal_ids]
    if not ids:
        return
    old = _pets_of(s, ids)
    _update_animals(s, ids, pet_id=None, assignment=Assignment.MANUAL)
    _lock(s, old)
    s.flush()
    update_pet_stats(s, old)
    sync_images(s, _images_of(s, animal_ids=ids))


@_synced
def ignore_animals(s: Session, animal_ids: Iterable[int], ignored: bool = True) -> None:
    """Remove detections (not an animal, not relevant) or restore them."""
    ids = [int(i) for i in animal_ids]
    if not ids:
        return
    old = _pets_of(s, ids)
    if ignored:
        _update_animals(s, ids, ignored=True, pet_id=None, assignment=Assignment.MANUAL)
    else:
        _update_animals(s, ids, ignored=False, assignment=Assignment.AUTO)
    _lock(s, old)
    s.flush()
    update_pet_stats(s, old)
    sync_images(s, _images_of(s, animal_ids=ids))


@_synced
def set_pet_species(s: Session, pet_id: int, species: str) -> Pet:
    """Correct the species of a pet (and of its detections: the detector took a dog for a
    bear)."""
    if species not in SPECIES:
        raise ValueError(f"unknown species {species!r}")
    pet = s.get(Pet, pet_id)
    if pet is None:
        raise LookupError(f"pet {pet_id} not found")
    s.execute(update(Animal).where(Animal.pet_id == pet_id).values(species=species))
    pet.species = species
    pet.locked = True
    s.flush()
    sync_images(s, _images_of(s, pet_ids=[pet_id]))
    return pet


def similar_pets(s: Session, pet_id: int, limit: int = 8) -> list[tuple[int, float]]:
    """Pets whose average is closest to this pet's (merge candidates), most similar first."""
    pet = s.get(Pet, pet_id)
    if pet is None or pet.centroid is None:
        return []
    me = from_blob(pet.centroid)
    rows = [
        r
        for r in s.execute(select(Pet.id, Pet.centroid).where(Pet.centroid.is_not(None)))
        if r[0] != pet_id and len(from_blob(r[1])) == len(me)
    ]
    if not rows:
        return []
    sims = stack([r[1] for r in rows]) @ me
    order = np.argsort(-sims)
    return [(rows[i][0], float(sims[i])) for i in order[:limit]]
