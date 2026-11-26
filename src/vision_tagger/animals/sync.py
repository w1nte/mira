"""The ``pet`` and ``animal`` properties of every photo (source ``animal``).

Every photo with an animal gets ``animal = <species>`` (English, translated in the UI like other
values; the species of its pet, else the detector's) and, for each named pet in it,
``pet = <name>``. Stored in ``image_properties`` like the place properties, so the gallery
facets and filters, the properties page, the XMP keywords and the search work without extra code.
Renames and merges made on the properties page (``property_aliases``) are applied, so a value
merged or deleted there stays merged or deleted.

The species is part of the search document (``search.document``); pet names are not (like
person names, they are matched by the structured part of a query), so naming a pet does not make
the search index re-embed its photos.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from ..db.models import Animal, ImageProperty, Pet, PropertyAlias
from ..db.util import chunked
from ..pipeline.control import mark_pending

if TYPE_CHECKING:
    from ..db.session import Database

PROPERTY_SOURCE = "animal"
PET_KEYS = ("pet", "animal")

Props = set[tuple[str, str]]


def _aliases(s: Session) -> dict[tuple[str, str], str | None]:
    return {
        (a.key, a.from_value): a.to_value
        for a in s.scalars(select(PropertyAlias).where(PropertyAlias.key.in_(PET_KEYS)))
    }


def desired(s: Session, image_ids: Iterable[int] | None = None) -> dict[int, Props]:
    """``{image id: {(key, value), ...}}`` as the animals say (``None``: every photo)."""
    aliases = _aliases(s)
    stmt = (
        select(Animal.image_id, Animal.species, Pet.species, Pet.name)
        .outerjoin(Pet, Pet.id == Animal.pet_id)
        .where(Animal.ignored.is_(False))
    )
    parts = (
        [stmt]
        if image_ids is None
        else [stmt.where(Animal.image_id.in_(p)) for p in chunked(sorted(set(image_ids)))]
    )
    out: dict[int, Props] = defaultdict(set)
    for query in parts:
        for image_id, species, pet_species, pet_name in s.execute(query):
            for key, value in (("animal", pet_species or species), ("pet", pet_name)):
                if not value:
                    continue
                value = aliases.get((key, value), value)
                if value:
                    out[image_id].add((key, value))
    return out


def _existing(s: Session, image_ids: Iterable[int] | None = None) -> dict[int, Props]:
    stmt = select(ImageProperty.image_id, ImageProperty.key, ImageProperty.value).where(
        ImageProperty.key.in_(PET_KEYS)
    )
    parts = (
        [stmt]
        if image_ids is None
        else [stmt.where(ImageProperty.image_id.in_(p)) for p in chunked(sorted(set(image_ids)))]
    )
    out: dict[int, Props] = defaultdict(set)
    for query in parts:
        for image_id, key, value in s.execute(query):
            out[image_id].add((key, value))
    return out


def _species(props: Props) -> Props:
    return {p for p in props if p[0] == "animal"}


def _write(s: Session, image_ids: list[int], want: dict[int, Props], have: dict[int, Props]):
    s.execute(
        delete(ImageProperty).where(
            ImageProperty.image_id.in_(image_ids), ImageProperty.key.in_(PET_KEYS)
        )
    )
    rows = [
        {"image_id": i, "key": k, "value": v, "source": PROPERTY_SOURCE}
        for i in image_ids
        for k, v in sorted(want.get(i, set()))
    ]
    if rows:
        s.execute(insert(ImageProperty), rows)
    # the search document mentions the species, not the pet names
    resend = [i for i in image_ids if _species(want.get(i, set())) != _species(have.get(i, set()))]
    if resend:
        mark_pending(s, "embed", resend)


def sync_images(s: Session, image_ids: Iterable[int]) -> int:
    """Bring the properties of some photos up to date (inside the caller's transaction)."""
    ids = sorted(set(image_ids))
    if not ids:
        return 0
    want, have = desired(s, ids), _existing(s, ids)
    stale = [i for i in ids if want.get(i, set()) != have.get(i, set())]
    for part in chunked(stale):
        _write(s, part, want, have)
    return len(stale)


def sync_all(db: Database, chunk: int = 500) -> int:
    """Bring every photo's properties up to date: find the outdated photos without the write
    lock, then update them in short transactions (each re-reads its photos, so an edit made in
    the meantime is never overwritten with older data)."""
    with db.session() as s:
        want, have = desired(s), _existing(s)
    stale = sorted(i for i in set(want) | set(have) if want.get(i, set()) != have.get(i, set()))
    changed = 0
    for part in chunked(stale, chunk):
        with db.write() as s:
            changed += sync_images(s, part)
    return changed
