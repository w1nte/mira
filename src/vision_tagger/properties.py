"""Property value management: merge, rename and delete values (persisted as aliases)."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .db.models import ImageProperty, PropertyAlias
from .pipeline.control import mark_pending


def _set_alias(s: Session, key: str, from_value: str, to_value: str | None) -> None:
    alias = s.scalar(
        select(PropertyAlias).where(
            PropertyAlias.key == key, PropertyAlias.from_value == from_value
        )
    )
    if alias is None:
        s.add(PropertyAlias(key=key, from_value=from_value, to_value=to_value))
    else:
        alias.to_value = to_value
    # re-point aliases that pointed at the old value (keeps chains short)
    s.execute(
        update(PropertyAlias)
        .where(PropertyAlias.key == key, PropertyAlias.to_value == from_value)
        .values(to_value=to_value)
    )
    if to_value is not None:
        # an alias target must not itself be aliased away
        s.execute(
            delete(PropertyAlias).where(
                PropertyAlias.key == key, PropertyAlias.from_value == to_value
            )
        )


def merge_values(s: Session, key: str, values: Iterable[str], target: str) -> int:
    """Merge ``values`` of ``key`` into ``target``. Returns the number of changed rows."""
    target = target.strip()
    if not target:
        raise ValueError("target value must not be empty")
    changed = 0
    affected: set[int] = set()
    for value in {v for v in values if v and v != target}:
        _set_alias(s, key, value, target)
        rows = s.scalars(
            select(ImageProperty).where(ImageProperty.key == key, ImageProperty.value == value)
        ).all()
        have_target = set(
            s.scalars(
                select(ImageProperty.image_id).where(
                    ImageProperty.key == key, ImageProperty.value == target
                )
            )
        )
        for row in rows:
            affected.add(row.image_id)
            if row.image_id in have_target:
                s.delete(row)
            else:
                row.value = target
                have_target.add(row.image_id)
            changed += 1
    s.flush()
    mark_pending(s, "embed", affected)  # search documents changed
    return changed


def rename_value(s: Session, key: str, value: str, new_value: str) -> int:
    return merge_values(s, key, [value], new_value)


def delete_value(s: Session, key: str, value: str) -> int:
    """Remove a value everywhere and drop it from future VLM results."""
    _set_alias(s, key, value, None)
    affected = list(
        s.scalars(
            select(ImageProperty.image_id).where(
                ImageProperty.key == key, ImageProperty.value == value
            )
        )
    )
    res = s.execute(
        delete(ImageProperty).where(ImageProperty.key == key, ImageProperty.value == value)
    )
    mark_pending(s, "embed", affected)
    return res.rowcount or 0


def remove_alias(s: Session, alias_id: int) -> None:
    """Forget an alias (future VLM results keep the original value again)."""
    s.execute(delete(PropertyAlias).where(PropertyAlias.id == alias_id))


def list_aliases(s: Session, key: str | None = None) -> list[PropertyAlias]:
    stmt = select(PropertyAlias).order_by(PropertyAlias.key, PropertyAlias.from_value)
    if key:
        stmt = stmt.where(PropertyAlias.key == key)
    return list(s.scalars(stmt))
