"""How the NSFW verdict of a photo is made and stored.

Every photo has at most one ``image_properties`` row with ``key = "nsfw"``:

==============  ==============================================================================
``value``       ``yes`` / ``no``: the verdict (what the covers and the filters use)
``source``      who made it: ``manual`` (the user), ``classifier`` or ``vlm`` (the vision model)
``confidence``  the classifier's last score (0..1); kept when the user or the vision model made
                the verdict, so it can be made again without running the model
==============  ==============================================================================

The classifier and the vision model are combined with "or": a photo either of them finds NSFW is
NSFW (the classifier is good at nudity and sexual content, the vision model also sees violence and
gore). The vision model's answer is read from its stored result (``vlm_results``), so each of the
two can look again without the other. A manual verdict always wins: the classifier and the vision
model then only update the score. Back to "automatic" (``set_manual(..., None)``) makes the
verdict from the stored score and answer again.

All functions work inside the caller's (short) write transaction.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import Animal, Face, Image, ImageProperty, VlmResult
from ..db.util import chunked
from . import KEY, NO, YES

MANUAL = "manual"
CLASSIFIER = "classifier"
VLM = "vlm"

Verdict = tuple[str, str]  # (value, source)


def decide(score: float | None, vlm: bool | None, threshold: float) -> Verdict | None:
    """The automatic verdict; ``None`` while neither the classifier nor the vision model looked."""
    if score is not None and score >= threshold:
        return YES, CLASSIFIER
    if vlm:
        return YES, VLM
    if score is not None:
        return NO, CLASSIFIER
    if vlm is not None:
        return NO, VLM
    return None


def _flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        return None
    return bool(value)


def vlm_flags(s: Session, image_ids: Iterable[int]) -> dict[int, bool | None]:
    """The vision model's ``nsfw`` answers (``None``: no answer, e.g. analysed before)."""
    out: dict[int, bool | None] = {}
    answer = func.json_extract(VlmResult.raw_json, f"$.{KEY}")
    for part in chunked(sorted(set(image_ids))):
        for image_id, value in s.execute(
            select(VlmResult.image_id, answer).where(VlmResult.image_id.in_(part))
        ):
            out[image_id] = _flag(value)
    return out


def rows(s: Session, image_ids: Iterable[int]) -> dict[int, ImageProperty]:
    """The ``nsfw`` row of each photo (a stray second row is removed)."""
    out: dict[int, ImageProperty] = {}
    for part in chunked(sorted(set(image_ids))):
        stmt = (
            select(ImageProperty)
            .where(ImageProperty.key == KEY, ImageProperty.image_id.in_(part))
            .order_by(ImageProperty.id)
        )
        for row in s.scalars(stmt):
            if row.image_id in out:
                s.delete(row)
            else:
                out[row.image_id] = row
    return out


def _write(
    s: Session, image_id: int, row: ImageProperty | None, verdict: Verdict | None, score
) -> None:
    if verdict is None:
        if row is not None:
            s.delete(row)
        return
    value, source = verdict
    if row is None:
        s.add(
            ImageProperty(image_id=image_id, key=KEY, value=value, source=source, confidence=score)
        )
    else:
        row.value, row.source, row.confidence = value, source, score


def record_score(s: Session, image_id: int, score: float, threshold: float) -> str:
    """Store the classifier's score of a photo; returns the verdict (``yes`` / ``no``)."""
    score = round(float(score), 4)
    row = rows(s, [image_id]).get(image_id)
    if row is not None and row.source == MANUAL:
        row.confidence = score
        return row.value
    verdict = decide(score, vlm_flags(s, [image_id]).get(image_id), threshold)
    assert verdict is not None
    _write(s, image_id, row, verdict, score)
    return verdict[0]


def record_vlm(s: Session, image_id: int, flag: Any, threshold: float) -> str | None:
    """Combine the vision model's answer with the stored score; returns the verdict."""
    row = rows(s, [image_id]).get(image_id)
    if row is not None and row.source == MANUAL:
        return row.value
    score = row.confidence if row is not None else None
    verdict = decide(score, _flag(flag), threshold)
    _write(s, image_id, row, verdict, score)
    return verdict[0] if verdict else None


def set_manual(s: Session, image_ids: Iterable[int], value: str | None, threshold: float) -> int:
    """Mark photos NSFW (``"yes"``), not NSFW (``"no"``) or automatic again (``None``).

    Returns how many photos changed. Unknown ids are ignored.
    """
    if value not in (YES, NO, None):
        raise ValueError(f"invalid NSFW value {value!r}")
    ids: list[int] = []
    for part in chunked(sorted(set(image_ids))):
        ids += s.scalars(select(Image.id).where(Image.id.in_(part)))
    current = rows(s, ids)
    changed = 0
    if value is None:
        manual = [i for i in ids if i in current and current[i].source == MANUAL]
        flags = vlm_flags(s, manual)
        for i in manual:
            row = current[i]
            _write(s, i, row, decide(row.confidence, flags.get(i), threshold), row.confidence)
            changed += 1
        return changed
    for i in ids:
        row = current.get(i)
        if row is not None and row.source == MANUAL and row.value == value:
            continue
        _write(s, i, row, (value, MANUAL), row.confidence if row is not None else None)
        changed += 1
    return changed


# --------------------------------------------------------------------------- lookups
def _nsfw_ids():
    return select(ImageProperty.image_id).where(
        ImageProperty.key == KEY, ImageProperty.value == YES
    )


def nsfw_images(s: Session) -> frozenset[int]:
    """Ids of the photos that are NSFW (one range scan of the property index)."""
    return frozenset(s.scalars(_nsfw_ids()))


def faces_of_nsfw_images(s: Session) -> dict[int, int]:
    """``{face id: image id}`` for the faces in NSFW photos (their crops get a cover too)."""
    stmt = select(Face.id, Face.image_id).where(Face.image_id.in_(_nsfw_ids()))
    return {face_id: image_id for face_id, image_id in s.execute(stmt)}


def animals_of_nsfw_images(s: Session) -> dict[int, int]:
    stmt = select(Animal.id, Animal.image_id).where(Animal.image_id.in_(_nsfw_ids()))
    return {animal_id: image_id for animal_id, image_id in s.execute(stmt)}
