"""Deep search: let the LLM re-rank the best candidates and explain each choice."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Cluster, Face, Image, ImageProperty, Person, VlmResult
from ..db.util import chunked
from .document import date_words

log = logging.getLogger(__name__)

SYSTEM = """You judge photos from a private archive against a search request.
You get one compact description per photo (date, people, place, content). Rate how well each
photo matches the request from 0 (not at all) to 10 (perfect) and give a short reason of at most
12 words, written in the language of the request. Judge only from the descriptions.
Answer with one entry per photo id."""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number", "minimum": 0, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
            },
        }
    },
    "required": ["results"],
}

_KEYS = [
    "location_type",
    "event_type",
    "main_activity",
    "composition",
    "objects",
    "animals",
    "tags",
    "season",
    "time_of_day",
    "weather",
    "work_image",
    "people",
    "landmark",
]


@dataclass
class Judgement:
    score: float
    reason: str


def describe(s: Session, image_ids: list[int]) -> dict[int, str]:
    """Short, LLM-friendly description per image (includes person names)."""
    out: dict[int, str] = {}
    for part in chunked(image_ids):
        images = {i.id: i for i in s.scalars(select(Image).where(Image.id.in_(part)))}
        captions = {
            image_id: caption
            for image_id, caption in s.execute(
                select(VlmResult.image_id, VlmResult.caption).where(VlmResult.image_id.in_(part))
            )
        }
        props: dict[int, dict[str, list[str]]] = {}
        for image_id, key, value in s.execute(
            select(ImageProperty.image_id, ImageProperty.key, ImageProperty.value).where(
                ImageProperty.image_id.in_(part), ImageProperty.key.in_(_KEYS)
            )
        ):
            props.setdefault(image_id, {}).setdefault(key, []).append(value.replace("_", " "))
        people: dict[int, set[str]] = {}
        for image_id, name in s.execute(
            select(Face.image_id, Person.name)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .where(Face.image_id.in_(part), Face.ignored.is_(False))
        ):
            people.setdefault(image_id, set()).add(name)
        for image_id in part:
            img = images.get(image_id)
            if img is None:
                continue
            bits = [f"date: {date_words(img.recorded_at) or 'unknown'}"]
            if people.get(image_id):
                bits.append("people: " + ", ".join(sorted(people[image_id])))
            if img.folder:
                bits.append(f"folder: {img.folder}")
            if captions.get(image_id):
                bits.append(f"caption: {captions[image_id]}")
            for key, values in props.get(image_id, {}).items():
                bits.append(f"{key.replace('_', ' ')}: {', '.join(values[:6])}")
            out[image_id] = "; ".join(bits)
    return out


def rerank(
    backend, model: str | None, query: str, descriptions: dict[int, str], batch: int = 20
) -> dict[int, Judgement]:
    """Ask the LLM to score the candidates; ids it invents are ignored."""
    ids = list(descriptions)
    judged: dict[int, Judgement] = {}
    for start in range(0, len(ids), batch):
        part = ids[start : start + batch]
        listing = "\n".join(f"[{i}] {descriptions[i]}" for i in part)
        raw = backend.chat_json(
            SYSTEM, f"Search request: {query}\n\nPhotos:\n{listing}", SCHEMA, model=model
        )
        for item in raw.get("results") or []:
            try:
                image_id = int(item.get("id"))
                score = max(0.0, min(10.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                continue
            if image_id in part:
                judged[image_id] = Judgement(score, str(item.get("reason") or "").strip()[:160])
    return judged
