"""Turn a natural-language search request into a structured query.

The local LLM receives the real vocabulary (allowed values, the most frequent free-text
values, person and pet names, the date range) and answers a fixed JSON schema. Everything it
returns is validated against the database, so it can only refer to persons, pets and values that
exist; anything else becomes a free-text keyword (and still counts through the semantic index).
Persons and pets are strict filters ("Anna with Bello": photos with both).

Alternatives ("or") form groups: the alternatives of a group are OR-ed, groups are AND-ed and
exclusions always apply. "Anna or Ben at the beach" is (Anna or Ben) and beach; persons and pets
use ``persons_mode = "any"``, properties and places ``any_of``, dates ``dates``.

Without a usable LLM, :class:`KeywordInterpreter` does a simpler dictionary-based mapping.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import i18n
from ..animals import SPECIES
from ..db.models import Image, Person, Pet, PropertyAlias
from ..geo.places import PLACE_KEYS
from ..i18n import Lazy
from ..query import ImageFilter, _parse_date, property_counts, range_param
from ..vlm.normalize import Normalizer, singularize
from ..vlm.schema import Vocabulary, slug

log = logging.getLogger(__name__)

DERIVED_KEYS = {
    "work_image": ["yes", "maybe", "no"],
    "people": ["0", "1", "2", "3-5", "6-10", "11+"],
    "orientation": ["landscape", "portrait", "square", "panorama"],
    "capture_device": ["phone", "camera", "none"],
}
FREE_TEXT_EXAMPLES = 30
PROMPT_PLACES = 30  # most frequent place names in the prompt (all of them are validated)

DateRange = tuple[str | None, str | None]


# --------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class PropTerm:
    key: str
    value: str
    weight: float = 1.0


@dataclass
class Interpretation:
    query: str
    persons: list[tuple[int, str]] = field(default_factory=list)
    persons_mode: str = "all"  # persons and pets together
    exclude_persons: list[tuple[int, str]] = field(default_factory=list)
    pets: list[tuple[int, str]] = field(default_factory=list)
    exclude_pets: list[tuple[int, str]] = field(default_factory=list)
    must: list[PropTerm] = field(default_factory=list)
    should: list[PropTerm] = field(default_factory=list)
    exclude: list[PropTerm] = field(default_factory=list)
    # required alternatives: a photo has one property (or place) of every group
    any_of: list[list[PropTerm]] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    dates: list[DateRange] = field(default_factory=list)  # alternative ranges ("2019 or 2021")
    semantic_query: str = ""
    #: what the photos should show, for the image-text model: without names, places, dates and
    #: excluded things (it can't tell "no dogs" from "dogs"); "" = nothing to look for
    visual_query: str = ""
    keywords: list[str] = field(default_factory=list)
    mode: str = "llm"  # "llm", "basic" or "similar"
    # shown in the UI; interpretations are cached for every language, so notes are Lazy
    notes: list[str | Lazy] = field(default_factory=list)

    @property
    def who_any(self) -> bool:
        """Several persons / pets of which one is enough ("Anna or Ben")."""
        return self.persons_mode == "any" and len(self.persons) + len(self.pets) > 1

    def hard_filter(self, relaxed: bool = False) -> ImageFilter:
        """The strict part: persons, pets, required / excluded properties and the dates."""
        return ImageFilter(
            persons=[pid for pid, _ in self.persons],
            pets=[pid for pid, _ in self.pets],
            persons_mode=self.persons_mode,
            exclude_persons=[pid for pid, _ in self.exclude_persons],
            exclude_pets=[pid for pid, _ in self.exclude_pets],
            props=[] if relaxed else [(t.key, t.value) for t in self.must],
            prop_groups=[] if relaxed else [[(t.key, t.value) for t in g] for g in self.any_of],
            exclude_props=[(t.key, t.value) for t in self.exclude],
            date_from=self.date_from,
            date_to=self.date_to,
            dates=list(self.dates),
        )

    @property
    def required(self) -> bool:
        """Required properties that could be relaxed into preferences."""
        return bool(self.must or self.any_of)

    def soft_terms(self, relaxed: bool = False) -> list[PropTerm]:
        if not relaxed:
            return list(self.should)
        out = [*self.should, *self.must]
        for t in (t for g in self.any_of for t in g):
            if all((o.key, o.value) != (t.key, t.value) for o in out):
                out.append(t)
        return out


@dataclass
class SearchContext:
    """What the interpreter may refer to (built from the vocabulary and the database)."""

    allowed: dict[str, list[str]]  # closed vocabularies (enum, boolean, derived)
    examples: dict[str, list[str]]  # most frequent values of free-text keys
    existing: dict[str, set[str]]  # every value present in the DB, per key
    persons: list[tuple[int, str]]
    year_range: tuple[int, int] | None
    normalizer: Normalizer
    fingerprint: str = ""
    pets: list[tuple[int, str]] = field(default_factory=list)  # named pets
    pet_species: dict[int, str] = field(default_factory=dict)
    # (key, value) of place, city and country, most frequent first
    places: list[tuple[str, str]] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return sorted(set(self.allowed) | set(self.examples))


def build_context(
    s: Session,
    vocab: Vocabulary,
    synonyms_path,
    counts: dict[str, list[tuple[str, int]]] | None = None,
) -> SearchContext:
    """``counts``: ``property_counts(s)`` if the caller has it already (it is expensive)."""
    allowed: dict[str, list[str]] = {}
    free_keys: list[str] = []
    for f in vocab.fields.values():
        if not f.store:
            continue
        base = f.base_type
        if base in ("enum", "enum_list"):
            allowed[f.name] = list(f.values)
        elif base == "boolean":
            allowed[f.name] = ["yes", "no"]
        elif base in ("list", "short_text"):
            free_keys.append(f.name)
    allowed.update(DERIVED_KEYS)
    if counts is None:
        counts = property_counts(s)
    existing = {k: {v for v, _ in vals} for k, vals in counts.items()}
    if existing.get("animal"):  # the animal detector ran: its species are a closed vocabulary
        allowed["animal"] = list(SPECIES)
    examples = {
        k: [v for v, _ in counts.get(k, [])[:FREE_TEXT_EXAMPLES]] for k in [*free_keys, "year"]
    }
    examples = {k: v for k, v in examples.items() if v}
    persons = [(p.id, p.name) for p in s.scalars(select(Person).order_by(Person.name))]
    named_pets = list(s.scalars(select(Pet).where(Pet.name.is_not(None)).order_by(Pet.name)))
    pets = [(p.id, p.name or "") for p in named_pets]
    lo, hi = s.execute(
        select(func.min(Image.recorded_at), func.max(Image.recorded_at)).where(
            Image.missing.is_(False)
        )
    ).one()
    aliases = {(a.key, a.from_value): a.to_value for a in s.scalars(select(PropertyAlias))}
    places = [
        (k, v)
        for _n, k, v in sorted(
            ((n, k, v) for k in PLACE_KEYS for v, n in counts.get(k, [])),
            key=lambda t: (-t[0], t[1], t[2]),
        )
    ]
    fingerprint = (
        f"{vocab.fingerprint}:{len(persons)}:{len(pets)}:{sum(len(v) for v in existing.values())}"
    )
    return SearchContext(
        allowed=allowed,
        examples=examples,
        existing=existing,
        persons=persons,
        year_range=(lo.year, hi.year) if lo and hi else None,
        normalizer=Normalizer.from_file(synonyms_path, aliases),
        fingerprint=fingerprint,
        pets=pets,
        pet_species={p.id: p.species for p in named_pets},
        places=places,
    )


# --------------------------------------------------------------------------- validation
def _find_name(names: list[tuple[int, str]], name: str) -> tuple[int, str] | None:
    key = " ".join(str(name).split()).casefold()
    if not key:
        return None
    exact = [p for p in names if p[1].casefold() == key]
    if exact:
        return exact[0]
    # first-name match if unambiguous ("Anna" -> "Anna Example")
    partial = [p for p in names if p[1].casefold().split()[0] == key]
    return partial[0] if len(partial) == 1 else None


def _find_person(ctx: SearchContext, name: str) -> tuple[int, str] | None:
    return _find_name(ctx.persons, name)


def _find_pet(ctx: SearchContext, name: str) -> tuple[int, str] | None:
    return _find_name(ctx.pets, name)


def _fold(text: str) -> str:
    """Case- and accent-insensitive form of a name ("Köln" -> "koln")."""
    text = unicodedata.normalize("NFKD", " ".join(str(text).split()).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _find_place(ctx: SearchContext, name: str, prefer: str = "") -> PropTerm | None:
    """A place, city or country with this name (``prefer``: the key the LLM named)."""
    wanted = _fold(name)
    if not wanted:
        return None
    order = [k for k in (prefer, "city", "place", "country") if k in PLACE_KEYS]
    matches = [(k, v) for k, v in ctx.places if _fold(v) == wanted]
    if not matches:  # "Neustadt" for the only place "Neustadt (Coburg), Germany"
        short = {(k, v) for k, v in ctx.places if _fold(re.split(r"[,(]", v)[0]) == wanted}
        matches = list(short) if len({v for _, v in short}) == 1 else []
    if not matches:
        return None
    key, value = min(matches, key=lambda kv: order.index(kv[0]))
    return PropTerm(key, value)


def canonical_term(ctx: SearchContext, key: str, value: str) -> str | None:
    """Map a (key, value) proposal onto a value that exists; ``None`` if impossible."""
    key = slug(key)
    if key in ctx.allowed:
        v = ctx.normalizer.resolve_alias(key, slug(value))
        return v if v in ctx.allowed[key] else None
    if key in ctx.examples or key in ctx.existing:
        v = ctx.normalizer.canonical(key, str(value))
        if v and v in ctx.existing.get(key, set()):
            return v
    return None


def _valid_date(value: object, end: bool) -> str | None:
    if not value:
        return None
    s = str(value).strip()
    return s if _parse_date(s, end=end) is not None else None


def _date_alternative(value: object) -> DateRange | None:
    """``2019``, ``2019-07``, ``2019-07-14`` or ``FROM..TO`` (either end may be empty)."""
    first, sep, last = str(value).strip().partition("..")
    start = _valid_date(first, end=False)
    end = _valid_date(last if sep else first, end=True)
    if (first.strip() and start is None) or (sep and last.strip() and end is None):
        return None
    return (start, end) if start or end else None


def _label(kind: str, item: object) -> str:
    """An alternative as plain text (for notes)."""
    if kind in ("person", "pet"):
        return item[1]  # type: ignore[index]
    if kind == "date":
        start, end = item  # type: ignore[misc]
        return range_param(start, end)
    return item.value  # type: ignore[attr-defined]


def validate(raw: dict, ctx: SearchContext, query: str) -> Interpretation:
    """Validate an LLM answer against the context. Unknown things become keywords."""
    it = Interpretation(query=query, mode="llm")
    keywords: list[str] = [str(k) for k in raw.get("keywords") or [] if str(k).strip()]

    def who(names: Iterable[object], pets: Iterable[object]) -> tuple[list, list]:
        """Persons and pets; a name the LLM put into the wrong list still counts."""
        found_persons: list[tuple[int, str]] = []
        found_pets: list[tuple[int, str]] = []
        for name, pet_first in [(n, False) for n in names or []] + [(n, True) for n in pets or []]:
            person, pet = _find_person(ctx, str(name)), _find_pet(ctx, str(name))
            if pet is not None and (pet_first or person is None):
                if pet not in found_pets:
                    found_pets.append(pet)
            elif person is not None:
                if person not in found_persons:
                    found_persons.append(person)
            else:
                it.notes.append(Lazy("unknown person '%(name)s' searched as text", name=name))
                keywords.append(str(name))
        return found_persons, found_pets

    it.persons, it.pets = who(_items(raw.get("persons")), _items(raw.get("pets")))
    it.persons_mode = "any" if raw.get("persons_mode") == "any" else "all"

    def term(item: dict) -> PropTerm | None:
        """A property or place, validated; anything else becomes a keyword."""
        key, value = str(item.get("key") or ""), str(item.get("value") or "")
        canon = canonical_term(ctx, key, value) if slug(key) not in PLACE_KEYS else None
        if canon is not None:
            return PropTerm(slug(key), canon)
        place = _find_place(ctx, value, prefer=slug(key)) if value.strip() else None
        if place is None and value.strip():
            keywords.append(value.replace("_", " "))
        return place

    def terms(items: Iterable[object], with_weight: bool) -> list[PropTerm]:
        out: list[PropTerm] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            found = term(item)
            if found is None:
                continue
            weight = 1.0
            if with_weight:
                try:
                    weight = max(0.05, min(1.0, float(item.get("weight", 1.0))))
                except (TypeError, ValueError):
                    weight = 1.0
            found = PropTerm(found.key, found.value, weight)
            if all((t.key, t.value) != (found.key, found.value) for t in out):
                out.append(found)
        return out

    it.must = terms(_items(raw.get("must")), False)
    it.should = terms(_items(raw.get("should")), True)
    it.exclude = terms(_items(raw.get("exclude")), False)
    it.date_from = _valid_date(raw.get("date_from"), end=False)
    it.date_to = _valid_date(raw.get("date_to"), end=True)

    # ---- alternatives: groups of {key, value}; "person", "pet", "place" and "date" keys
    def alternative(item: object) -> tuple[str, object] | None:
        if not isinstance(item, dict):
            return None
        key, value = slug(item.get("key") or ""), str(item.get("value") or "").strip()
        if not value:
            return None
        if key in ("person", "persons", "pet", "pets"):
            is_pet = key.startswith("pet")
            found_persons, found_pets = who([] if is_pet else [value], [value] if is_pet else [])
            if found_pets:
                return "pet", found_pets[0]
            return ("person", found_persons[0]) if found_persons else None
        if key == "date":
            dates = _date_alternative(value)
            if dates is None:
                keywords.append(value)
            return ("date", dates) if dates else None
        found = term({"key": key, "value": value})
        return ("property", found) if found else None

    who_groups: list[list[tuple[str, tuple[int, str]]]] = []
    for group in _items(raw.get("any_of")):
        found_alternatives = _unique_alternatives(alternative(a) for a in _items(group))
        families = {"who" if k in ("person", "pet") else k for k, _ in found_alternatives}
        if len(found_alternatives) == 1:  # one alternative left: a plain condition
            kind, item = found_alternatives[0]
            if kind == "person" and item not in it.persons:
                it.persons.append(item)  # type: ignore[arg-type]
            elif kind == "pet" and item not in it.pets:
                it.pets.append(item)  # type: ignore[arg-type]
            elif kind == "property":
                it.must.append(item)  # type: ignore[arg-type]
            elif kind == "date" and not (it.date_from or it.date_to):
                it.date_from, it.date_to = item  # type: ignore[misc]
        elif families == {"who"}:
            who_groups.append(found_alternatives)  # type: ignore[arg-type]
        elif families == {"property"}:
            it.any_of.append([t for _, t in found_alternatives])  # type: ignore[misc]
        elif families == {"date"}:
            it.dates += [r for _, r in found_alternatives if r not in it.dates]  # type: ignore[misc]
        elif found_alternatives:  # "dogs or 2019": no strict meaning, only a preference
            it.should += [t for k, t in found_alternatives if k == "property"]  # type: ignore[misc]
            it.notes.append(
                Lazy(
                    "Alternatives of different kinds only rank the results: %(alternatives)s",
                    alternatives=" / ".join(_label(k, a) for k, a in found_alternatives),
                )
            )
    _merge_who(it, who_groups)
    if len(it.dates) == 1 and not (it.date_from or it.date_to):
        (it.date_from, it.date_to), it.dates = it.dates[0], []

    not_persons, not_pets = who(_items(raw.get("exclude_persons")), _items(raw.get("exclude_pets")))
    it.exclude_persons = [p for p in not_persons if p not in it.persons]
    it.exclude_pets = [p for p in not_pets if p not in it.pets]
    required = {(t.key, t.value) for t in it.must} | {
        (t.key, t.value) for g in it.any_of for t in g
    }
    it.should = _dedup_terms(t for t in it.should if (t.key, t.value) not in required)
    it.must = _dedup_terms(it.must)
    it.semantic_query = str(raw.get("semantic_query") or "").strip() or query
    it.visual_query = str(raw.get("semantic_query") or "").strip() or _drop_names(
        query, [name for _id, name in [*it.persons, *it.pets]]
    )
    it.keywords = _dedup_words(keywords)
    return it


def _drop_names(text: str, names: Iterable[str]) -> str:
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(_word(name.casefold()), _NAME, text, flags=re.IGNORECASE)
    return _trim(_without_names(text))


def _items(value: object) -> list:
    """A list from the LLM answer; anything else (a string, a number, null) is empty."""
    return list(value) if isinstance(value, list) else []


def _unique_alternatives(items: Iterable[tuple[str, object] | None]) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for item in items:
        if item is None:
            continue
        kind, value = item
        same = (lambda o: (o.key, o.value)) if kind == "property" else (lambda o: o)
        if all(k != kind or same(v) != same(value) for k, v in out):
            out.append(item)
    return out


def _dedup_terms(terms: Iterable[PropTerm]) -> list[PropTerm]:
    out: list[PropTerm] = []
    for t in terms:
        if all((o.key, o.value) != (t.key, t.value) for o in out):
            out.append(t)
    return out


def _merge_who(it: Interpretation, groups: list[list[tuple[str, tuple[int, str]]]]) -> None:
    """Person / pet alternatives use the "any" switch of the gallery filter.

    "Anna or Ben" is exact. With more (other persons required, or several groups, as in "Anna
    with Ben or Carl") the filter is the wider "any of them": photos with more of them rank
    first.
    """
    if not groups:
        return
    exact = len(groups) == 1 and (not (it.persons or it.pets) or it.persons_mode == "any")
    for kind, item in (a for g in groups for a in g):
        target = it.persons if kind == "person" else it.pets
        if item not in target:
            target.append(item)
    if not exact and it.persons_mode != "any":
        it.notes.append(Lazy("Persons and pets with “or”: photos with more of them come first"))
    it.persons_mode = "any"


def _dedup_words(words: Iterable[str]) -> list[str]:
    out: list[str] = []
    for w in words:
        w = " ".join(str(w).split())
        if len(w) >= 2 and w.casefold() not in {o.casefold() for o in out}:
            out.append(w)
    return out[:12]


# --------------------------------------------------------------------------- LLM prompt
SYSTEM = """You translate a search request for a private photo archive into a JSON query.
Today is {today}. The archive covers {years}.

Rules:
- Use ONLY the keys and values listed below. Values must match exactly.
- "must": conditions the user clearly requires. "should": helpful aspects, weight 0..1
  (1 = very important). "exclude": things the user does not want ("no", "without", "not").
- Prefer "should" over "must" unless the request is explicit; never invent constraints.
- persons: only names from the person list (any spelling the user used maps to that name).
  pets: only names from the pet list (animals the user named; never put a pet in persons).
  persons_mode "all" = all of them (persons and pets) together, "any" = at least one of them
  ("Anna or Ben").
- any_of: alternatives the user allows ("A or B", "either A or B", in any language: oder, ou,
  o, of, lub, или, 或, または). Each group lists {{key, value}} alternatives; a photo needs one of
  every group. An alternative is a listed property, or key "person", "pet", "place" (a city,
  country or place) or "date" (YYYY, YYYY-MM, YYYY-MM-DD or FROM..TO). "and", "with" and
  lists without "or" are not alternatives.
- An animal without a name ("dogs", "a cat") is a property (animal or animals), not a pet.
- Dates as YYYY, YYYY-MM or YYYY-MM-DD; use "" when there is no date. Resolve relative
  dates ("last summer", "letzten Winter") using today's date. A season without a year is the
  season property, not a date.
- "Work photos" means work_image = yes; "private" or "no work" means exclude work_image = yes.
- semantic_query: one short English sentence describing the wanted picture, without names.
- keywords: important words of the request that fit no listed value (in their original form).

Examples (names and values only show the format):
- "Anna or Ben at the beach": persons Anna, Ben; persons_mode "any"; should location_type=beach
- "beach or mountains in summer 2021": any_of [[location_type=beach, location_type=mountain]];
  should season=summer; date_from 2021, date_to 2021
- "Fotos von 2019 oder 2021": any_of [[date=2019, date=2021]]
- "chiens ou chats à Paris ou Rome": any_of [[animals=dog, animals=cat], [place=Paris,
  place=Rome]]

Keys and allowed values:
{allowed}

Free-text keys (frequent values; other values may exist):
{examples}

Persons: {persons}
Pets: {pets}
Places: {places}
"""

TERM = {
    "type": "object",
    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
    "required": ["key", "value"],
}
WEIGHTED_TERM = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "value": {"type": "string"},
        "weight": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["key", "value", "weight"],
}
SCHEMA = {
    "type": "object",
    "properties": {
        "persons": {"type": "array", "items": {"type": "string"}},
        "persons_mode": {"type": "string", "enum": ["all", "any"]},
        "exclude_persons": {"type": "array", "items": {"type": "string"}},
        "pets": {"type": "array", "items": {"type": "string"}},
        "exclude_pets": {"type": "array", "items": {"type": "string"}},
        "must": {"type": "array", "items": TERM},
        "should": {"type": "array", "items": WEIGHTED_TERM},
        "exclude": {"type": "array", "items": TERM},
        "any_of": {"type": "array", "items": {"type": "array", "items": TERM}},
        "date_from": {"type": "string"},
        "date_to": {"type": "string"},
        "semantic_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "persons",
        "persons_mode",
        "exclude_persons",
        "pets",
        "exclude_pets",
        "must",
        "should",
        "exclude",
        "any_of",
        "date_from",
        "date_to",
        "semantic_query",
        "keywords",
    ],
}


def system_prompt(ctx: SearchContext, today: date | None = None) -> str:
    today = today or date.today()
    years = f"{ctx.year_range[0]}-{ctx.year_range[1]}" if ctx.year_range else "unknown years"
    allowed = "\n".join(f"- {k}: {', '.join(v)}" for k, v in sorted(ctx.allowed.items()))
    examples = "\n".join(f"- {k}: {', '.join(v)}" for k, v in sorted(ctx.examples.items()))
    persons = ", ".join(name for _, name in ctx.persons) or "(none named yet)"
    pets = (
        ", ".join(f"{name} ({ctx.pet_species.get(pid, 'animal')})" for pid, name in ctx.pets)
        or "(none named yet)"
    )
    names: list[str] = []
    for _key, value in ctx.places:
        if _fold(value) not in {_fold(n) for n in names}:
            names.append(value)
        if len(names) >= PROMPT_PLACES:
            break
    return SYSTEM.format(
        today=today.isoformat(),
        years=years,
        allowed=allowed,
        examples=examples,
        persons=persons,
        pets=pets,
        places=", ".join(names) or "(none yet)",
    )


# --------------------------------------------------------------------------- keyword fallback
NEGATIONS = {
    "no",
    "not",
    "without",
    "except",
    "exclude",
    "excluding",
    "ohne",
    "kein",
    "keine",
    "keinen",
    "keiner",
    "nicht",
    "außer",
    "ausser",
}
#: "or" in the UI languages: between two matches of the same kind they are alternatives
OR_WORDS = {
    "or",
    "oder",
    "ou",
    "o",
    "of",
    "lub",
    "albo",
    "или",
    "либо",
    "或",
    "或者",
    "还是",
    "または",
    "か",
    "oppure",
}
#: "of" is English too ("photos of Anna"): next to an English article it is not "or"
ENGLISH_ARTICLES = {"the", "a", "an", "my", "our", "this", "that", "these", "those", "some"}
#: these never stand between alternatives
AND_WORDS = {"and", "und", "with", "mit", "et", "y", "e", "en", "i", "и", "和", "と", "avec", "con"}
#: "either A or B": not a keyword
EITHER_WORDS = {"either", "entweder", "soit"}
STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "and",
    "or",
    "with",
    "to",
    "from",
    "for",
    "by",
    "photo",
    "photos",
    "picture",
    "pictures",
    "image",
    "images",
    "show",
    "find",
    "me",
    "my",
    "some",
    "all",
    "during",
    "while",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "that",
    "this",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einen",
    "und",
    "oder",
    "mit",
    "im",
    "am",
    "beim",
    "vom",
    "von",
    "zum",
    "zur",
    "auf",
    "bei",
    "für",
    "fotos",
    "foto",
    "bilder",
    "bild",
    "zeige",
    "zeig",
    "mir",
    "meine",
    "mein",
    "alle",
    "aus",
    "wo",
    "als",
}
MONTHS = {
    "january": 1,
    "januar": 1,
    "jan": 1,
    "february": 2,
    "februar": 2,
    "feb": 2,
    "march": 3,
    "märz": 3,
    "maerz": 3,
    "april": 4,
    "may": 5,
    "mai": 5,
    "june": 6,
    "juni": 6,
    "july": 7,
    "juli": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "october": 10,
    "oktober": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dezember": 12,
    "dez": 12,
    "dec": 12,
}
# common words (English / German) -> (key, value)
WORD_TERMS: dict[str, tuple[str, str]] = {
    "work": ("work_image", "yes"),
    "arbeit": ("work_image", "yes"),
    "job": ("work_image", "yes"),
    "beach": ("location_type", "beach"),
    "strand": ("location_type", "beach"),
    "snow": ("weather", "snowy"),
    "schnee": ("weather", "snowy"),
    "rain": ("weather", "rainy"),
    "regen": ("weather", "rainy"),
    "sunny": ("weather", "sunny"),
    "sonne": ("weather", "sunny"),
    "christmas": ("event_type", "christmas"),
    "weihnachten": ("event_type", "christmas"),
    "birthday": ("event_type", "birthday"),
    "geburtstag": ("event_type", "birthday"),
    "wedding": ("event_type", "wedding"),
    "hochzeit": ("event_type", "wedding"),
    "party": ("event_type", "party"),
    "feier": ("event_type", "party"),
    "vacation": ("event_type", "vacation"),
    "holiday": ("event_type", "vacation"),
    "urlaub": ("event_type", "vacation"),
    "concert": ("event_type", "concert"),
    "konzert": ("event_type", "concert"),
    "summer": ("season", "summer"),
    "sommer": ("season", "summer"),
    "winter": ("season", "winter"),
    "spring": ("season", "spring"),
    "frühling": ("season", "spring"),
    "autumn": ("season", "autumn"),
    "fall": ("season", "autumn"),
    "herbst": ("season", "autumn"),
    "night": ("time_of_day", "night"),
    "nacht": ("time_of_day", "night"),
    "sunset": ("time_of_day", "golden_hour"),
    "sonnenuntergang": ("time_of_day", "golden_hour"),
    "mountain": ("location_type", "mountain"),
    "berg": ("location_type", "mountain"),
    "berge": ("location_type", "mountain"),
    "bergen": ("location_type", "mountain"),
    "sea": ("location_type", "sea"),
    "meer": ("location_type", "sea"),
    "lake": ("location_type", "lake_river"),
    "forest": ("location_type", "forest"),
    "wald": ("location_type", "forest"),
    "city": ("location_type", "city"),
    "stadt": ("location_type", "city"),
    "office": ("location_type", "office"),
    "büro": ("location_type", "office"),
    "restaurant": ("location_type", "restaurant"),
    "garden": ("location_type", "garden"),
    "garten": ("location_type", "garden"),
    "home": ("location_type", "home"),
    "zuhause": ("location_type", "home"),
    "dog": ("animals", "dog"),
    "hund": ("animals", "dog"),
    "hunde": ("animals", "dog"),
    "cat": ("animals", "cat"),
    "katze": ("animals", "cat"),
    "katzen": ("animals", "cat"),
    "horse": ("animals", "horse"),
    "pferd": ("animals", "horse"),
    "car": ("vehicles", "car"),
    "auto": ("vehicles", "car"),
    "selfie": ("composition", "selfie"),
    "portrait": ("composition", "portrait"),
    "porträt": ("composition", "portrait"),
    "group": ("composition", "group"),
    "gruppe": ("composition", "group"),
    "gruppenfoto": ("composition", "group"),
    "screenshot": ("image_type", "screenshot"),
    "document": ("image_type", "document"),
    "dokument": ("image_type", "document"),
    "receipt": ("image_type", "receipt"),
    "quittung": ("image_type", "receipt"),
    "beleg": ("image_type", "receipt"),
    "whiteboard": ("image_type", "whiteboard"),
    "meme": ("image_type", "meme"),
    "indoor": ("setting", "indoor"),
    "drinnen": ("setting", "indoor"),
    "outdoor": ("setting", "outdoor"),
    "draußen": ("setting", "outdoor"),
    "food": ("composition", "food"),
    "essen": ("composition", "food"),
    "nsfw": ("nsfw", "yes"),
}
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
_HAS_CJK = re.compile(f"[{_CJK}]")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)年?")
#: words that are no first names on their own
_FUNCTION_WORDS = OR_WORDS | AND_WORDS | STOPWORDS | NEGATIONS | EITHER_WORDS
#: translations that are no search words: "ja" and "是" are "yes", "中" is "medium"
_NOT_SEARCH_WORDS = {
    "yes",
    "no",
    "none",
    "unknown",
    "other",
    "ok",
    "maybe",
    "not applicable",
    "low",
    "medium",
    "high",
    "fast",
    "slow",
    "normal",
}
#: technical properties a translated word does not stand for
_TECHNICAL_KEYS = {
    "aesthetic",
    "sharpness",
    "exposure",
    "lighting",
    "text_present",
    "text_language",
    "people",
    "orientation",
    "capture_device",
}
#: place names are matched from this length on ("Rom", "Ulm")
MIN_PLACE_NAME = 3
_PLACE_ORDER = {"city": 0, "place": 1, "country": 2}


def _word(text: str) -> str:
    """``text`` as a whole word. CJK is written without spaces: CJK neighbours are fine."""
    return rf"(?<![^\W{_CJK}]){re.escape(text)}(?![^\W{_CJK}])"


@cache
def _translated_words(lang: str) -> dict[str, str]:
    """Single words of the UI language -> their English text (``"chien"`` -> ``"dog"``).

    Lets the keyword mode understand property words in the language of the UI; the vocabulary
    values are in the catalogs. English needs none (and "chat" stays a word in English).
    """
    out: dict[str, str] = {}
    for msgid, text in i18n.catalog(lang).items():
        english = msgid.split(i18n.CONTEXT_SEP, 1)[-1]
        if not isinstance(text, str) or len(english.split()) > 3 or "%" in english:
            continue
        word, english = text.casefold().strip(), english.casefold()
        if english in _NOT_SEARCH_WORDS or word in _FUNCTION_WORDS:
            continue
        if word and (_HAS_CJK.search(word) or re.fullmatch(r"\w+", word)):
            out.setdefault(word, english)
    return out


def _joins(gap: str) -> bool:
    """The text between two matches makes them alternatives: "or" (+ articles), "/"."""
    words = re.findall(r"\w+", gap)
    if not words:
        return gap.strip() == "/"
    ors = [w for w in words if w in OR_WORDS]
    if len(ors) != 1 or any(w in AND_WORDS or w in NEGATIONS for w in words):
        return False
    if ors[0] == "of" and any(w in ENGLISH_ARTICLES for w in words):
        return False
    # articles and prepositions may stand around it ("oder in den", "ou à la")
    return all(w in OR_WORDS or w in STOPWORDS or len(w) <= 3 for w in words)


def _listed(gap: str) -> bool:
    """A comma between two matches: a list, alternatives if an "or" ends it ("A, B or C")."""
    return not re.findall(r"\w+", gap) and "," in gap


@dataclass
class _Match:
    start: int
    end: int
    family: str  # "who", "place", "prop" or "date"
    item: Any  # (kind, (id, name)) / PropTerm / (year | None, month | None)
    negated: bool = False


class KeywordInterpreter:
    """Dictionary-based fallback when no LLM is available.

    English and German words, the vocabulary in the language of the UI, the names of persons,
    pets and places and dates. "A or B" (in any UI language) between two matches of the same
    kind makes them alternatives: persons and pets use the "any" switch, properties and places
    become a required group, dates alternative ranges. Single property words only rank.
    """

    def interpret(self, query: str, ctx: SearchContext) -> Interpretation:
        it = Interpretation(query=query, mode="basic")
        lowered = query.casefold()
        text = lowered  # matches are blanked out here, so nothing is matched twice
        semantic = query
        found: list[_Match] = []

        def take(start: int, end: int, family: str, item: Any, negated: bool | None = None):
            nonlocal text
            if negated is None:
                negated = _negated(lowered[:start])
            found.append(_Match(start, end, family, item, negated))
            text = text[:start] + " " * (end - start) + text[end:]

        # persons first, then pets (a name both have is the person's), then places
        for kind, names, find in (
            ("person", ctx.persons, _find_person),
            ("pet", ctx.pets, _find_pet),
        ):
            for pid, name in sorted(names, key=lambda p: -len(p[1])):
                parts = name.casefold().split()
                if not parts or (parts == [parts[0]] and parts[0] in OR_WORDS):
                    continue
                m = re.search(_word(name.casefold()), text)
                # the first name alone, if unambiguous ("Ou Yang": not the French "ou")
                first = parts[0]
                if m is None and first not in _FUNCTION_WORDS and find(ctx, first) is not None:
                    m = re.search(_word(first), text)
                if m:
                    take(m.start(), m.end(), "who", (kind, (pid, name)))
                    semantic = re.sub(re.escape(m.group(0)), _NAME, semantic, flags=re.IGNORECASE)
        for key, value in sorted(ctx.places, key=lambda kv: (-len(kv[1]), _PLACE_ORDER[kv[0]])):
            name = value.casefold()
            if len(name) >= MIN_PLACE_NAME and name in text:
                m = re.search(_word(name), text)
                if m:
                    take(m.start(), m.end(), "place", PropTerm(key, value))
        # dates: "2021", "July 2021", "2019 or 2021", "June or July 2021"
        years = list(_YEAR.finditer(text))
        months = [m for m in re.finditer(r"\w+", text) if m.group(0) in MONTHS]
        paired: set[int] = set()
        for y in years:
            before = [m for m in months if m.end() <= y.start()]
            month = before[-1] if before else None
            if month is not None and re.fullmatch(r"[\s,]*(of\s+)?", text[month.end() : y.start()]):
                paired.add(month.start())
                take(
                    month.start(), y.end(), "date", (int(y.group(1)), MONTHS[month.group(0)]), False
                )
            else:
                take(y.start(), y.end(), "date", (int(y.group(1)), None), False)
        for m in months:
            if m.start() not in paired:
                take(m.start(), m.end(), "date", (None, MONTHS[m.group(0)]), False)
        # property words: CJK has no spaces (substrings), other languages word by word
        words = _translated_words(i18n.current())
        cjk = sorted(
            [w for w in WORD_TERMS if _HAS_CJK.search(w)]
            + [w for w in words if _HAS_CJK.search(w)],
            key=len,
            reverse=True,
        )
        for word in cjk:
            m = re.search(re.escape(word), text)
            term = self._translated(words.get(word, word), ctx) if m else None
            if m and term is not None:
                take(m.start(), m.end(), "prop", term)
        tokens = list(re.finditer(r"\w+", text))
        negate_next = False
        consumed: set[int] = set()
        for i, m in enumerate(tokens):
            tok = m.group(0)
            if tok in NEGATIONS:
                negate_next = True
                continue
            if tok.isdigit():  # years are handled as the date range
                continue
            nxt = tokens[i + 1].group(0) if i + 1 < len(tokens) else None
            term = self._term(tok, nxt, ctx)
            if term is None and tok not in _FUNCTION_WORDS:
                english = words.get(tok) or words.get(singularize(tok))
                term = self._translated(english, ctx) if english else None
            if term is not None:
                consumed.add(i)
                found.append(_Match(m.start(), m.end(), "prop", term, negate_next))
                negate_next = False
            elif tok not in STOPWORDS and tok not in MONTHS and tok not in OR_WORDS:
                negate_next = False

        self._apply(it, lowered, found)
        it.keywords = _dedup_words(
            t
            for i, t in enumerate(m.group(0) for m in tokens)
            if i not in consumed
            and t not in STOPWORDS
            and t not in NEGATIONS
            and t not in OR_WORDS
            and t not in EITHER_WORDS
            and not t.isdigit()
            and t not in MONTHS
            and len(t) > 2
        )
        it.semantic_query = _without_names(semantic) or query
        it.visual_query = _visual_text(lowered, found, query)
        return it

    @staticmethod
    def _apply(it: Interpretation, lowered: str, found: list[_Match]) -> None:
        """Matches -> interpretation; neighbours joined by "or" are alternatives."""
        found.sort(key=lambda f: f.start)
        chains: list[tuple[list[_Match], bool]] = []  # (matches, joined by "or")
        for f in found:
            if chains and chains[-1][0][-1].family == f.family:
                last, joined = chains[-1]
                gap = lowered[last[-1].end : f.start]
                if _joins(gap) or _listed(gap):
                    last.append(f)
                    chains[-1] = (last, joined or _joins(gap))
                    continue
            chains.append(([f], False))

        who_groups: list[list[tuple[str, tuple[int, str]]]] = []
        date_items: list[list[tuple[int | None, int | None]]] = []
        any_dates = False
        for matches, joined in chains:
            alternatives = joined and len(matches) > 1
            family = matches[0].family
            if family == "date":
                date_items.append([f.item for f in matches])
                any_dates = any_dates or alternatives
                continue
            for f in matches:
                negated = matches[0].negated if alternatives else f.negated
                if family == "who":
                    kind, item = f.item
                    if negated:
                        target = it.exclude_persons if kind == "person" else it.exclude_pets
                    elif alternatives:
                        continue  # below, as a group
                    else:
                        target = it.persons if kind == "person" else it.pets
                    if item not in target:
                        target.append(item)
                elif negated:
                    it.exclude.append(f.item)
                elif not alternatives:
                    it.should.append(f.item)
            if alternatives and not matches[0].negated:
                if family == "who":
                    who_groups.append([f.item for f in matches])
                else:
                    it.any_of.append(_dedup_terms(f.item for f in matches))
        _merge_who(it, who_groups)
        for group in [g for g in it.any_of if len(g) < 2]:  # "beach or Strand": one value
            it.should += group
        it.any_of = [g for g in it.any_of if len(g) > 1]
        _apply_dates(it, date_items, any_dates)
        required = {(t.key, t.value) for g in it.any_of for t in g}
        it.should = _dedup_terms(t for t in it.should if (t.key, t.value) not in required)
        it.exclude = _dedup_terms(it.exclude)

    @classmethod
    def _translated(cls, english: str, ctx: SearchContext) -> PropTerm | None:
        """The property a word of the UI language stands for (via its English text)."""
        term = cls._term(english, None, ctx)
        return term if term is not None and term.key not in _TECHNICAL_KEYS else None

    @staticmethod
    def _term(tok: str, nxt: str | None, ctx: SearchContext) -> PropTerm | None:
        candidates: list[tuple[str, str]] = []
        if tok in WORD_TERMS:
            candidates.append(WORD_TERMS[tok])
        single = singularize(tok)
        if single in WORD_TERMS:
            candidates.append(WORD_TERMS[single])
        for phrase in ([f"{tok} {nxt}", f"{tok}_{nxt}"] if nxt else []) + [single, tok]:
            for key, values in ctx.allowed.items():
                if phrase.replace(" ", "_") in values and key not in ("people",):
                    candidates.append((key, phrase.replace(" ", "_")))
            for key, values in ctx.existing.items():
                if key in ctx.examples and phrase in values:
                    candidates.append((key, phrase))
        for key, value in candidates:
            canon = canonical_term(ctx, key, value)
            if canon is not None:
                return PropTerm(key, canon, 1.0)
        return None


def _apply_dates(
    it: Interpretation, chains: list[list[tuple[int | None, int | None]]], alternatives: bool
) -> None:
    """Years and months -> one date range, or alternative ranges ("2019 or 2021")."""
    items = [item for chain in chains for item in chain]
    years = [y for y, _ in items if y is not None]
    if not years:
        return
    if not alternatives:  # one range: from the first to the last year (or one month)
        months = [m for _, m in items if m is not None]
        lo, hi = min(years), max(years)
        if months and lo == hi:
            it.date_from = it.date_to = f"{lo}-{months[0]:02d}"
        else:
            it.date_from, it.date_to = str(lo), str(hi)
        return
    ranges: list[DateRange] = []
    for chain in chains:
        for i, (year, month) in enumerate(chain):
            # "June or July 2021": a month without a year takes the next year given
            later = [y for y, _ in chain[i + 1 :] if y is not None]
            year = year or (later[0] if later else None)
            if year is None:
                continue
            period = f"{year}-{month:02d}" if month else str(year)
            if (period, period) not in ranges:
                ranges.append((period, period))
    if len(ranges) == 1:
        it.date_from, it.date_to = ranges[0]
    else:
        it.dates = ranges


_NAME = "\x00"  # where a person or pet name was


def _without_names(text: str) -> str:
    """The request without the names (not in the documents) and the "or" between them."""
    words = text.replace(_NAME, f" {_NAME} ").split()
    joiners = (OR_WORDS - {"of"}) | EITHER_WORDS
    kept = [
        w
        for i, w in enumerate(words)
        if w != _NAME
        and w.strip(",;")
        and not (
            w.casefold().strip(",;") in joiners
            and _NAME in words[max(0, i - 1) : i] + words[i + 1 : i + 2]
        )
    ]
    return " ".join(kept)


def _trim(text: str) -> str:
    """Without function words at either end ("dogs in" -> "dogs"); "" if nothing else."""
    words = text.split()
    while words and words[0].casefold().strip(",;.") in _FUNCTION_WORDS:
        words.pop(0)
    while words and words[-1].casefold().strip(",;.") in _FUNCTION_WORDS:
        words.pop()
    return " ".join(words)


def _visual_text(lowered: str, found: list[_Match], query: str = "") -> str:
    """What the photos should show: the request without what the filters take care of
    (persons, pets, places, dates) and without the excluded things ("no dogs", "ohne Hund"):
    the image-text model would look for them instead. In the letters of ``query`` when it
    lines up with ``lowered`` (shown as "Looking for")."""
    source = query if len(query) == len(lowered) else lowered
    covered = [False] * len(lowered)
    spans = [(f.start, f.end) for f in found if f.family in ("who", "place", "date") or f.negated]
    for m in re.finditer(r"\w+", lowered):  # a negation and the word after it
        if m.group(0) in NEGATIONS:
            nxt = re.match(r"\W*\w+", lowered[m.end() :])
            spans.append((m.start(), m.end() + (nxt.end() if nxt else 0)))
    for start, end in spans:
        covered[start:end] = [True] * (end - start)
    out: list[str] = []
    for i, ch in enumerate(source):
        if not covered[i]:
            out.append(ch)
        elif i == 0 or not covered[i - 1]:
            out.append(f" {_NAME} ")
    return _trim(_without_names("".join(out)))


def _negated(prefix: str) -> bool:
    words = re.findall(r"\w+", prefix)[-2:]
    return any(w in NEGATIONS for w in words)


def llm_interpret(backend, model: str | None, query: str, ctx: SearchContext) -> Interpretation:
    raw = backend.chat_json(system_prompt(ctx), f"Search request: {query}", SCHEMA, model=model)
    return validate(raw, ctx, query)
