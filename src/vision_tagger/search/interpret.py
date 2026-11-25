"""Turn a natural-language search request into a structured query.

The local LLM receives the real vocabulary (allowed values, the most frequent free-text
values, person names, the date range) and answers a fixed JSON schema. Everything it returns
is validated against the database, so it can only refer to persons and values that exist;
anything else becomes a free-text keyword (and still counts through the semantic index).

Without a usable LLM, :class:`KeywordInterpreter` does a simpler dictionary-based mapping.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import Image, Person, PropertyAlias
from ..query import ImageFilter, _parse_date, property_counts
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
    persons_mode: str = "all"
    exclude_persons: list[tuple[int, str]] = field(default_factory=list)
    must: list[PropTerm] = field(default_factory=list)
    should: list[PropTerm] = field(default_factory=list)
    exclude: list[PropTerm] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    semantic_query: str = ""
    keywords: list[str] = field(default_factory=list)
    mode: str = "llm"  # "llm" or "basic"
    notes: list[str] = field(default_factory=list)

    def hard_filter(self, relaxed: bool = False) -> ImageFilter:
        """The strict part: persons, required / excluded properties and the date range."""
        return ImageFilter(
            persons=[pid for pid, _ in self.persons],
            persons_mode=self.persons_mode,
            exclude_persons=[pid for pid, _ in self.exclude_persons],
            props=[] if relaxed else [(t.key, t.value) for t in self.must],
            exclude_props=[(t.key, t.value) for t in self.exclude],
            date_from=self.date_from,
            date_to=self.date_to,
        )

    def soft_terms(self, relaxed: bool = False) -> list[PropTerm]:
        return list(self.should) + (list(self.must) if relaxed else [])


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

    @property
    def keys(self) -> list[str]:
        return sorted(set(self.allowed) | set(self.examples))


def build_context(s: Session, vocab: Vocabulary, synonyms_path) -> SearchContext:
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
    counts = property_counts(s)
    existing = {k: {v for v, _ in vals} for k, vals in counts.items()}
    examples = {
        k: [v for v, _ in counts.get(k, [])[:FREE_TEXT_EXAMPLES]] for k in [*free_keys, "year"]
    }
    examples = {k: v for k, v in examples.items() if v}
    persons = [(p.id, p.name) for p in s.scalars(select(Person).order_by(Person.name))]
    lo, hi = s.execute(
        select(func.min(Image.recorded_at), func.max(Image.recorded_at)).where(
            Image.missing.is_(False)
        )
    ).one()
    aliases = {(a.key, a.from_value): a.to_value for a in s.scalars(select(PropertyAlias))}
    fingerprint = f"{vocab.fingerprint}:{len(persons)}:{sum(len(v) for v in existing.values())}"
    return SearchContext(
        allowed=allowed,
        examples=examples,
        existing=existing,
        persons=persons,
        year_range=(lo.year, hi.year) if lo and hi else None,
        normalizer=Normalizer.from_file(synonyms_path, aliases),
        fingerprint=fingerprint,
    )


# --------------------------------------------------------------------------- validation
def _find_person(ctx: SearchContext, name: str) -> tuple[int, str] | None:
    key = " ".join(str(name).split()).casefold()
    if not key:
        return None
    exact = [p for p in ctx.persons if p[1].casefold() == key]
    if exact:
        return exact[0]
    # first-name match if unambiguous ("Anna" -> "Anna Example")
    partial = [p for p in ctx.persons if p[1].casefold().split()[0] == key]
    return partial[0] if len(partial) == 1 else None


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


def validate(raw: dict, ctx: SearchContext, query: str) -> Interpretation:
    """Validate an LLM answer against the context. Unknown things become keywords."""
    it = Interpretation(query=query, mode="llm")
    keywords: list[str] = [str(k) for k in raw.get("keywords") or [] if str(k).strip()]

    def persons(names: Iterable[object]) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for name in names or []:
            p = _find_person(ctx, str(name))
            if p is None:
                it.notes.append(f"unknown person '{name}' searched as text")
                keywords.append(str(name))
            elif p not in out:
                out.append(p)
        return out

    it.persons = persons(raw.get("persons") or [])
    it.exclude_persons = [
        p for p in persons(raw.get("exclude_persons") or []) if p not in it.persons
    ]
    it.persons_mode = "any" if raw.get("persons_mode") == "any" else "all"

    def terms(items: Iterable[object], with_weight: bool) -> list[PropTerm]:
        out: list[PropTerm] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            key, value = str(item.get("key") or ""), str(item.get("value") or "")
            canon = canonical_term(ctx, key, value)
            if canon is None:
                if value.strip():
                    keywords.append(value.replace("_", " "))
                continue
            weight = 1.0
            if with_weight:
                try:
                    weight = max(0.05, min(1.0, float(item.get("weight", 1.0))))
                except (TypeError, ValueError):
                    weight = 1.0
            term = PropTerm(slug(key), canon, weight)
            if all((t.key, t.value) != (term.key, term.value) for t in out):
                out.append(term)
        return out

    it.must = terms(raw.get("must") or [], False)
    it.should = [
        t
        for t in terms(raw.get("should") or [], True)
        if (t.key, t.value) not in {(m.key, m.value) for m in it.must}
    ]
    it.exclude = terms(raw.get("exclude") or [], False)
    it.date_from = _valid_date(raw.get("date_from"), end=False)
    it.date_to = _valid_date(raw.get("date_to"), end=True)
    it.semantic_query = str(raw.get("semantic_query") or "").strip() or query
    it.keywords = _dedup_words(keywords)
    return it


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
  persons_mode "all" = all of them together, "any" = at least one of them.
- Dates as YYYY, YYYY-MM or YYYY-MM-DD; use "" when there is no date. Resolve relative
  dates ("last summer", "letzten Winter") using today's date. A season without a year is the
  season property, not a date.
- "Work photos" means work_image = yes; "private" or "no work" means exclude work_image = yes.
- semantic_query: one short English sentence describing the wanted picture, without names.
- keywords: important words of the request that fit no listed value (in their original form).

Keys and allowed values:
{allowed}

Free-text keys (frequent values; other values may exist):
{examples}

Persons: {persons}
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
        "must": {"type": "array", "items": TERM},
        "should": {"type": "array", "items": WEIGHTED_TERM},
        "exclude": {"type": "array", "items": TERM},
        "date_from": {"type": "string"},
        "date_to": {"type": "string"},
        "semantic_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "persons",
        "persons_mode",
        "exclude_persons",
        "must",
        "should",
        "exclude",
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
    return SYSTEM.format(
        today=today.isoformat(), years=years, allowed=allowed, examples=examples, persons=persons
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
    "cat": ("animals", "cat"),
    "katze": ("animals", "cat"),
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
}


class KeywordInterpreter:
    """Dictionary-based fallback when no LLM is available (English and German words)."""

    def interpret(self, query: str, ctx: SearchContext) -> Interpretation:
        it = Interpretation(query=query, mode="basic")
        lowered = query.casefold()
        semantic = query
        # persons: full names first, then unambiguous first names
        for pid, name in sorted(ctx.persons, key=lambda p: -len(p[1])):
            pattern = r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)"
            first = r"(?<!\w)" + re.escape(name.casefold().split()[0]) + r"(?!\w)"
            m = re.search(pattern, lowered) or (
                re.search(first, lowered) if _find_person(ctx, name.split()[0]) else None
            )
            if m and (pid, name) not in it.persons + it.exclude_persons:
                negated = _negated(lowered[: m.start()])
                (it.exclude_persons if negated else it.persons).append((pid, name))
                semantic = re.sub(re.escape(m.group(0)), "", semantic, flags=re.IGNORECASE)
                lowered = lowered[: m.start()] + " " * (m.end() - m.start()) + lowered[m.end() :]
        # dates
        years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", lowered)]
        months = [MONTHS[w] for w in re.findall(r"\w+", lowered) if w in MONTHS]
        if years:
            lo, hi = min(years), max(years)
            if months and lo == hi:
                it.date_from = it.date_to = f"{lo}-{months[0]:02d}"
            else:
                it.date_from, it.date_to = str(lo), str(hi)
        # property words
        tokens = re.findall(r"\w+", lowered)
        negate_next = False
        consumed: set[int] = set()
        for i, tok in enumerate(tokens):
            if tok in NEGATIONS:
                negate_next = True
                continue
            term = self._term(tok, tokens[i + 1] if i + 1 < len(tokens) else None, ctx)
            if term is not None:
                consumed.add(i)
                target = it.exclude if negate_next else it.should
                if all((t.key, t.value) != (term.key, term.value) for t in target):
                    target.append(term)
                negate_next = False
            elif tok not in STOPWORDS and not tok.isdigit() and tok not in MONTHS:
                negate_next = False
        it.keywords = _dedup_words(
            t
            for i, t in enumerate(tokens)
            if i not in consumed
            and t not in STOPWORDS
            and t not in NEGATIONS
            and not t.isdigit()
            and t not in MONTHS
            and len(t) > 2
        )
        it.semantic_query = " ".join(semantic.split()) or query
        return it

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


def _negated(prefix: str) -> bool:
    words = re.findall(r"\w+", prefix)[-2:]
    return any(w in NEGATIONS for w in words)


def llm_interpret(backend, model: str | None, query: str, ctx: SearchContext) -> Interpretation:
    raw = backend.chat_json(system_prompt(ctx), f"Search request: {query}", SCHEMA, model=model)
    return validate(raw, ctx, query)
