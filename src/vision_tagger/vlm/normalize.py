"""Canonicalisation of free-text property values."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

_IRREGULAR = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "geese": "goose",
    "knives": "knife",
    "leaves": "leaf",
    "shelves": "shelf",
    "loaves": "loaf",
    "wolves": "wolf",
    "calves": "calf",
}
_KEEP = {
    "glasses",
    "sunglasses",
    "jeans",
    "pants",
    "shorts",
    "trousers",
    "scissors",
    "clothes",
    "headphones",
    "earphones",
    "binoculars",
    "stairs",
    "news",
    "series",
    "species",
    "chess",
    "dress",
    "grass",
    "glass",
    "bus",
    "gas",
    "lens",
    "cactus",
    "canvas",
    "christmas",
    "mattress",
    "fries",
    "sweets",
    "goggles",
    "tennis",
    "cross",
    "boss",
    "compass",
    "hummus",
    "couscous",
    "asparagus",
    "octopus",
    "walrus",
    "virus",
    "status",
    "campus",
    "circus",
    "bonus",
    "atlas",
}


def singularize_word(word: str) -> str:
    w = word.lower()
    if len(w) <= 3 or w in _KEEP:
        return w
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if w.endswith("oes") and w not in ("shoes", "toes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")):
        return w[:-1]
    return w


def singularize(phrase: str) -> str:
    """Singularise the head noun (last word) of a phrase."""
    parts = phrase.split(" ")
    if not parts:
        return phrase
    parts[-1] = singularize_word(parts[-1])
    return " ".join(parts)


def clean_text(value: str) -> str:
    s = str(value).strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"^(a|an|the|some|several|many|two|three|\d+)\s+", "", s)
    s = re.sub(r"[^\w\s&'+-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip(" -'")
    return s


class Normalizer:
    """lowercase -> strip articles/punctuation -> singularise -> synonyms -> user aliases."""

    def __init__(
        self,
        synonyms: dict[str, dict[str, list[str]]] | None = None,
        aliases: dict[tuple[str, str], str | None] | None = None,
    ) -> None:
        self._syn: dict[str, dict[str, str]] = {}
        for section, mapping in (synonyms or {}).items():
            table: dict[str, str] = {}
            for canonical, alias_list in (mapping or {}).items():
                canon = singularize(clean_text(canonical))
                table[canon] = canon
                for alias in alias_list or []:
                    table[singularize(clean_text(alias))] = canon
                    table[clean_text(alias)] = canon
            self._syn[section] = table
        self.aliases = dict(aliases or {})

    @classmethod
    def from_file(
        cls, path: Path | None, aliases: dict[tuple[str, str], str | None] | None = None
    ) -> Normalizer:
        data: dict[str, dict[str, list[str]]] = {}
        if path is not None and Path(path).exists():
            with Path(path).open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        return cls(data, aliases)

    def canonical(self, key: str, value: str, *, free_text: bool = True) -> str | None:
        """Return the canonical value, or ``None`` if the value should be dropped."""
        if free_text:
            v = clean_text(value)
            if not v or v in ("none", "nothing", "n a", "unknown"):
                return None
            v = singularize(v)
            v = self._syn.get(key, {}).get(v) or self._syn.get("global", {}).get(v) or v
        else:
            v = str(value)
        return self.resolve_alias(key, v)

    def resolve_alias(self, key: str, value: str) -> str | None:
        seen = set()
        while (key, value) in self.aliases and value not in seen:
            seen.add(value)
            target = self.aliases[(key, value)]
            if target is None:
                return None
            value = target
        return value

    def canonical_list(self, key: str, values: Iterable[str], limit: int) -> list[str]:
        out: list[str] = []
        for v in values:
            c = self.canonical(key, v)
            if c and c not in out:
                out.append(c)
            if len(out) >= limit:
                break
        return out
