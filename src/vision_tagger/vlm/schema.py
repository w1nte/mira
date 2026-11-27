"""Vocabulary-driven JSON schema and lenient result coercion for the VLM stage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

BASE_TYPES = {"enum", "enum_list", "list", "boolean", "integer", "text", "short_text"}
CONFIDENCE_SUFFIX = "_confidence"
# Free text of a model answer is stored as one line of at most this many characters. The model
# may copy text it sees in a photo (a sign, a screenshot); that text later reaches other
# prompts (search), so it must stay short and cannot fake lines of its own.
MAX_TEXT_CHARS = 400
MAX_SHORT_TEXT_CHARS = 120
MAX_ITEM_CHARS = 60


def default_vocabulary_path() -> Path:
    return Path(str(resources.files("vision_tagger.resources") / "vocabulary.yaml"))


def default_synonyms_path() -> Path:
    return Path(str(resources.files("vision_tagger.resources") / "synonyms.yaml"))


def slug(value: Any) -> str:
    """Normalise an enum-like answer: ``"Meeting Room"`` -> ``"meeting_room"``."""
    s = str(value).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s.strip("_")


@dataclass
class FieldSpec:
    name: str
    type: str
    description: str = ""
    values: list[str] = field(default_factory=list)
    max_items: int = 10
    min: int = 0
    max: int = 1000
    store: bool = True  # yaml key `property`: store as filterable property

    @property
    def inferred(self) -> bool:
        return self.type.startswith("inferred_")

    @property
    def base_type(self) -> str:
        return self.type.removeprefix("inferred_")

    def fallback(self) -> str | None:
        for candidate in ("unknown", "other", "none", "not_applicable"):
            if candidate in self.values:
                return candidate
        return None


class Vocabulary:
    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        self.version = str(data.get("version", 1))
        self.fields: dict[str, FieldSpec] = {}
        for name, spec in (data.get("fields") or {}).items():
            spec = dict(spec or {})
            ftype = spec.pop("type", "short_text")
            if ftype.removeprefix("inferred_") not in BASE_TYPES:
                raise ValueError(f"unknown field type {ftype!r} for {name!r}")
            values = [slug(v) for v in spec.pop("values", []) or []]
            if ftype.removeprefix("inferred_") in ("enum", "enum_list") and not values:
                raise ValueError(f"field {name!r} needs values")
            self.fields[name] = FieldSpec(
                name=name,
                type=ftype,
                values=values,
                description=str(spec.pop("description", "")).strip(),
                store=bool(spec.pop("property", True)),
                **{k: v for k, v in spec.items() if k in ("max_items", "min", "max")},
            )

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path | None = None) -> Vocabulary:
        path = path or default_vocabulary_path()
        with Path(path).open(encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh) or {})

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    # ------------------------------------------------------------------ JSON schema
    def json_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for f in self.fields.values():
            base = f.base_type
            if base == "enum":
                s: dict[str, Any] = {"type": "string", "enum": f.values}
            elif base == "enum_list":
                s = {
                    "type": "array",
                    "items": {"type": "string", "enum": f.values},
                    "maxItems": f.max_items,
                }
            elif base == "list":
                s = {"type": "array", "items": {"type": "string"}, "maxItems": f.max_items}
            elif base == "boolean":
                s = {"type": "boolean"}
            elif base == "integer":
                s = {"type": "integer", "minimum": f.min, "maximum": f.max}
            else:
                s = {"type": "string"}
            if f.description:
                s["description"] = f.description
            props[f.name] = s
            if f.inferred:
                props[f.name + CONFIDENCE_SUFFIX] = {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": f"Confidence (0-1) for {f.name}.",
                }
        return {"type": "object", "properties": props, "required": list(props)}

    def field_guide(self) -> str:
        lines = []
        for f in self.fields.values():
            marker = " (assume)" if f.inferred else ""
            line = f"- {f.name}{marker}: {f.description}"
            if f.values:
                line += f" Allowed: {', '.join(f.values)}."
            if f.base_type in ("list", "enum_list"):
                line += f" Max {f.max_items} items."
            if f.inferred:
                line += f" Also give {f.name}{CONFIDENCE_SUFFIX} between 0 and 1."
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------ coercion
    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Leniently coerce a model answer into the vocabulary.

        Unknown enum values fall back to ``unknown``/``other`` (or are dropped), lists are
        de-duplicated and capped, numbers clamped. Raises ``ValueError`` if the answer is
        unusable (not an object or fewer than half of the fields present).
        """
        if not isinstance(raw, dict):
            raise ValueError("VLM answer is not a JSON object")
        present = sum(1 for k in self.fields if k in raw)
        if present < max(1, len(self.fields) // 2):
            raise ValueError(f"VLM answer misses most fields ({present}/{len(self.fields)})")
        out: dict[str, Any] = {}
        for f in self.fields.values():
            value = raw.get(f.name)
            base = f.base_type
            if base == "enum":
                v = slug(value) if value not in (None, "") else ""
                out[f.name] = v if v in f.values else f.fallback()
            elif base == "enum_list":
                items = value if isinstance(value, list) else ([value] if value else [])
                vals = [slug(i) for i in items]
                out[f.name] = _dedup([v for v in vals if v in f.values])[: f.max_items]
            elif base == "list":
                items = value if isinstance(value, list) else ([value] if value else [])
                words = (_one_line(i, MAX_ITEM_CHARS) for i in items)
                out[f.name] = [w for w in words if w]
            elif base == "boolean":
                out[f.name] = _to_bool(value)
            elif base == "integer":
                try:
                    n = int(float(value))
                except (TypeError, ValueError):
                    n = None
                out[f.name] = None if n is None else max(f.min, min(f.max, n))
            else:
                limit = MAX_SHORT_TEXT_CHARS if base == "short_text" else MAX_TEXT_CHARS
                s = "" if value is None else _one_line(value, limit)
                out[f.name] = s if s.lower() not in ("none", "null", "n/a", "unknown") else ""
            if f.inferred:
                conf = raw.get(f.name + CONFIDENCE_SUFFIX)
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = 0.5
                out[f.name + CONFIDENCE_SUFFIX] = round(max(0.0, min(1.0, c)), 3)
        return out


def _one_line(value: Any, limit: int) -> str:
    return " ".join(str(value).split())[:limit].strip()


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return None
