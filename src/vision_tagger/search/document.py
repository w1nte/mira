"""The text document that represents an image in the semantic search index.

It contains everything extracted about the image except person names: persons are matched
through the structured part of a query, so renaming a person never requires re-embedding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime

from ..db.models import Image, ImageProperty, VlmResult

DOC_VERSION = "1"

_SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

# property keys in the order they appear in the document (others follow alphabetically)
_ORDER = [
    "image_type",
    "composition",
    "setting",
    "location_type",
    "landmark",
    "event_type",
    "main_activity",
    "people",
    "objects",
    "animals",
    "vehicles",
    "food",
    "tags",
    "time_of_day",
    "season",
    "weather",
    "lighting",
    "dominant_colors",
    "text_language",
    "work_image",
    "work_category",
    "likely_purpose",
    "is_personal_memory",
    "is_work_related",
    "contains_sensitive_info",
    "aesthetic",
    "sharpness",
    "exposure",
    "orientation",
    "capture_device",
    "year",
]
_SKIP = {"has_gps", "text_present"}


def _label(key: str) -> str:
    return key.replace("_", " ")


def date_words(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return f"{dt:%B %Y}, {_SEASONS[dt.month]} {dt.year}"


def build_document(img: Image, props: Iterable[ImageProperty], vlm: VlmResult | None) -> str:
    grouped: dict[str, list[str]] = {}
    for p in props:
        if p.key in _SKIP:
            continue
        value = _label(p.value)
        if p.key in ("is_personal_memory", "is_work_related", "contains_sensitive_info"):
            value = f"{_label(p.key)} {value}"
        if value not in grouped.setdefault(p.key, []):
            grouped[p.key].append(value)
    lines: list[str] = []
    raw = (vlm.raw_json if vlm else None) or {}
    caption = (vlm.caption if vlm else None) or raw.get("caption") or ""
    if caption:
        lines.append(caption.strip())
    text = str(raw.get("text_excerpt") or "").strip()
    if text:
        lines.append(f"Visible text: {text}")
    keys = [k for k in _ORDER if k in grouped] + sorted(k for k in grouped if k not in _ORDER)
    for key in keys:
        lines.append(f"{_label(key).capitalize()}: {', '.join(grouped[key])}.")
    if img.folder:
        lines.append("Folder: " + " / ".join(part for part in img.folder.split("/") if part))
    when = date_words(img.recorded_at)
    if when:
        lines.append(f"Date: {when}.")
    if img.camera:
        lines.append(f"Camera: {img.camera}.")
    return "\n".join(lines)


def doc_hash(document: str) -> str:
    return hashlib.sha1(document.encode("utf-8")).hexdigest()
