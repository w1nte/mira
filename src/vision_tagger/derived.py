"""Rule-based derived properties that combine VLM answers with file metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DerivedConfig
from .db.models import Image

WORKISH_IMAGE_TYPES = {"screenshot", "document", "whiteboard", "slide", "receipt", "scan"}
SCREEN_SIZES = {
    (1280, 720),
    (1366, 768),
    (1440, 900),
    (1536, 864),
    (1600, 900),
    (1680, 1050),
    (1920, 1080),
    (1920, 1200),
    (2560, 1080),
    (2560, 1440),
    (2560, 1600),
    (2880, 1800),
    (3440, 1440),
    (3840, 2160),
    (1170, 2532),
    (1179, 2556),
    (1080, 2400),
    (1080, 2340),
    (1080, 1920),
    (1284, 2778),
    (1290, 2796),
    (750, 1334),
    (828, 1792),
    (1242, 2688),
    (1440, 3200),
    (1440, 3120),
}
PHONE_MAKES = (
    "apple",
    "samsung",
    "google",
    "xiaomi",
    "huawei",
    "oneplus",
    "oppo",
    "vivo",
    "motorola",
    "nokia",
    "sony mobile",
    "honor",
    "realme",
    "fairphone",
    "lg electronics",
    "htc",
)


@dataclass
class Derived:
    key: str
    value: str
    confidence: float | None = None


def people_bucket(n: int | None) -> str | None:
    if n is None:
        return None
    if n <= 2:
        return str(n)
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def orientation(width: int | None, height: int | None) -> str | None:
    if not width or not height:
        return None
    ratio = width / height
    if 0.95 <= ratio <= 1.05:
        return "square"
    if ratio > 2.0:
        return "panorama"
    return "landscape" if ratio > 1 else "portrait"


def capture_device(meta: dict[str, Any]) -> str:
    make = str(meta.get("make") or "").lower()
    if not make and not meta.get("model"):
        return "none"
    if any(make.startswith(p) for p in PHONE_MAKES):
        return "phone"
    return "camera"


def looks_like_screen(img: Image) -> bool:
    if img.meta.get("make") or img.meta.get("model"):
        return False
    size = (img.width or 0, img.height or 0)
    return size in SCREEN_SIZES or (size[1], size[0]) in SCREEN_SIZES or img.ext == ".png"


def work_score(img: Image, vlm: dict[str, Any], cfg: DerivedConfig) -> float:
    """Combine VLM assumptions with metadata into a 0..1 'work image' score."""
    is_work = vlm.get("is_work_related")
    conf = float(vlm.get("is_work_related_confidence", 0.5) or 0.5)
    if is_work is True:
        p = conf
    elif is_work is False:
        p = 1.0 - conf
    else:
        p = 0.5
    score = 0.6 * p
    if vlm.get("image_type") in WORKISH_IMAGE_TYPES:
        score += 0.2
    wc = vlm.get("work_category")
    if wc and wc not in ("not_work", "other"):
        score += 0.1 * float(vlm.get("work_category_confidence", 0.5) or 0.5) * 2
    if looks_like_screen(img):
        score += 0.1
    haystack = f"{img.rel_path}".lower()
    if any(k.lower() in haystack for k in cfg.work_path_keywords):
        score += 0.15
    if vlm.get("is_personal_memory") is True:
        score -= 0.2 * float(vlm.get("is_personal_memory_confidence", 0.5) or 0.5)
    return round(max(0.0, min(1.0, score)), 3)


def metadata_properties(img: Image) -> list[Derived]:
    """Properties available right after indexing (stored with ``source="exif"``)."""
    out: list[Derived] = []
    o = orientation(img.width, img.height)
    if o:
        out.append(Derived("orientation", o))
    if img.recorded_at:
        out.append(Derived("year", str(img.recorded_at.year)))
    out.append(Derived("capture_device", capture_device(img.meta or {})))
    if (img.meta or {}).get("gps"):
        out.append(Derived("has_gps", "yes"))
    return out


def derive(img: Image, vlm: dict[str, Any], cfg: DerivedConfig) -> list[Derived]:
    """Properties that need the VLM answer (stored with ``source="derived"``)."""
    out: list[Derived] = []
    score = work_score(img, vlm, cfg)
    label = (
        "yes"
        if score >= cfg.work_yes_threshold
        else "maybe"
        if score >= cfg.work_maybe_threshold
        else "no"
    )
    out.append(Derived("work_image", label, score))
    bucket = people_bucket(vlm.get("people_count"))
    if bucket is not None:
        out.append(Derived("people", bucket))
    return out
