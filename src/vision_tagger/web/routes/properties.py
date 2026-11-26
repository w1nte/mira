"""Property explorer: all keys/values with counts, sample images, merge/rename/delete."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from sqlalchemy import select

from ...db.models import ImageProperty
from ...i18n import N_, _, _n
from ...properties import delete_value, list_aliases, merge_values, remove_alias
from ...query import ImageFilter
from ...vlm.schema import Vocabulary
from ..common import back, cached_property_counts, ctx, redirect, render

router = APIRouter()

# Display order of property groups on the overview page (titles translated when shown).
GROUPS: list[tuple[str, list[str]]] = [
    (
        N_("Assumptions"),
        [
            "work_image",
            "is_work_related",
            "work_category",
            "is_personal_memory",
            "likely_purpose",
            "contains_sensitive_info",
        ],
    ),
    (
        N_("Scene"),
        [
            "image_type",
            "setting",
            "location_type",
            "time_of_day",
            "season",
            "weather",
            "lighting",
            "landmark",
        ],
    ),
    (N_("People"), ["people", "composition", "main_activity", "event_type"]),
    (
        N_("Content"),
        [
            "objects",
            "animals",
            "vehicles",
            "food",
            "dominant_colors",
            "text_present",
            "text_language",
            "tags",
        ],
    ),
    (N_("Quality"), ["sharpness", "exposure", "aesthetic"]),
    (N_("File"), ["year", "orientation", "capture_device", "has_gps"]),
]


def _grouped(counts: dict[str, list[tuple[str, int]]]) -> list[tuple[str, list[str]]]:
    seen: set[str] = set()
    out: list[tuple[str, list[str]]] = []
    for title, keys in GROUPS:
        present = [k for k in keys if k in counts]
        seen.update(present)
        if present:
            out.append((_(title), present))
    rest = sorted(k for k in counts if k not in seen)
    if rest:
        out.append((_("Other"), rest))
    return out


@router.get("/properties")
def overview(request: Request, limit: int = 24) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    with c.db.session() as s:
        counts = cached_property_counts(c, s, f)
    return render(
        request,
        "properties.html",
        counts=counts,
        groups=_grouped(counts),
        f=f,
        limit=limit,
    )


@router.get("/properties/{key}")
def key_detail(request: Request, key: str, samples: int = 8) -> Response:
    c = ctx(request)
    f = ImageFilter.from_params(request.query_params)
    with c.db.session() as s:
        values = cached_property_counts(c, s, f, key=key).get(key, [])
        sample_ids: dict[str, list[int]] = {}
        for value, _n in values[:200]:
            stmt = (
                select(ImageProperty.image_id)
                .where(ImageProperty.key == key, ImageProperty.value == value)
                .order_by(
                    ImageProperty.confidence.desc().nulls_last(), ImageProperty.image_id.desc()
                )
                .limit(samples)
            )
            sample_ids[value] = list(s.scalars(stmt))
        aliases = list_aliases(s, key)
    try:
        spec = Vocabulary.load(c.settings.vlm.vocabulary_file).fields.get(key)
    except Exception:  # pragma: no cover - broken vocabulary file
        spec = None
    return render(
        request,
        "property_key.html",
        key=key,
        values=values,
        samples=sample_ids,
        aliases=aliases,
        spec=spec,
        f=f,
    )


@router.post("/properties/{key}/merge")
def merge(
    request: Request,
    key: str,
    values: list[str] = Form(default_factory=list),
    target: str = Form(""),
) -> Response:
    c = ctx(request)
    url = back(request, f"/properties/{key}")
    target = target.strip() or (values[0] if values else "")
    if not values or not target:
        return redirect(request, url, err=_("Select values and a target value."))
    if len(values) < 2 and target in values:
        return redirect(request, url, err=_("Select at least two values or enter a new name."))
    with c.db.write() as s:
        n = merge_values(s, key, values, target)
    return redirect(
        request,
        url,
        msg=_n(
            "Merged into '%(target)s' (%(num)d entry updated).",
            "Merged into '%(target)s' (%(num)d entries updated).",
            n,
            target=target,
        ),
    )


@router.post("/properties/{key}/delete")
def delete(request: Request, key: str, values: list[str] = Form(default_factory=list)) -> Response:
    c = ctx(request)
    url = back(request, f"/properties/{key}")
    if not values:
        return redirect(request, url, err=_("Select values to delete."))
    with c.db.write() as s:
        n = sum(delete_value(s, key, v) for v in values)
    return redirect(
        request,
        url,
        msg=_(
            "Deleted %(values)s (%(entries)s).",
            values=_n("%(num)d value", "%(num)d values", len(values)),
            entries=_n("%(num)d entry", "%(num)d entries", n),
        ),
    )


@router.post("/aliases/{alias_id}/delete")
def delete_alias(request: Request, alias_id: int) -> Response:
    c = ctx(request)
    with c.db.write() as s:
        remove_alias(s, alias_id)
    return redirect(request, back(request, "/properties"), msg=_("Alias removed."))


@router.get("/partials/value-options")
def value_options(request: Request, key: str = "") -> Response:
    c = ctx(request)
    values: list[tuple[str, int]] = []
    if key:
        with c.db.session() as s:
            values = cached_property_counts(c, s, None, key=key).get(key, [])[:500]
    return render(request, "partials/value_options.html", values=values)
