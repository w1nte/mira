"""Template setup and small helpers shared by the route modules."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from sqlalchemy.orm import Session

from .. import APP_NAME, i18n
from ..i18n import N_, _
from ..query import ImageFilter, property_counts
from .context import AppContext

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.add_extension("jinja2.ext.i18n")
templates.env.install_gettext_callables(  # type: ignore[attr-defined]
    i18n.gettext, i18n.ngettext, newstyle=True, pgettext=i18n.pgettext
)
templates.env.globals.update(
    languages=i18n.LANGUAGES,
    current_lang=i18n.current,
    js_i18n=i18n.js_strings,
    app_name=APP_NAME,
)


def ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def render(request: Request, name: str, **context: Any) -> Response:
    context.setdefault("msg", request.query_params.get("msg"))
    context.setdefault("err", request.query_params.get("err"))
    return templates.TemplateResponse(request, name, context)


def redirect(
    request: Request, url: str, msg: str | None = None, err: str | None = None
) -> Response:
    if msg or err:
        sep = "&" if "?" in url else "?"
        url = url + sep + urlencode({k: v for k, v in (("msg", msg), ("err", err)) if v})
    if request.headers.get("HX-Request"):
        return Response(status_code=204, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


def back(request: Request, default: str = "/") -> str:
    """Where to return after a POST: the referring page of *this* UI, else ``default``.

    Only the path and query of a same-host Referer are used (no open redirects).
    """
    ref = request.headers.get("referer")
    if not ref:
        return default
    parts = urlsplit(ref)
    if parts.netloc.lower() != request.headers.get("host", "").lower() or not parts.path:
        return default
    query = "&".join(p for p in parts.query.split("&") if p and not p.startswith(("msg=", "err=")))
    return parts.path + (f"?{query}" if query else "")


def cached_property_counts(
    c: AppContext,
    s: Session,
    f: ImageFilter | None,
    *,
    key: str | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, list[tuple[str, int]]]:
    """:func:`query.property_counts`; for the whole library cached until the next commit.

    Counting millions of property rows is the slowest part of the property, gallery and
    overview pages. The result is shared: do not modify it.
    """
    if f is not None and not f.is_empty:
        return property_counts(s, f, key=key, keys=keys)
    wanted = tuple(keys) if keys is not None else None
    return c.cache.get(
        ("property_counts", key, wanted),
        lambda s2: property_counts(s2, None, key=key, keys=wanted),
    )


def ints(values: list[str] | None) -> list[int]:
    out = []
    for v in values or []:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- template filters
def fmt_dt(value: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return value.strftime(fmt) if value else "–"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "–"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "–"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# keys whose display name is not just the key: the detector's "animal" next to the vision
# model's free-text "animals" would read as the same thing twice
KEY_LABELS = {"animal": "Species"}


def label_text(key: str) -> str:
    """English display name of a property key (``location_type`` -> ``Location type``)."""
    return KEY_LABELS.get(key) or key.replace("_", " ").capitalize()


def pretty_text(value: object) -> str:
    """English display form of a property value (``meeting_room`` -> ``meeting room``)."""
    return str(value).replace("_", " ")


def label(key: str | None) -> str:
    """Translated display name of a property key (the key itself stays English)."""
    if not key:
        return ""
    return i18n.gettext(label_text(key))


# property keys whose values have their own catalog entries ("orientation|landscape"): the
# same English word means something else elsewhere (composition "landscape" / "portrait")
VALUE_CONTEXTS = frozenset({"orientation"})


def pretty(value: object, key: str | None = None) -> str:
    """Translated display form of a property value; stored values and URLs stay English.

    Values of the vocabulary are in the catalogs; free text (objects, tags) is shown as it is
    unless it happens to be a known word. Pass the ``key`` (``{{ v | pretty(key) }}``): some
    words need it as context.
    """
    text = pretty_text(value)
    if key in VALUE_CONTEXTS:
        return i18n.pgettext(key, text)
    return i18n.gettext(text)


# display names of internal identifiers (badges, the activity indicator); unknown ones as they are
STAGE_TITLES = {
    "faces": N_("Faces"),
    "ages": N_("Face ages"),
    "animals": N_("Animals"),
    "vlm": N_("Properties"),
    "embed": N_("Search index"),
}
STATUS_TITLES = {
    "pending": N_("pending"),
    "running": N_("running"),
    "done": N_("done"),
    "error": N_("error"),
    "skipped": N_("skipped"),
    "completed": N_("completed"),
    "failed": N_("failed"),
    "paused": N_("paused"),
    "stopped": N_("stopped"),
    "interrupted": N_("interrupted"),
}
JOB_TITLES = {
    "index": N_("Index"),
    "recluster": N_("Recluster"),
    "regroup": N_("Regroup animals"),
    "references": N_("References"),
    "export": N_("Export"),
    "sidecars": N_("Sidecars"),
    "import": N_("Database import"),
    "locations": N_("Location check"),
}


def _title(titles: dict[str, str], name: str | None) -> str:
    if not name:
        return ""
    return i18n.gettext(titles[name]) if name in titles else name


def stage_title(name: str | None) -> str:
    return _title(STAGE_TITLES, name)


def status_title(name: str | None) -> str:
    return _title(STATUS_TITLES, name)


def job_title(name: str | None) -> str:
    return _title(JOB_TITLES, name)


def pet_label(pet: Any) -> str:
    """A pet's name; unnamed groups are called by species and number ("Dog #12")."""
    if pet is None:
        return ""
    if pet.name:
        return pet.name
    species = pretty(pet.species, "animal")
    return _("%(species)s #%(id)s", species=species[:1].upper() + species[1:], id=pet.id)


def pct(part: int | float, total: int | float) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


_ICON_TOKEN = re.compile(r"^[a-z0-9 -]*$")


def icon(name: str, cls: str = "") -> Markup:
    """An inline SVG icon from the sprite in ``partials/icons.html`` (``<symbol id="i-NAME">``).

    Only literal names and classes are expected; anything else is a programming error.
    """
    if not name or not _ICON_TOKEN.fullmatch(name) or not _ICON_TOKEN.fullmatch(cls):
        raise ValueError(f"invalid icon {name!r} / class {cls!r}")
    classes = f"i {cls}".strip()
    return Markup(f'<svg class="{classes}" aria-hidden="true"><use href="#i-{name}"/></svg>')


templates.env.filters.update(
    dt=fmt_dt, bytes=fmt_bytes, duration=fmt_duration, label=label, pretty=pretty
)
templates.env.globals.update(
    pct=pct,
    urlencode=urlencode,
    now=datetime.now,
    icon=icon,
    stage_title=stage_title,
    status_title=status_title,
    job_title=job_title,
    pet_label=pet_label,
)
