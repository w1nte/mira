"""Template setup and small helpers shared by the route modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .context import AppContext

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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
    ref = request.headers.get("referer") or default
    # drop old flash params
    for key in ("msg=", "err="):
        if key in ref:
            base, _, query = ref.partition("?")
            parts = [p for p in query.split("&") if not p.startswith(key)]
            ref = base + ("?" + "&".join(parts) if parts else "")
    return ref


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


def label(key: str | None) -> str:
    if not key:
        return ""
    return key.replace("_", " ").capitalize()


def pretty(value: object) -> str:
    """Display form of a property value (``meeting_room`` -> ``meeting room``)."""
    return str(value).replace("_", " ")


def pct(part: int | float, total: int | float) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


templates.env.filters.update(
    dt=fmt_dt, bytes=fmt_bytes, duration=fmt_duration, label=label, pretty=pretty
)
templates.env.globals.update(pct=pct, urlencode=urlencode, now=datetime.now)
