"""Installable app (PWA): web manifest and service worker, served from the site root."""

from __future__ import annotations

import hashlib
from functools import cache

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ... import __version__
from ...i18n import _
from ..common import STATIC_DIR

router = APIRouter()


@cache
def _static_version() -> str:
    """Changes whenever a static file changes, so installed apps pick up new versions."""
    h = hashlib.sha256(__version__.encode())
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.is_file():
            st = path.stat()
            h.update(f"{path.relative_to(STATIC_DIR)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


@router.get("/manifest.webmanifest")
def manifest() -> JSONResponse:
    icons = [
        {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
        {
            "src": "/static/icons/icon-maskable-512.png",
            "sizes": "512x512",
            "type": "image/png",
            "purpose": "maskable",
        },
    ]
    return JSONResponse(
        {
            "id": "/",
            "name": "vision-tagger",
            "short_name": "vision-tagger",
            "description": _(
                "Your photos: people, places and moments, found on your own computer."
            ),
            "start_url": "/",
            "scope": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#171717",
            "icons": icons,
            "shortcuts": [
                {"name": _("Search"), "url": "/search", "icons": icons[:1]},
                {"name": _("Images"), "url": "/images", "icons": icons[:1]},
                {"name": _("Import"), "url": "/import", "icons": icons[:1]},
            ],
        },
        media_type="application/manifest+json",
    )


@router.get("/sw.js")
def service_worker() -> Response:
    code = (
        (STATIC_DIR / "sw.js").read_text(encoding="utf-8").replace("__VERSION__", _static_version())
    )
    return Response(
        code,
        media_type="text/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )
