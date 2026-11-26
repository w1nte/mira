"""The "What's new" page: the bundled changelog, the running version marked.

The entries are English (``resources/CHANGELOG.md``); the page around them is translated.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from markupsafe import Markup

from ... import __version__, changelog
from ..common import render

router = APIRouter()


def _item(text: str) -> Markup:
    return Markup(changelog.inline_html(text))  # everything escaped but code and web links


@router.get("/changelog")
def changelog_page(request: Request) -> Response:
    releases = changelog.releases()
    current = changelog.find(releases, __version__)
    return render(
        request,
        "changelog.html",
        # Unreleased only when it lists something
        releases=[r for r in releases if r.items or not r.unreleased],
        current=current,
        version=__version__,
        kinds=changelog.KINDS,
        item=_item,
    )
