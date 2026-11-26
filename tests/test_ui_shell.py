"""The app shell: sidebar layout rules and the loading indicator.

The browser behaviour itself was checked with Playwright; these tests keep the rules that make
it work from being lost in later edits.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from vision_tagger.web.app import create_app
from vision_tagger.web.common import STATIC_DIR, TEMPLATES_DIR

CSS = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")


def rule(selector: str) -> str:
    """The declarations of the first rule whose selector list is exactly ``selector``."""
    match = re.search(r"(?:^|\n)" + re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert match, f"no CSS rule for {selector}"
    return match.group(1)


def test_sidebar_never_scrolls_sideways():
    # long words, paths and translations wrap or are cut off instead of widening the sidebar
    sidebar = rule(".sidebar")
    assert "overflow-x: hidden" in sidebar
    assert "overflow-wrap: anywhere" in sidebar
    # the grid of the footer (activity, buttons, language) can't grow past the sidebar
    assert "minmax(0, 1fr)" in rule(".sidebar-foot")
    # a menu entry with a long label grows in height instead of overflowing
    navlink = rule(".navlink")
    assert "min-height" in navlink and not re.search(r"(?<!-)height:", navlink)
    assert "min-width: 0" in rule(".navlink > span")
    # file paths in the activity indicator end with an ellipsis
    small = rule(".activity .what small")
    assert "text-overflow: ellipsis" in small and "overflow: hidden" in small
    assert "white-space: normal" in rule(".install-btn")


def test_loading_indicator_is_on_every_page(settings):
    with TestClient(create_app(settings)) as c:
        for path in ("/", "/images", "/pipeline", "/import"):
            page = c.get(path).text
            assert '<div class="page-progress" aria-hidden="true"></div>' in page, path
    assert ".is-loading .page-progress" in CSS
    assert "prefers-reduced-motion" in CSS.split(".page-progress", 1)[1]


def test_loading_indicator_ignores_polling_and_downloads():
    assert "window.vtLoading" in JS
    # polling and lazy loading run all the time: they must not show the bar
    assert "/\\b(every|load|revealed|intersect)\\b/" in JS
    assert 'a.hasAttribute("download")' in JS
    # a request whose element is swapped out still ends the bar
    assert 'xhr.addEventListener("loadend"' in JS
    # navigations started from scripts show it too
    assert JS.count("window.vtLoading.start();") >= 2


def test_download_links_keep_the_page():
    """Downloads don't leave the page, so they must not start the loading indicator."""
    for name, href in (
        ("images.html", "/images/download.zip"),
        ("partials/search_results.html", "/search/download.zip"),
        ("generations_person.html", "/animation.webp?download=1"),
    ):
        text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
        links = [tag for tag in re.findall(r"<a [^>]*>", text) if href in tag]
        assert links, name
        assert all(" download " in tag for tag in links), name
