"""The gallery's lightbox: the tiles' data, the overlay's markup, the large preview, NSFW covers.

The script itself (static/app.js, "Lightbox") needs a browser; these tests check what the server
hands it.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import select, update

from vision_tagger.db.models import Image
from vision_tagger.imaging import PREVIEW_SIZE, thumbnail_path
from vision_tagger.indexer import Indexer
from vision_tagger.web.app import create_app
from vision_tagger.web.common import STATIC_DIR

from .conftest import make_image
from .test_nsfw import DE, SHOW, _library, ok

DATE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d$")


class Tiles(HTMLParser):
    """The attributes of each tile link (``<a class="thumb" data-id>``) of a gallery page."""

    def __init__(self, page: str) -> None:
        super().__init__()
        self.links: list[dict[str, str | None]] = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("class") == "thumb" and "data-id" in a:
            self.links.append(a)


def tiles(page: str) -> dict[int, dict[str, str | None]]:
    return {int(a["data-id"]): a for a in Tiles(page).links}


def dialog(page: str) -> str:
    match = re.search(r'<dialog class="lightbox".*?</dialog>', page, re.S)
    assert match, "no lightbox on the page"
    return match.group(0)


@pytest.fixture
def mixed(settings, photos):
    """Six photos with Anna and Ben on each; three of them NSFW (see test_nsfw)."""
    client = _library(settings, photos, nsfw_all=False)
    with client.ctx.db.session() as s:
        client.nsfw = set(s.scalars(select(Image.id).where(Image.width == 300)))
        client.files = {img.id: img.filename for img in s.scalars(select(Image))}
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def all_nsfw(settings, photos):
    client = _library(settings, photos, nsfw_all=True)
    yield client
    client.__exit__(None, None, None)


def test_gallery_tiles_carry_what_the_lightbox_shows(mixed):
    client = mixed
    page = ok(client.get("/images")).text
    # the grid opts in, inside the selection scope; one overlay on the page
    assert '<div class="thumbs" id="grid" data-lightbox>' in page
    assert page.index("data-select-scope") < page.index("data-lightbox>")
    assert page.count("<dialog") == 1
    box = dialog(page)
    assert 'aria-modal="true"' in box and 'aria-labelledby="lb-title"' in box
    assert f'data-preview-size="{PREVIEW_SIZE}"' in box
    links = tiles(page)
    assert sorted(links) == sorted(client.files)
    safe = [i for i in links if i not in client.nsfw]
    assert len(safe) == 3 and len(client.nsfw) == 3
    for i in safe:
        a = links[i]
        assert a["href"] == f"/images/{i}"  # Ctrl+click / middle click: the photo page
        assert a["data-preview-url"] == f"/media/preview/{i}"
        assert a["data-title"] == client.files[i]
        assert DATE.match(a["data-date"] or "")
        assert int(a["data-w"]) == 320 and int(a["data-h"]) == 240
        assert json.loads(a["data-people"]) == ["Anna", "Ben"]
        assert "data-covered" not in a and "data-missing" not in a
    # the next pages (HTMX) carry the same
    more = tiles(ok(client.get("/images/page?page=1")).text)
    assert more[safe[0]]["data-preview-url"] == f"/media/preview/{safe[0]}"
    assert json.loads(more[safe[0]]["data-people"]) == ["Anna", "Ben"]
    # the texts, translated
    page = ok(client.get("/images", headers=DE)).text
    box = dialog(page)
    for text in (
        'aria-label="Schließen"',
        "Details öffnen",
        'aria-label="Vorheriges Foto"',
        'aria-label="Nächstes Foto"',
        'title="Original öffnen"',
        "Nicht jugendfrei (NSFW) \u2013 öffne das Foto, um es zu sehen",
        "Datei nicht gefunden \u2013 gezeigt wird das gespeicherte Vorschaubild.",
    ):
        assert text in box, text
    assert "Open details" not in page


def test_covered_photos_have_no_picture_url_for_the_lightbox(mixed):
    client = mixed
    page = ok(client.get("/images")).text
    links = tiles(page)
    for i in client.nsfw:
        a = links[i]
        assert "data-covered" in a and a["href"] == f"/images/{i}"  # a click opens the page
        for name in ("data-preview-url", "data-people", "data-w", "data-h"):
            assert name not in a, name
        assert f"/media/preview/{i}" not in page and f"/media/thumb/{i}" not in page
    # the overlay itself holds no picture URL: the script takes them from the tiles
    box = dialog(page)
    assert "/media/" not in box and " src=" not in box
    # "Show NSFW photos without asking": they behave like any photo
    shown = tiles(ok(client.get("/images", headers=SHOW)).text)
    assert all("data-covered" not in a for a in shown.values())
    assert all(a["data-preview-url"] == f"/media/preview/{i}" for i, a in shown.items())


def test_a_library_of_nsfw_photos_has_no_preview_url(all_nsfw):
    for url in ("/images", "/images?sort=name", "/images/page?page=1"):
        page = ok(all_nsfw.get(url)).text
        assert tiles(page) and all("data-covered" in a for a in tiles(page).values()), url
        assert "data-preview-url" not in page and "/media/preview/" not in page, url


def test_selection_and_page_loading_markup(mixed):
    client = mixed
    client.ctx.settings.web.page_size = 2
    page = ok(client.get("/images")).text
    # the tick box sits beside the link, not in it: a click on it never opens anything
    assert re.search(r'</a>\s*<input class="pick" type="checkbox" name="image_ids"', page)
    # the lightbox loads the next page through the gallery's own element at the end
    assert re.search(r'<div hx-get="/images/page\?page=2" [^>]*data-lightbox-more', page)
    assert "data-lightbox-more" not in ok(client.get("/images/page?page=3")).text
    # the script leaves clicks with a modifier key, during a selection and on covers alone
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    lightbox = js[js.index("// Lightbox (partials/lightbox.html)") :]
    click = re.compile(r'closest\("\[data-lightbox\] a\[data-id\]"\).*?open\(a, true\)', re.S)
    handler = click.search(lightbox)
    assert handler and "window.vt.selecting(a)" in handler.group(0)
    assert "ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.altKey" in lightbox
    assert "stopImmediatePropagation" in lightbox  # no gallery shortcuts while it is open


def test_missing_photo_gets_its_thumbnail_and_a_note(mixed):
    client = mixed
    i = min(set(client.files) - client.nsfw)
    with client.ctx.db.write() as s:
        s.execute(update(Image).where(Image.id == i).values(missing=True))
    assert i not in tiles(ok(client.get("/images")).text)  # left out unless asked for
    for url in ("/images?missing=1", "/images/page?page=1&missing=1"):
        a = tiles(ok(client.get(url)).text)[i]
        assert "data-missing" in a and a["data-preview-url"] == f"/media/preview/{i}", url
        assert "data-w" not in a  # sized by the picture actually shown (a thumbnail)
        assert json.loads(a["data-people"]) == ["Anna", "Ben"]


@pytest.fixture
def library(settings, photos):
    make_image(photos / "big.jpg", seed=1, size=(2400, 1800))
    make_image(photos / "small.jpg", seed=2, size=(320, 240))
    make_image(photos / "gone.jpg", seed=3, size=(2000, 1500))
    make_image(photos / "lost.jpg", seed=4, size=(2000, 1500))
    app = create_app(settings)
    with TestClient(app) as client:
        Indexer(app.state.ctx.db, settings).run()
        with app.state.ctx.db.session() as s:
            client.ids = {img.filename: img.id for img in s.scalars(select(Image))}
        client.settings = settings
        yield client


def _size(resp) -> tuple[int, int]:
    from io import BytesIO

    assert resp.headers["content-type"] == "image/webp"
    with PILImage.open(BytesIO(resp.content)) as im:
        return im.size


def test_preview_is_a_cached_large_copy(library, photos):
    client, ids = library, library.ids
    thumbs = client.settings.thumbs_dir
    # large photos: PREVIEW_SIZE on the longer side, cached next to the thumbnails
    resp = ok(client.get(f"/media/preview/{ids['big.jpg']}"))
    assert _size(resp) == (PREVIEW_SIZE, PREVIEW_SIZE * 3 // 4)
    assert "max-age" in resp.headers["cache-control"]
    cached = thumbnail_path(thumbs, ids["big.jpg"], PREVIEW_SIZE)
    assert cached.exists()
    stamp = cached.stat().st_mtime_ns
    ok(client.get(f"/media/preview/{ids['big.jpg']}"))
    assert cached.stat().st_mtime_ns == stamp  # served from the cache
    # small ones are never enlarged
    assert _size(ok(client.get(f"/media/preview/{ids['small.jpg']}"))) == (320, 240)
    # a file that is gone: its cached thumbnail; with nothing cached the placeholder
    (photos / "gone.jpg").unlink()
    w, h = _size(ok(client.get(f"/media/preview/{ids['gone.jpg']}")))
    assert max(w, h) == client.settings.web.thumb_size
    (photos / "lost.jpg").unlink()
    thumbnail_path(thumbs, ids["lost.jpg"], client.settings.web.thumb_size).unlink()
    resp = client.get(f"/media/preview/{ids['lost.jpg']}")
    assert resp.status_code == 404 and resp.headers["content-type"].startswith("image/svg")
    assert client.get("/media/preview/9999").status_code == 404
