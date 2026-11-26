from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import select

from vision_tagger import imaging
from vision_tagger.db.models import Image as ImageRow
from vision_tagger.imaging import ensure_thumbnail, load_image, thumbnail_path
from vision_tagger.indexer import Indexer
from vision_tagger.web.app import create_app

from .conftest import make_image


def big_photo(path: Path, size=(2400, 1800), orientation: int = 1, fmt: str | None = None):
    """A camera-sized photo with a red block top-left (to check orientation)."""
    im = Image.new("RGB", size, (40, 90, 160))
    ImageDraw.Draw(im).rectangle([0, 0, size[0] // 3, size[1] // 3], fill=(230, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "HEIF":
        im.save(path, quality=80, exif=exif.tobytes(), thumbnails=[320])
    else:
        im.save(path, quality=90, exif=exif)
    return path


def ids_and_paths(db):
    with db.session() as s:
        return s.execute(select(ImageRow.id, ImageRow.path).order_by(ImageRow.id)).all()


def test_indexing_writes_gallery_thumbnails(db, settings, photos):
    big_photo(photos / "landscape.jpg")
    big_photo(photos / "portrait.jpg", orientation=6)  # stored sideways, shown upright
    make_image(photos / "small.png", seed=3, size=(200, 150), fmt="PNG")
    stats = Indexer(db, settings).run()
    size = settings.web.thumb_size
    assert stats.new == 3 and stats.thumbnails == 3 and stats.thumbnails_todo == 0
    sizes = {}
    for image_id, path in ids_and_paths(db):
        thumb = thumbnail_path(settings.thumbs_dir, image_id, size)
        assert thumb.exists()
        with Image.open(thumb) as im:
            sizes[Path(path).name] = im.size
            if Path(path).name == "portrait.jpg":
                assert im.getpixel((5, 5))[0] < 100  # red block is now top-right
                assert im.getpixel((im.width - 5, 5))[0] > 200
    assert sizes["landscape.jpg"] == (size, size * 3 // 4)
    assert sizes["portrait.jpg"] == (size * 3 // 4, size)
    assert sizes["small.png"] == (200, 150)  # never enlarged


def test_missing_thumbnails_are_made_for_files_indexed_earlier(db, settings, photos):
    for i in range(5):
        make_image(photos / f"p{i}.jpg", seed=i, size=(640, 480))
    settings.index.thumbnails = False
    Indexer(db, settings).run()
    assert not any(settings.thumbs_dir.rglob("*.webp"))  # switched off

    settings.index.thumbnails = True
    stats = Indexer(db, settings).run()  # nothing changed on disk
    assert stats.unchanged == 5 and stats.thumbnails_todo == 5 and stats.thumbnails_warmed == 5
    for image_id, _path in ids_and_paths(db):
        assert thumbnail_path(settings.thumbs_dir, image_id, settings.web.thumb_size).exists()
    assert Indexer(db, settings).run().thumbnails_todo == 0


def test_thumbnail_step_stops_on_request(db, settings, photos):
    for i in range(3):
        make_image(photos / f"p{i}.jpg", seed=i)
    settings.index.thumbnails = False
    Indexer(db, settings).run()
    settings.index.thumbnails = True
    stop = {"now": False}

    def progress(stats):  # the user presses "stop" once the thumbnail step has started
        if stats.thumbnails_todo:
            stop["now"] = True

    stats = Indexer(db, settings, progress=progress, should_stop=lambda: stop["now"]).run()
    assert stats.thumbnails_todo == 3 and stats.thumbnails_warmed == 0


def test_small_sizes_come_from_the_cached_thumbnail(tmp_path, monkeypatch):
    src = big_photo(tmp_path / "photo.jpg")
    base = ensure_thumbnail(tmp_path / "thumbs", 1, src, 320)

    def no_decode(*args, **kwargs):
        raise AssertionError("the original photo must not be decoded again")

    monkeypatch.setattr(imaging, "load_image", no_decode)
    small = ensure_thumbnail(tmp_path / "thumbs", 1, src, 96, base_size=320)
    with Image.open(small) as im:
        assert im.size == (96, 72)
    assert base.exists()


def test_stale_thumbnails_are_rebuilt(tmp_path):
    src = big_photo(tmp_path / "photo.jpg")
    thumb = ensure_thumbnail(tmp_path / "thumbs", 1, src, 160)
    old = time.time() - 100
    os.utime(thumb, (old, old))  # the photo was edited after the thumbnail was made
    assert ensure_thumbnail(tmp_path / "thumbs", 1, src, 160).stat().st_mtime > old + 50


def test_draft_decoding_is_reduced_but_large_enough(tmp_path):
    jpg = big_photo(tmp_path / "photo.jpg", size=(4000, 3000))
    im = load_image(jpg, max_px=640, draft_px=320)
    assert 320 <= max(im.size) <= 640  # decoded at 1/8, not 12 megapixels
    heic = big_photo(tmp_path / "photo.heic", size=(4000, 3000), orientation=6, fmt="HEIF")
    im = load_image(heic, max_px=640, draft_px=320)
    assert im.size == (240, 320)  # the embedded preview, upright like the full image
    assert im.getpixel((5, 5))[0] < 100 and im.getpixel((235, 5))[0] > 200
    assert load_image(heic).size == (3000, 4000)  # without draft_px: the full image


@pytest.fixture
def web(settings, photos):
    big_photo(photos / "a.jpg")
    app = create_app(settings)
    with TestClient(app) as c:
        Indexer(app.state.ctx.db, settings).run()
        yield c


def test_thumb_route_serves_cached_and_derived_sizes(web, settings):
    image_id = 1
    base = thumbnail_path(settings.thumbs_dir, image_id, settings.web.thumb_size)
    assert base.exists()  # written by the indexer
    r = web.get(f"/media/thumb/{image_id}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    r = web.get(f"/media/thumb/{image_id}?size=96")
    assert r.status_code == 200
    with Image.open(thumbnail_path(settings.thumbs_dir, image_id, 96)) as im:
        assert max(im.size) == 96
    assert web.get("/media/thumb/999").status_code == 404
