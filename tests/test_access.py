"""Which files Mira may open for a photo (``access.photo_file``).

Photo paths come from the database, and a database can be imported: a crafted file could name
the server's key, its configuration or other people's files as a "photo". Mira must never read
them, in the web UI, the ZIP download or the location check.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vision_tagger.access import is_below, photo_file
from vision_tagger.config import Settings
from vision_tagger.db.models import Image
from vision_tagger.imaging import PREVIEW_SIZE, thumbnail_path
from vision_tagger.indexer import Indexer
from vision_tagger.web.app import create_app

from .conftest import make_image
from .helpers import add_face, add_image


def remote_settings(tmp_path: Path, **kw) -> Settings:
    s = Settings(
        mode="remote",
        remote={"upload_dir": tmp_path / "library", "auto_process": False},
        paths={"data_dir": tmp_path / "data", "references_dir": tmp_path / "refs"},
        vlm={"backend": "fake"},
        web={"password": "s3cret"},
        **kw,
    )
    s.ensure_dirs()
    return s


# --------------------------------------------------------------------------- rules
def test_only_image_files_by_their_own_extension(settings, tmp_path):
    assert photo_file(settings, tmp_path / "a.jpg") == tmp_path / "a.jpg"
    assert photo_file(settings, str(tmp_path / "B.JPEG")) is not None
    assert photo_file(settings, tmp_path / "x.heic") is not None
    for bad in ("secret.key", "vision_tagger.sqlite3", "config.toml", "a.jpg.txt", ".jpg", ""):
        assert photo_file(settings, tmp_path / bad if bad else bad) is None, bad
    assert photo_file(settings, None) is None
    assert photo_file(settings, "/x/a\x00.jpg") is None


def test_local_mode_reads_photos_anywhere(settings, tmp_path):
    """Local mode: the library folders are the user's own choice (any drive, a NAS)."""
    assert photo_file(settings, tmp_path / "elsewhere" / "a.png") is not None


def test_remote_mode_reads_only_below_the_upload_folder(tmp_path):
    s = remote_settings(tmp_path)
    inside = s.upload_dir / "trip" / "a.jpg"
    assert photo_file(s, inside) == inside
    assert photo_file(s, tmp_path / "elsewhere" / "a.jpg") is None
    assert photo_file(s, s.upload_dir / ".." / "elsewhere" / "a.jpg") is None  # no escape
    assert photo_file(s, f"{s.upload_dir}-other/a.jpg") is None  # a sibling with that prefix
    assert photo_file(s, s.upload_dir) is None
    refs = tmp_path / "refs" / "Anna" / "1.jpg"
    assert photo_file(s, refs) is None
    assert photo_file(s, refs, extra_roots=[tmp_path / "refs"]) == refs


def test_is_below_is_lexical(tmp_path):
    assert is_below(tmp_path / "a" / "b.jpg", tmp_path / "a")
    assert not is_below(tmp_path / "a" / ".." / "b.jpg", tmp_path / "a")
    assert not is_below(tmp_path / "a", tmp_path / "a")
    assert not is_below(tmp_path / "ab" / "c.jpg", tmp_path / "a")


# --------------------------------------------------------------------------- web app
@pytest.fixture
def crafted(tmp_path, monkeypatch):
    """Remote mode with one real photo and rows a crafted database could contain."""
    monkeypatch.setattr("vision_tagger.web.routes.auth.time.sleep", lambda s: None)
    s = remote_settings(tmp_path)
    make_image(s.upload_dir / "trip" / "ok.jpg", seed=1)
    secret = s.data_dir / "secret.key"  # written by create_app (session key)
    elsewhere = make_image(tmp_path / "elsewhere" / "private.jpg", seed=2)
    app = create_app(s)
    with TestClient(app) as c:
        assert c.post("/login", data={"password": "s3cret"}).status_code == 200
        db = app.state.ctx.db
        Indexer(db, s).run()
        with db.write() as session:
            ok_id = session.scalar(select(Image.id).where(Image.filename == "ok.jpg"))
            rows = {
                "key": add_image(session, 101, path=str(secret)),  # ext ".jpg" in the database
                "outside": add_image(session, 102, path=str(elsewhere)),
                "escape": add_image(
                    session, 103, path=str(s.upload_dir / ".." / "elsewhere" / "private.jpg")
                ),
            }
            face = add_face(session, rows["outside"].id, np.ones(512, dtype=np.float32))
            ids = {name: img.id for name, img in rows.items()}
        assert secret.exists()
        c.ids, c.ok_id, c.face_id, c.settings, c.db = ids, ok_id, face.id, s, db
        yield c


def test_media_routes_never_read_files_outside_the_library(crafted):
    c = crafted
    assert c.get(f"/media/image/{c.ok_id}").status_code == 200
    assert c.get(f"/media/thumb/{c.ok_id}").headers["content-type"] == "image/webp"
    for name, image_id in c.ids.items():
        r = c.get(f"/media/image/{image_id}")
        assert r.status_code == 404, name
        assert b"SQLite" not in r.content and len(r.content) < 200, name
        for route in ("thumb", "preview"):
            r = c.get(f"/media/{route}/{image_id}")
            assert r.status_code == 404 and r.headers["content-type"] == "image/svg+xml", name
    assert c.get(f"/media/face/{c.face_id}").status_code == 404
    assert c.get(f"/media/aligned/{c.face_id}").status_code == 404
    for image_id in c.ids.values():  # nothing was decoded either
        for size in (c.settings.web.thumb_size, PREVIEW_SIZE):
            assert not thumbnail_path(c.settings.thumbs_dir, image_id, size).exists()


def test_original_is_served_with_the_type_of_the_file_itself(crafted):
    c = crafted
    with c.db.write() as s:  # the database claims another type
        s.get(Image, c.ok_id).ext = ".png"
    r = c.get(f"/media/image/{c.ok_id}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_zip_download_leaves_out_files_outside_the_library(crafted):
    c = crafted
    r = c.get("/images/download.zip")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "ok.jpg" in names
    assert not [n for n in names if n.startswith("img10")]
    ids = ",".join(str(i) for i in [c.ok_id, *c.ids.values()])
    names = zipfile.ZipFile(io.BytesIO(c.get(f"/search/download.zip?ids={ids}").content)).namelist()
    assert [n for n in names if not n.endswith((".xmp", ".json"))] == ["ok.jpg"]


def test_location_check_flags_files_mira_may_not_open(crafted):
    c = crafted
    assert Indexer(c.db, c.settings).check_locations() == len(c.ids)
    with c.db.session() as s:
        missing = set(s.scalars(select(Image.id).where(Image.missing)))
    assert missing == set(c.ids.values())
