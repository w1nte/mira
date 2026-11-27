"""Which files Mira may open for a photo (``access.photo_file``).

Photo paths come from the database, and a database can be imported: a crafted file could name
the server's key, its configuration or other people's files as a "photo". Mira must never read
them, in the web UI, the ZIP download or the location check.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vision_tagger.access import checker, is_below, network_share, photo_file
from vision_tagger.config import Settings
from vision_tagger.db.models import Image
from vision_tagger.export import export_images, stream_zip, write_sidecars_in_place
from vision_tagger.imaging import PREVIEW_SIZE, thumbnail_path
from vision_tagger.indexer import Indexer
from vision_tagger.query import ImageFilter
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


# --------------------------------------------------------------------------- network shares
SHARE = "//attacker.example/share"  # the forward-slash form is an absolute path everywhere


@pytest.fixture
def stats(monkeypatch):
    """Paths on the attacker's share that something looked at (os.stat: exists, isdir...)."""
    looked: list[str] = []
    real = os.stat

    def stat(path, *args, **kwargs):
        if "attacker" in os.fspath(path):
            looked.append(os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    return looked


def test_network_share_of_a_path():
    assert network_share(r"\\NAS\Photos\2021\a.jpg") == r"\\nas\photos"
    assert network_share("//nas/photos/a.jpg") == r"\\nas\photos"
    assert network_share(r"C:\Photos\a.jpg") is None
    assert network_share("/home/me/a.jpg") is None


def test_network_photos_only_on_shares_of_library_folders(settings):
    nas = r"\\nas\photos"
    assert photo_file(settings, rf"{nas}\2021\a.jpg") is None  # no library folder there
    assert photo_file(settings, rf"{nas}\2021\a.jpg", roots=[rf"{nas}\2021"]) is not None
    assert photo_file(settings, r"\\NAS\Photos\x\b.jpg", roots=[nas]) is not None  # any case
    assert photo_file(settings, rf"{SHARE}/a.jpg", roots=[nas]) is None  # another host
    assert photo_file(settings, r"\\nas\other\a.jpg", roots=[nas]) is None  # another share
    assert checker(settings, [nas])(rf"{nas}\a.jpg") and not checker(settings)(rf"{nas}\a.jpg")


def test_location_check_never_contacts_unknown_shares(db, settings, photos, stats):
    real = make_image(photos / "here.jpg", seed=3)
    with db.write() as s:
        on_share = add_image(s, 1, path=f"{SHARE}/a.jpg")
        root_on_share = add_image(s, 2, path=str(real))  # only the root column is on it
        on_share.root = root_on_share.root = SHARE
        ids = on_share.id, root_on_share.id
    assert Indexer(db, settings).check_locations() == 1
    with db.session() as s:
        assert [s.get(Image, i).missing for i in ids] == [True, False]
    assert stats == []


def test_scan_relinks_without_contacting_unknown_shares(db, settings, photos, stats):
    """A found file with the content of a photo whose stored path is on an unknown share: the
    photo is linked to the found file; the share is never looked at."""
    real = make_image(photos / "here.jpg", seed=4)
    Indexer(db, settings).run()
    with db.write() as s:
        img = s.scalar(select(Image))
        img.path, img.path_key, img.root = f"{SHARE}/a.jpg", f"{SHARE}/a.jpg", SHARE
        image_id = img.id
    Indexer(db, settings).run()
    with db.session() as s:
        img = s.get(Image, image_id)
        assert img.path == str(real) and not img.missing
    assert stats == []


# --------------------------------------------------------------------------- export names
def test_export_stays_inside_the_target_folder(db, settings, photos, tmp_path):
    src = make_image(photos / "a.jpg", seed=1)
    with db.write() as s:
        for n, (rel, name) in enumerate(
            [
                ("../../outside/rel.jpg", "a.jpg"),
                ("..\\..\\outside\\win.jpg", "a.jpg"),
                ("C:/Windows/drive.jpg", "a.jpg"),
                ("", "../../outside/flat.jpg"),
            ]
        ):
            img = add_image(s, n, path=str(src))
            img.rel_path, img.filename = rel, name
    out = tmp_path / "export" / "target"
    for layout in ("folder", "flat", "date"):
        stats = export_images(db, ImageFilter(), out, allowed=checker(settings), layout=layout)
        assert stats.exported == 4 and not stats.errors, (layout, stats.errors)
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.jpg")}
    assert written - {"photos/a.jpg"} == {p for p in written if p.startswith("export/target/")}, (
        written
    )
    assert (out / "outside" / "rel.jpg").exists() and (out / "outside" / "win.jpg").exists()
    assert (out / "flat.jpg").exists()


def test_export_does_not_follow_links_out_of_the_target(db, settings, photos, tmp_path):
    src = make_image(photos / "a.jpg", seed=1)
    out, elsewhere = tmp_path / "target", tmp_path / "elsewhere"
    out.mkdir()
    elsewhere.mkdir()
    try:
        (out / "sub").symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need extra rights here")
    with db.write() as s:
        add_image(s, 1, path=str(src)).rel_path = "sub/x.jpg"
    stats = export_images(db, ImageFilter(), out, allowed=checker(settings), layout="folder")
    assert stats.exported == 0 and "outside the target folder" in stats.errors[0]
    assert not list(elsewhere.iterdir())


def test_export_reads_only_allowed_photos(db, settings, photos, tmp_path):
    src = make_image(photos / "a.jpg", seed=1)
    with db.write() as s:
        add_image(s, 1, path=str(src))
        add_image(s, 2, path=str(settings.data_dir / "secret.key"))
    (settings.data_dir / "secret.key").write_bytes(b"key")
    stats = export_images(db, ImageFilter(), tmp_path / "out", allowed=checker(settings))
    assert stats.exported == 1 and stats.skipped == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "export_manifest.csv",
        "img1.jpg",
    ]
    stats = write_sidecars_in_place(db, ImageFilter(), allowed=checker(settings))
    assert stats.sidecars == 1 and not (settings.data_dir / "secret.key.xmp").exists()


def test_zip_entry_names_are_plain_file_names(db, settings, photos):
    src = make_image(photos / "a.jpg", seed=1)
    with db.write() as s:
        add_image(s, 1, path=str(src)).filename = "../../evil.jpg"
        add_image(s, 2, path=str(src)).filename = "C:\\x\\con.jpg"
        ids = [1, 2]
    data = b"".join(stream_zip(db, ids, allowed=checker(settings), metadata=False))
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == ["evil.jpg", "_con.jpg"]
