from __future__ import annotations

import io
import time
import zipfile
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from vision_tagger.config import Settings
from vision_tagger.db.models import Image
from vision_tagger.uploads import UploadError, UploadStore, safe_relative_path
from vision_tagger.web.app import create_app

from .conftest import make_image


def jpeg_bytes(seed: int) -> bytes:
    import numpy as np
    from PIL import Image as PILImage

    arr = np.random.default_rng(seed).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).resize((320, 240)).save(buf, "JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------- local mode
def test_local_folders_browser_and_scan(settings, photos, tmp_path):
    make_image(photos / "a" / "x.jpg", seed=1)
    make_image(photos / "a" / "y.jpg", seed=2)
    app = create_app(settings)
    with TestClient(app) as c:
        page = c.get("/import").text
        assert (
            "Your photo folders" in page and "vtFolderPicker" in page and "vtUploader" not in page
        )
        listing = c.get("/api/fs", params={"path": str(photos)}).json()
        assert [f["name"] for f in listing["folders"]] == ["a"] and listing["images"] == 0
        assert c.get("/api/fs", params={"path": str(photos / "a")}).json()["images"] == 2
        assert c.get("/api/fs").json()["folders"]  # start: home folder (and drives / root)
        c.post("/import/roots", data={"path": str(photos / "a")})
        assert str((photos / "a").resolve()) in c.get("/import").text
        assert "Not a folder" in c.post("/import/roots", data={"path": str(tmp_path / "nope")}).text
        r = c.post("/actions/index", data={"analyse": ""}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/import")
        _wait(app)
        with app.state.ctx.db.session() as s:
            assert s.scalar(select(func.count(Image.id))) == 2
        for root in (photos / "a", photos):  # the configured folder counts as well
            c.post("/import/roots/remove", data={"path": str(root.resolve())})
        assert "No folders yet" in c.get("/import").text
        assert c.post("/api/uploads", json={"name": "a.jpg", "size": 1}).status_code == 403


# --------------------------------------------------------------------------- remote mode
@pytest.fixture
def remote(tmp_path):
    s = Settings(
        mode="remote",
        remote={"upload_dir": tmp_path / "library", "auto_process": True},
        paths={"data_dir": tmp_path / "data"},
        vlm={"backend": "fake"},
        pipeline={"stages": ["vlm"], "batch_size": 4},
        web={"allowed_hosts": ["testserver"]},
    )
    s.ensure_dirs()
    return s


def test_safe_paths():
    assert str(safe_relative_path("../../etc/passwd")) == "etc/passwd"
    assert str(safe_relative_path("C:\\Users\\x\\IMG 1.jpg")) == "C_/Users/x/IMG 1.jpg"
    assert str(safe_relative_path("trip/con.jpg")) == "trip/_con.jpg"
    with pytest.raises(UploadError):
        safe_relative_path("../..")


def test_upload_protocol_resume_and_placement(remote):
    store = UploadStore(remote)
    data = jpeg_bytes(1)
    up = store.create("IMG_0001.jpg", len(data))
    assert store.write(up.id, 0, data[:1000]).received == 1000
    assert store.write(up.id, 0, data[:1000]).received == 1000  # retried chunk: ignored
    assert store.write(up.id, 5000, data[5000:6000]).received == 1000  # gap: client resyncs
    with pytest.raises(UploadError):
        store.complete(up.id)  # incomplete
    store.write(up.id, 1000, data[1000:])
    done, placed = store.complete(up.id)
    assert done.state == "done" and placed[0].read_bytes() == data
    assert placed[0].parent == remote.upload_dir / date.today().isoformat()
    assert store.complete(up.id)[1] == []  # completing twice is harmless
    # a second file with the same name does not overwrite the first
    up2 = store.create("IMG_0001.jpg", len(data))
    store.write(up2.id, 0, data)
    assert store.complete(up2.id)[1][0].name == "IMG_0001 (2).jpg"
    # folder uploads keep their structure; only photos and ZIPs are accepted
    up3 = store.create("Holiday 2020/day1/a.jpg", len(data))
    store.write(up3.id, 0, data)
    assert store.complete(up3.id)[1][0] == remote.upload_dir / "Holiday 2020" / "day1" / "a.jpg"
    with pytest.raises(UploadError):
        store.create("notes.txt", 10)
    with pytest.raises(UploadError):
        store.get("../../secret")


def test_zip_upload_is_unpacked_safely(remote):
    store = UploadStore(remote)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Summer/a.jpg", jpeg_bytes(1))
        zf.writestr("../../escape.jpg", jpeg_bytes(2))  # path trick: lands inside the folder
        zf.writestr("__MACOSX/._a.jpg", b"junk")
        zf.writestr("readme.txt", b"not a photo")
    data = buf.getvalue()
    up = store.create("photos.zip", len(data))
    store.write(up.id, 0, data)
    done, placed = store.complete(up.id)
    rel = sorted(p.relative_to(remote.upload_dir).as_posix() for p in placed)
    assert rel == ["photos/Summer/a.jpg", "photos/escape.jpg"] and done.files == 2
    assert not (remote.upload_dir.parent / "escape.jpg").exists()

    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.jpg", b"\0" * 5_000_000)  # compresses ~1000:1
    data = bomb.getvalue()
    up = store.create("bomb.zip", len(data))
    store.write(up.id, 0, data)
    with pytest.raises(UploadError, match="ratio"):
        store.complete(up.id)


def test_remote_uploads_are_indexed_and_processed(remote):
    app = create_app(remote)
    with TestClient(app) as c:
        app.state.ctx.ingest.quiet_seconds = 0.2
        page = c.get("/import").text
        assert "vtUploader" in page and "vtFolderPicker" not in page
        assert c.get("/api/fs").status_code == 403  # no browsing of the server's disks
        for seed in range(3):
            data = jpeg_bytes(seed)
            up = c.post("/api/uploads", json={"name": f"p{seed}.jpg", "size": len(data)}).json()
            r = c.put(f"/api/uploads/{up['id']}?offset=0", content=data)
            assert r.json()["received"] == len(data)
            assert c.get(f"/api/uploads/{up['id']}").json()["received"] == len(data)
            assert c.post(f"/api/uploads/{up['id']}/complete").json()["state"] == "done"
        assert c.post("/api/uploads", json={"name": "x.exe", "size": 3}).status_code == 400
        r = c.put(
            "/api/uploads/abc?offset=0", content=b"x", headers={"Origin": "http://evil.example"}
        )
        assert r.status_code == 403
        deadline = time.time() + 30
        ctx = app.state.ctx
        while time.time() < deadline:
            with ctx.db.session() as s:
                n = s.scalar(select(func.count(Image.id))) or 0
            if n == 3 and not ctx.jobs.busy and not ctx.runner.running and ctx.runner.last_summary:
                break
            time.sleep(0.2)
        assert n == 3
        assert ctx.runner.last_summary and ctx.runner.last_summary.processed.get("vlm") == 3


def _wait(app, timeout=20):
    deadline = time.time() + timeout
    while app.state.ctx.jobs.busy:
        assert time.time() < deadline
        time.sleep(0.05)
