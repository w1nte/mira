"""Database export / import, the location check, missing files and re-linking moved photos."""

from __future__ import annotations

import io
import re
import shutil
import sqlite3
import time
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from typer.testing import CliRunner

from vision_tagger.config import Settings
from vision_tagger.db import transfer
from vision_tagger.db.migrate import head_revision
from vision_tagger.db.models import (
    AppSetting,
    Control,
    Face,
    Image,
    ImageProperty,
    Person,
    Run,
    StageStatus,
    utcnow,
)
from vision_tagger.faces import clustering as C
from vision_tagger.indexer import ROOTS_KEY, Indexer, stored_roots
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.ages import AgesStage
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.faces import FacesStage
from vision_tagger.pipeline.stages.vlm import VlmStage
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app
from vision_tagger.web.context import Job

from .conftest import make_image
from .helpers import FakeAgeEstimator
from .test_faces_stage import FakeAnalyzer
from .test_migrations import build_at, fill_first_version

N_PHOTOS = 5


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def analyse(ctx, settings) -> None:
    stages = {
        "faces": FacesStage(
            ctx.db, settings, analyzer=FakeAnalyzer(), age_estimator=FakeAgeEstimator()
        ),
        "ages": AgesStage(ctx.db, settings, estimator=FakeAgeEstimator()),
        "vlm": VlmStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
        "embed": EmbedStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
    }
    PipelineRunner(ctx.db, settings, stages.__getitem__).run()


def wait_jobs(ctx, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while ctx.jobs.busy:
        assert time.time() < deadline, "job did not finish"
        time.sleep(0.05)


def snapshot(db) -> dict[int, tuple[str, int, int]]:
    """image id -> (file name, faces, properties)."""
    with db.session() as s:
        faces = dict(s.execute(select(Face.image_id, func.count()).group_by(Face.image_id)).all())
        props = dict(
            s.execute(
                select(ImageProperty.image_id, func.count()).group_by(ImageProperty.image_id)
            ).all()
        )
        return {
            i: (name, faces.get(i, 0), props.get(i, 0))
            for i, name in s.execute(select(Image.id, Image.filename))
        }


def missing_ids(db) -> set[int]:
    with db.session() as s:
        return set(s.scalars(select(Image.id).where(Image.missing)))


def import_file(client, data: bytes, name: str = "library.sqlite3", **kw):
    return client.post(
        "/database/import",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
        **kw,
    )


@pytest.fixture
def library(settings, photos):
    """An analysed library (fake models) with a named person, served by the web app."""
    for i in range(N_PHOTOS):
        make_image(photos / "trip" / f"IMG_2021010{i}_120000.jpg", seed=i)
    app = create_app(settings)
    with TestClient(app) as c:
        ctx = app.state.ctx
        Indexer(ctx.db, settings).run()
        analyse(ctx, settings)
        with ctx.db.write() as s:
            cid = s.scalar(select(Face.cluster_id).where(Face.cluster_id.is_not(None)))
            C.name_cluster(s, cid, "Anna")
        c.ctx = ctx
        yield c


def exported(client) -> bytes:
    r = ok(client.get("/database/export"))
    return r.content


# --------------------------------------------------------------------------- export
def test_export_downloads_a_consistent_snapshot(library, settings, tmp_path):
    db = library.ctx.db
    # an open write transaction of another connection (the pipeline) neither blocks the
    # export nor leaks into it
    writer = sqlite3.connect(settings.db_path, isolation_level=None, timeout=5)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO persons (name, created_at) VALUES ('Uncommitted', '2024-01-01')")
    try:
        r = ok(library.get("/database/export"))
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    expected = f"vision-tagger-{date.today().isoformat()}.sqlite3"
    assert expected in r.headers["content-disposition"]
    assert r.headers["content-type"] == "application/vnd.sqlite3"
    assert r.content.startswith(b"SQLite format 3\x00")
    copy = tmp_path / "copy.sqlite3"
    copy.write_bytes(r.content)
    con = sqlite3.connect(copy)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"  # one file
        assert con.execute("SELECT count(*) FROM images").fetchone()[0] == N_PHOTOS
        names = {row[0] for row in con.execute("SELECT name FROM persons")}
        assert "Anna" in names and "Uncommitted" not in names
        assert con.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            head_revision()
        )
    finally:
        con.close()
    # the temporary copy in the data folder is gone after the download
    assert not [p for p in transfer.transfer_dir(settings).iterdir() if p.name.startswith("ex")]
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(Person).where(Person.name == "Anna"))
    page = ok(library.get("/database")).text
    assert 'href="/database/export" download' in page and 'enctype="multipart/form-data"' in page


def test_failed_export_is_reported(library, settings, monkeypatch):
    blocker = settings.data_dir / "not-a-folder"
    blocker.write_text("x")
    monkeypatch.setattr(transfer, "temp_file", lambda _s, _k: blocker / "export.sqlite3")
    page = ok(library.get("/database/export")).text
    assert "The export failed:" in page


def test_cli_export_and_import(library, settings, tmp_path, monkeypatch):
    from vision_tagger import cli

    monkeypatch.setattr(cli, "load_settings", lambda _config=None: settings)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["db", "export", str(tmp_path)])
    assert r.exit_code == 0, r.output
    target = tmp_path / transfer.export_name()
    assert target.exists() and "exported" in r.output
    assert runner.invoke(cli.app, ["db", "export", str(target)]).exit_code == 1  # exists
    with library.ctx.db.write() as s:
        s.add(Person(name="Added later"))
    before = target.read_bytes()
    r = runner.invoke(cli.app, ["db", "import", str(target)])
    assert r.exit_code == 0, r.output
    assert target.read_bytes() == before  # the given file is not changed
    assert f"imported {N_PHOTOS} photos" in r.output and "missing: 0 of" in r.output
    with library.ctx.db.session() as s:
        assert s.scalar(select(Person.id).where(Person.name == "Added later")) is None


# --------------------------------------------------------------------------- import
def test_import_replaces_the_library_keeps_a_backup_and_clears_caches(library, settings):
    c, db = library, library.ctx.db
    data = exported(c)
    before = snapshot(db)
    # changes after the export, and state that must not survive the import
    with db.write() as s:
        s.add(Person(name="Zed"))
        s.merge(AppSetting(key=transfer.LIBRARY_ID_KEY, value="old"))
    with db.session() as s:
        face_id = s.scalar(select(Face.id))
    ok(c.get(f"/media/face/{face_id}"))
    ok(c.get("/media/thumb/1"))
    aligned = settings.cache_dir / "aligned" / "0000" / "1_512_v1.webp"
    aligned.parent.mkdir(parents=True)
    aligned.write_bytes(b"x")
    assert any(settings.thumbs_dir.rglob("*.webp")) and any(settings.face_crops_dir.rglob("*"))
    animal_crop = settings.animal_crops_dir / "0" / "1.jpg"  # keyed by animal id, too
    animal_crop.parent.mkdir(parents=True, exist_ok=True)
    animal_crop.write_bytes(b"jpeg")
    library.ctx.search  # noqa: B018 - creates the search engine (in-memory index)
    assert library.ctx._search is not None

    r = import_file(c, data, follow_redirects=False)
    assert r.status_code == 303 and "/database?msg=Importing" in r.headers["location"]
    wait_jobs(library.ctx)
    job = library.ctx.jobs.history[0]
    assert job.name == "import" and job.status == "done", job.message
    assert f"Database imported: {N_PHOTOS} photos, 0 of them" in str(job.message)

    assert snapshot(db) == before  # same ids, faces and properties
    with db.session() as s:
        assert s.scalar(select(Person.id).where(Person.name == "Zed")) is None
        assert s.scalar(select(Person.id).where(Person.name == "Anna")) is not None
        new_id = s.get(AppSetting, transfer.LIBRARY_ID_KEY).value
    assert new_id not in (None, "old", "0")
    assert not missing_ids(db)  # the photos are where they were
    backups = list(settings.data_dir.glob("vision_tagger.sqlite3.bak-import-*"))
    assert len(backups) == 1
    con = sqlite3.connect(backups[0])
    assert con.execute("SELECT count(*) FROM persons WHERE name = 'Zed'").fetchone()[0] == 1
    con.close()
    # caches keyed by ids are gone (made again on demand)
    assert not [p for p in settings.thumbs_dir.rglob("*") if p.is_file()]
    assert not [p for p in settings.face_crops_dir.rglob("*") if p.is_file()]
    assert not animal_crop.exists()
    assert not aligned.exists()
    assert library.ctx._search is None
    assert not list(transfer.transfer_dir(settings).iterdir())  # temp file removed
    # the live database stays in WAL mode and the app keeps working
    con = sqlite3.connect(settings.db_path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()
    # the first page view afterwards tells the browser to drop its cached thumbnails, once
    r = ok(c.get("/images"))
    assert r.headers.get("clear-site-data") == '"cache"'
    assert c.cookies.get("vt_lib") == new_id
    assert "clear-site-data" not in ok(c.get("/database")).headers
    ok(c.get("/media/thumb/1"))
    sw = ok(c.get("/sw.js")).text
    assert f'const LIBRARY = "{new_id}"' in sw


def test_import_resets_pipeline_state_of_the_other_computer(library, settings, tmp_path):
    db = library.ctx.db
    with db.write() as s:  # as if exported while a run was active elsewhere
        run = Run(stages="faces", batch_size=4, status="running", heartbeat_at=utcnow())
        s.add(run)
        s.flush()
        ctrl = s.get(Control, 1)
        ctrl.desired_state, ctrl.active_run_id, ctrl.heartbeat_at = "running", run.id, utcnow()
        ctrl.active_stage = "faces"
        s.query(StageStatus).filter(StageStatus.stage == "vlm").update({"status": "running"})
    file = tmp_path / "running.sqlite3"
    transfer.export_database(db, file)
    with db.write() as s:  # this computer: idle
        ctrl = s.get(Control, 1)
        ctrl.desired_state, ctrl.active_run_id, ctrl.heartbeat_at = "stopped", None, None
    ok(import_file(library, file.read_bytes()))
    wait_jobs(library.ctx)
    assert library.ctx.jobs.history[0].status == "done", library.ctx.jobs.history[0].message
    with db.session() as s:
        ctrl = s.get(Control, 1)
        assert ctrl.desired_state == "stopped" and ctrl.active_run_id is None
        assert ctrl.heartbeat_at is None
        assert not s.scalar(
            select(func.count()).select_from(StageStatus).where(StageStatus.status == "running")
        )
        assert s.scalar(select(Run.status).where(Run.id == run.id)) == "interrupted"
    assert "Pipeline idle" in ok(library.get("/partials/activity")).text


def test_import_is_refused_while_busy(library):
    c, ctx = library, library.ctx
    data = exported(c)
    ctx.jobs.current = Job(name="index")
    page = ok(import_file(c, data)).text
    assert "Another job is running. Wait until it has finished." in page
    assert "disabled" in ok(c.get("/database")).text
    ctx.jobs.current = None
    with ctx.db.write() as s:  # a pipeline runs in another process (heartbeat)
        run = Run(stages="faces", batch_size=4, status="running", heartbeat_at=utcnow())
        s.add(run)
        s.flush()
        ctrl = s.get(Control, 1)
        ctrl.active_run_id, ctrl.heartbeat_at, ctrl.desired_state = run.id, utcnow(), "paused"
    page = ok(import_file(c, data)).text
    assert "Stop the pipeline first." in page
    r = import_file(c, data, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403  # the usual protection of state-changing requests
    assert not ctx.jobs.history  # nothing was started
    assert not list(transfer.transfer_dir(ctx.settings).iterdir())  # upload removed


def test_import_refuses_files_that_are_not_mira_libraries(library, tmp_path):
    c, db = library, library.ctx.db
    before = snapshot(db)

    def refused(data: bytes, text: str) -> None:
        page = ok(import_file(c, data)).text
        assert text in page, page[:3000]
        assert not c.ctx.jobs.busy and snapshot(db) == before

    refused(b"", "Choose a database file first.")
    refused(b"hello, not a database" * 20, "This is not a database file.")
    foreign = tmp_path / "foreign.sqlite3"
    con = sqlite3.connect(foreign)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT)")
    con.commit()
    con.close()
    refused(foreign.read_bytes(), "This database is not a Mira library")
    newer = tmp_path / "newer.sqlite3"
    transfer.export_database(db, newer)
    con = sqlite3.connect(newer)
    con.execute("UPDATE alembic_version SET version_num = '9999'")
    con.commit()
    con.close()
    refused(newer.read_bytes(), "comes from a newer version of Mira (schema 9999")
    tricky = tmp_path / "trigger.sqlite3"
    transfer.export_database(db, tricky)
    con = sqlite3.connect(tricky)
    con.execute("CREATE TRIGGER wipe AFTER UPDATE ON images BEGIN DELETE FROM faces; END")
    con.commit()
    con.close()
    refused(tricky.read_bytes(), "triggers or views")
    damaged = bytearray(newer.read_bytes())
    damaged[100:] = b"\xff" * (len(damaged) - 100)  # header intact, pages destroyed
    refused(bytes(damaged), "The database file is damaged")
    assert not list(transfer.transfer_dir(c.ctx.settings).iterdir())


@pytest.mark.parametrize("versioned", [True, False])
def test_import_migrates_an_older_database(library, tmp_path, versioned):
    old = tmp_path / "old.sqlite3"
    build_at(old, "0001", keep_version_table=versioned)
    fill_first_version(old)
    ok(import_file(library, old.read_bytes()))
    wait_jobs(library.ctx)
    job = library.ctx.jobs.history[0]
    assert job.status == "done", job.message
    db = library.ctx.db
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(Image)) == 1
        assert s.scalar(select(Person.name)) == "Anna"
        assert s.scalar(select(Face.assignment)) == "manual"
    con = sqlite3.connect(library.ctx.settings.db_path)
    assert con.execute("SELECT version_num FROM alembic_version").fetchone()[0] == head_revision()
    con.close()
    assert missing_ids(db) == {1}  # /p/a.jpg is not on this computer
    page = ok(library.get("/database/missing")).text
    assert "a.jpg" in page and "/p" in page and "1 face · Anna" in page


def test_pipeline_cannot_start_during_an_import(library):
    library.ctx.jobs.current = Job(name="import")
    page = ok(library.post("/actions/run", data={"stages": ["faces"]})).text
    assert "Wait until the database import has finished." in page
    assert not library.ctx.runner.running
    library.ctx.jobs.current = None


def test_status_card_follows_the_job_and_reloads_when_it_ends(library):
    c, ctx = library, library.ctx
    job = Job(name="import", message="Replacing the library…")
    job.progress = {"processed": 3, "candidates": 12}
    ctx.jobs.current = job
    page = ok(c.get("/database")).text
    assert 'hx-get="/partials/database-status?watch=1"' in page and "width: 25.0%" in page
    assert "Database import" in page and "Replacing the library…" in page
    assert ok(c.get("/partials/database-status?watch=1")).text.count("watch=1") == 1
    job.status, job.message = "done", "Database imported: 5 photos, 0 of them not found."
    r = c.get(
        "/partials/database-status?watch=1",
        headers={"HX-Request": "true", "Referer": "http://testserver/database?msg=Importing"},
    )
    assert r.status_code == 204 and r.headers["hx-redirect"] == "/database"  # without the msg
    page = ok(c.get("/database")).text
    assert "watch=1" not in page and "Database imported: 5 photos" in page
    job.status, job.message = "failed", "The database file is damaged: x"
    assert "failed: The database file is damaged" in ok(c.get("/database/missing")).text
    # once another job ran, the old result is not shown any more (it may be out of date)
    ctx.jobs.current = Job(name="index", status="done", message="Index: 0 new, 5 moved")
    assert "Database import" not in ok(c.get("/database/missing")).text
    ctx.jobs.current = None


# --------------------------------------------------------------------------- missing files
def test_location_check_covers_images_outside_the_current_roots(library, settings, photos):
    c, db = library, library.ctx.db
    with db.write() as s:  # the library folder is no longer configured
        s.merge(AppSetting(key=ROOTS_KEY, value=[]))
    assert stored_roots(db, settings) == []
    gone = photos / "trip" / "IMG_20210101_120000.jpg"
    gone.unlink()
    r = c.post("/database/check", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/database/missing")
    wait_jobs(library.ctx)
    job = library.ctx.jobs.history[0]
    assert job.name == "locations" and job.status == "done"
    assert str(job.message) == f"Location check: {N_PHOTOS - 1} photos found, 1 missing."
    with db.session() as s:
        missing = s.scalars(select(Image.filename).where(Image.missing)).all()
    assert missing == [gone.name]
    # gallery, search and counts leave missing photos out, as before
    assert ok(c.get("/images")).text.count('class="thumb"') == N_PHOTOS - 1
    page = ok(c.get("/database/missing")).text
    assert gone.name in page and "1 photo" in page and "Check again" in page
    assert "Add or scan the folder where the photos are now" in page
    page = ok(c.get("/database")).text
    assert "1 missing file" in page and 'href="/database/missing"' in page
    # a missing photo without a cached thumbnail shows a placeholder, not a broken image
    image_id = next(iter(missing_ids(db)))
    for p in settings.thumbs_dir.rglob(f"{image_id}_*.webp"):
        p.unlink()
    r = c.get(f"/media/thumb/{image_id}")
    assert r.status_code == 404 and r.headers["content-type"].startswith("image/svg+xml")
    assert r.headers["cache-control"] == "no-store"
    assert f'<img src="/media/thumb/{image_id}" alt=' in ok(c.get(f"/images/{image_id}")).text
    # the file is back: "Check again" clears the flag
    make_image(gone, seed=0)
    c.post("/database/check")
    wait_jobs(library.ctx)
    assert not missing_ids(db)
    assert "No missing files" in ok(c.get("/database/missing")).text


def test_missing_page_is_paginated(library, monkeypatch):
    from vision_tagger.web.routes import database

    monkeypatch.setattr(database, "PER_PAGE", 2)
    with library.ctx.db.write() as s:
        s.query(Image).update({"missing": True})
    page = ok(library.get("/database/missing")).text
    assert f"{N_PHOTOS} photos" in page and "Page 1 of 3" in page
    assert len(re.findall(r'<td class="missing-thumb">', page)) == 2
    page = ok(library.get("/database/missing?page=3")).text
    assert len(re.findall(r'<td class="missing-thumb">', page)) == 1
    assert "Page 3 of 3" in ok(library.get("/database/missing?page=99")).text


def test_check_locations_looks_at_a_missing_folder_once(db, settings, monkeypatch):
    from vision_tagger import indexer

    from .helpers import add_image

    with db.write() as s:
        for n in range(30):
            img = add_image(s, n, path=f"/gone/library/sub{n % 3}/img{n}.jpg")
            img.root = "/gone/library"
        add_image(s, 99, path="/elsewhere/x.jpg").root = "/elsewhere/other"
    calls: list[str] = []
    real_isdir, real_exists = indexer.os.path.isdir, indexer.os.path.exists
    monkeypatch.setattr(indexer.os.path, "isdir", lambda p: calls.append(p) or real_isdir(p))
    monkeypatch.setattr(indexer.os.path, "exists", lambda p: calls.append(p) or real_exists(p))
    assert Indexer(db, settings).check_locations() == 31
    seen = [p for p in calls if p.startswith(("/gone", "/elsewhere"))]
    # one look at each root (and one at the folder of the file outside its root), instead of
    # one per file: a missing network drive would make each of them wait
    assert seen == ["/gone/library", "/elsewhere/other", "/elsewhere"]


# --------------------------------------------------------------------------- re-linking
def test_moved_photos_are_relinked_by_perceptual_hash(library, settings, photos, tmp_path):
    """Another computer: the photos are in another folder. A scan of it re-links them."""
    c, db = library, library.ctx.db
    data = exported(c)
    before = snapshot(db)
    new_home = tmp_path / "elsewhere" / "Pictures"
    new_home.parent.mkdir()
    shutil.move(photos, new_home)
    # a fresh data folder: this is the other computer
    other = Settings(
        paths={"data_dir": tmp_path / "other-data", "image_roots": []},
        vlm={"backend": "fake"},
        index={"workers": 2, "commit_every": 3},
        web={"allowed_hosts": ["testserver"]},
    )
    app = create_app(other)
    with TestClient(app) as c2:
        ctx = app.state.ctx
        ok(import_file(c2, data))
        wait_jobs(ctx)
        assert ctx.jobs.history[0].status == "done", ctx.jobs.history[0].message
        assert f"{N_PHOTOS} of them not found" in str(ctx.jobs.history[0].message)
        assert missing_ids(ctx.db) == set(before)
        assert stored_roots(ctx.db, other) == []  # the old folder does not exist here
        page = ok(c2.get("/database/missing")).text
        assert f"{N_PHOTOS} photos" in page and "IMG_20210100_120000.jpg" in page
        assert "Anna" in page
        # placeholders, no broken images, until the photos are found again
        assert c2.get("/media/thumb/1").headers["content-type"].startswith("image/svg")
        ok(c2.post("/import/roots", data={"path": str(new_home)}))
        r = c2.post("/actions/index", follow_redirects=False)
        assert r.status_code == 303
        wait_jobs(ctx)
        message = str(ctx.jobs.history[0].message)
        assert f"Index: 0 new, 0 duplicates, {N_PHOTOS} moved" in message
        assert snapshot(ctx.db) == before  # same ids, faces and properties
        assert not missing_ids(ctx.db)
        with ctx.db.session() as s:
            roots = set(s.scalars(select(Image.root)))
            assert roots == {str(new_home.resolve())}
            assert s.scalar(select(Person.name).where(Person.name == "Anna"))
        assert any(other.thumbs_dir.rglob("*.webp"))  # thumbnails made again
        assert ok(c2.get("/media/thumb/1")).headers["content-type"] == "image/webp"
        assert "No missing files" in ok(c2.get("/database/missing")).text


def test_remote_upload_of_a_missing_photo_relinks_it(tmp_path):
    from .test_import import jpeg_bytes

    s = Settings(
        mode="remote",
        remote={"upload_dir": tmp_path / "library", "auto_process": False},
        paths={"data_dir": tmp_path / "data"},
        vlm={"backend": "fake"},
    )
    s.ensure_dirs()
    data = jpeg_bytes(7)
    original = s.upload_dir / "old" / "beach.jpg"
    original.parent.mkdir(parents=True)
    original.write_bytes(data)
    app = create_app(s)
    with TestClient(app) as c:
        ctx = app.state.ctx
        ctx.ingest.quiet_seconds = 0.2
        Indexer(ctx.db, s).run()
        with ctx.db.write() as w:
            image_id = w.scalar(select(Image.id))
            w.add(ImageProperty(image_id=image_id, key="location_type", value="beach"))
        original.unlink()
        Indexer(ctx.db, s).check_locations()
        assert missing_ids(ctx.db) == {image_id}
        assert "Upload the photos again" in ok(c.get("/database/missing")).text
        up = c.post("/api/uploads", json={"name": "beach.jpg", "size": len(data)}).json()
        c.put(f"/api/uploads/{up['id']}?offset=0", content=data)
        assert c.post(f"/api/uploads/{up['id']}/complete").json()["state"] == "done"
        deadline = time.time() + 20
        while missing_ids(ctx.db) and time.time() < deadline:
            time.sleep(0.1)
        wait_jobs(ctx)
        assert not missing_ids(ctx.db)
        with ctx.db.session() as r:
            assert r.scalar(select(func.count()).select_from(Image)) == 1  # not added twice
            img = r.get(Image, image_id)
            assert Path(img.path).parent.name == date.today().isoformat()
            assert r.scalar(select(ImageProperty.value).where(ImageProperty.key == "location_type"))
        assert "1 missing photos found again" in str(ctx.jobs.history[0].message)


def test_database_page_renders_in_german(library):
    de = {"Accept-Language": "de"}
    page = ok(library.get("/database", headers=de)).text
    assert '<html lang="de"' in page and "Datenbank" in page
    with library.ctx.db.write() as s:
        s.query(Image).update({"missing": True})
    page = ok(library.get("/database/missing", headers=de)).text
    assert "Fehlende Dateien" in page
