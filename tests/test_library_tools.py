"""Library tools on the Database page: clearing the library, removing photos whose file is
missing, and showing files and folders in Explorer / Finder / the file manager."""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from typer.testing import CliRunner

from vision_tagger import removal, reveal
from vision_tagger.config import Settings
from vision_tagger.db import transfer
from vision_tagger.db.models import (
    Animal,
    AppSetting,
    Cluster,
    Collection,
    Control,
    DuplicateFile,
    Face,
    Image,
    ImageEmbedding,
    ImageProperty,
    Person,
    PersonReference,
    Pet,
    Place,
    PropertyAlias,
    Run,
    StageStatus,
    VisualEmbedding,
    utcnow,
)
from vision_tagger.faces import clustering as C
from vision_tagger.faces.vectors import to_blob
from vision_tagger.imaging import face_crop_path, thumbnail_path
from vision_tagger.indexer import ROOTS_KEY, Indexer, stored_roots
from vision_tagger.web.app import create_app
from vision_tagger.web.context import Job

from .conftest import make_image
from .helpers import add_animal, add_face, add_image, animal_centers, person_centers
from .test_transfer import analyse, wait_jobs

N_PHOTOS = 5
LOCAL = {"base_url": "http://127.0.0.1:8765", "client": ("127.0.0.1", 50123)}


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def count(db, model, *where) -> int:
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where)) or 0


@pytest.fixture
def desktop(monkeypatch):
    """A graphical session (Linux CI has none): the file manager buttons may appear."""
    monkeypatch.setenv("DISPLAY", ":0")


@pytest.fixture
def launched(desktop):
    """The file manager calls of the app, instead of starting a program."""
    calls: list[reveal.Launch] = []
    yield calls


@contextlib.contextmanager
def _library(settings, photos, launched=None, **client_kw):
    for i in range(N_PHOTOS):
        make_image(photos / "trip" / f"IMG_2021010{i}_120000.jpg", seed=i)
    app = create_app(settings)
    with TestClient(app, **client_kw) as client:
        ctx = app.state.ctx
        if launched is not None:
            ctx.file_manager.launcher = launched.append
            # one platform's commands on every CI runner (plan() is tested per platform)
            ctx.file_manager.platform = "linux"
        Indexer(ctx.db, settings).run()
        analyse(ctx, settings)
        with ctx.db.write() as s:
            cid = s.scalar(select(Face.cluster_id).where(Face.cluster_id.is_not(None)))
            C.name_cluster(s, cid, "Anna")
        client.ctx = ctx
        yield client


@pytest.fixture
def library(settings, photos):
    """An analysed library (fake models) with a named person, seen from another computer."""
    with _library(settings, photos) as client:
        yield client


@pytest.fixture
def local(settings, photos, launched):
    """The same library, seen from a browser on the computer Mira runs on."""
    with _library(settings, photos, launched, **LOCAL) as client:
        yield client


def lose(client, photos, *names: str) -> list[int]:
    """Delete photo files and let the location check flag them; returns their ids."""
    for name in names:
        (photos / "trip" / name).unlink()
    Indexer(client.ctx.db, client.ctx.settings).check_locations()
    with client.ctx.db.session() as s:
        return list(s.scalars(select(Image.id).where(Image.missing).order_by(Image.id)))


def pipeline_alive(db) -> None:
    """A pipeline that runs in another process (it sends heartbeats)."""
    with db.write() as s:
        run = Run(stages="faces", batch_size=4, status="running", heartbeat_at=utcnow())
        s.add(run)
        s.flush()
        ctrl = s.get(Control, 1)
        ctrl.active_run_id, ctrl.heartbeat_at, ctrl.desired_state = run.id, utcnow(), "running"


def pipeline_stopped(db) -> None:
    with db.write() as s:
        ctrl = s.get(Control, 1)
        ctrl.active_run_id, ctrl.heartbeat_at, ctrl.desired_state = None, None, "stopped"


# =========================================================================== remove missing
@pytest.fixture
def missing_library(db, settings, tmp_path):
    """Photos (most of them missing) with faces, persons, pets, places and cached pictures.

    img1 missing: Anna's face (cluster 1, also on img2), the named pet Rex, place "Old town"
    img2 present: Anna's face, a dog of an unnamed group (also on img4), place "Old town"
    img3 missing: Ben's only face (with a reference cut from it), place "Harbour" (only photo)
    img4 missing: a face of an unnamed cluster, an unnamed dog, the place the user named
    img5 missing in the database, but the file is there again
    img6 missing, but a copy of it (duplicate file) exists
    """
    present = tmp_path / "present.jpg"
    make_image(present, seed=1)
    back = tmp_path / "back.jpg"
    make_image(back, seed=2)
    copy = tmp_path / "copy.jpg"
    make_image(copy, seed=3)
    centers = person_centers(3)
    dogs = animal_centers(3)
    with db.write() as s:
        anna, ben = Person(name="Anna"), Person(name="Ben")
        s.add_all([anna, ben])
        s.flush()
        c_anna = Cluster(person_id=anna.id, locked=True)
        c_ben = Cluster(person_id=ben.id, locked=True)
        c_free = Cluster()
        old_town = Place(name="Old town", lat=1.0, lon=1.0)
        harbour = Place(name="Harbour", lat=2.0, lon=2.0)
        mine = Place(name="Our garden", lat=3.0, lon=3.0, user_named=True, source="user")
        rex = Pet(name="Rex", species="dog", locked=True)
        loose = Pet(species="dog")
        shared = Pet(species="dog")
        s.add_all([c_anna, c_ben, c_free, old_town, harbour, mine, rex, loose, shared])
        s.flush()
        imgs = {}
        for n, (path, missing, place) in {
            1: ("/gone/img1.jpg", True, old_town),
            2: (str(present), False, old_town),
            3: ("/gone/img3.jpg", True, harbour),
            4: ("/gone/img4.jpg", True, mine),
            5: (str(back), True, None),
            6: ("/gone/img6.jpg", True, None),
        }.items():
            imgs[n] = add_image(
                s, n, path=path, missing=missing, place_id=place.id if place else None
            )
        faces = {
            1: add_face(s, imgs[1].id, centers[0], cluster_id=c_anna.id),
            2: add_face(s, imgs[2].id, centers[0], cluster_id=c_anna.id),
            3: add_face(s, imgs[3].id, centers[1], cluster_id=c_ben.id),
            4: add_face(s, imgs[4].id, centers[2], cluster_id=c_free.id),
        }
        s.add(PersonReference(person_id=ben.id, face_id=faces[3].id, embedding=to_blob(centers[1])))
        add_animal(s, imgs[1].id, dogs[0], pet_id=rex.id)
        add_animal(s, imgs[4].id, dogs[1], pet_id=loose.id)
        add_animal(s, imgs[2].id, dogs[2], pet_id=shared.id)
        add_animal(s, imgs[4].id, dogs[2], pet_id=shared.id)
        s.add(ImageProperty(image_id=imgs[1].id, key="location_type", value="beach"))
        s.add(ImageEmbedding(image_id=imgs[1].id, model="m", dim=2, vector=b"\0" * 4, doc_hash="x"))
        s.add(
            VisualEmbedding(
                image_id=imgs[1].id, model="v", stage_version="1", dim=2, vector=b"\0" * 4
            )
        )
        s.add(DuplicateFile(image_id=imgs[6].id, path=str(copy), path_key=str(copy)))
        s.add(DuplicateFile(image_id=imgs[1].id, path="/gone/dup1.jpg", path_key="/gone/dup1"))
        s.flush()
        C.update_cluster_stats(s)
        from vision_tagger.animals.grouping import update_pet_stats

        update_pet_stats(s)
        ids = {
            "img": {n: i.id for n, i in imgs.items()},
            "face": {n: f.id for n, f in faces.items()},
            "cluster": {"anna": c_anna.id, "ben": c_ben.id, "free": c_free.id},
            "pet": {"rex": rex.id, "loose": loose.id, "shared": shared.id},
            "place": {"old": old_town.id, "harbour": harbour.id, "mine": mine.id},
            "person": {"anna": anna.id, "ben": ben.id},
        }
    with db.session() as s:
        ids["animal"] = {
            image_id: [a.id for a in s.scalars(select(Animal).where(Animal.image_id == image_id))]
            for image_id in ids["img"].values()
        }
    # cached pictures, keyed by ids
    for n in (1, 2):
        for size in (256, 512):
            path = thumbnail_path(settings.thumbs_dir, ids["img"][n], size)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"webp")
    for n in (1, 2):
        path = face_crop_path(settings.face_crops_dir, ids["face"][n])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"webp")
    aligned = settings.cache_dir / "aligned" / "0000" / f"{ids['face'][1]}_512_v1.webp"
    aligned.parent.mkdir(parents=True, exist_ok=True)
    aligned.write_bytes(b"webp")
    for animal_id in ids["animal"][ids["img"][1]] + ids["animal"][ids["img"][2]]:
        path = face_crop_path(settings.animal_crops_dir, animal_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"webp")
    return ids


def test_remove_missing_deletes_photos_and_cleans_up(db, settings, missing_library):
    ids = missing_library
    img = ids["img"]
    result = removal.remove_missing(db, settings)
    assert result.removed == 3 and sorted(result.removed_ids) == [img[1], img[3], img[4]]
    assert result.found == 2  # the file of img5 and the copy of img6 are there
    with db.session() as s:
        assert set(s.scalars(select(Image.id))) == {img[2], img[5], img[6]}
        assert s.get(Image, img[5]).missing is False  # found again
        assert s.get(Image, img[6]).missing is True  # the location check re-links it
        # cascades: faces, animals, properties, search index, duplicates, stage status
        assert set(s.scalars(select(Face.image_id))) == {img[2]}
        assert set(s.scalars(select(Animal.image_id))) == {img[2]}
        assert not s.scalar(select(func.count()).select_from(ImageProperty))
        assert not s.scalar(select(func.count()).select_from(ImageEmbedding))
        assert not s.scalar(select(func.count()).select_from(VisualEmbedding))
        assert list(s.scalars(select(DuplicateFile.image_id))) == [img[6]]
        assert set(s.scalars(select(StageStatus.image_id))) == {img[2], img[5], img[6]}
        # clusters: Anna's keeps its photo of img2, Ben's and the unnamed one are gone
        anna = s.get(Cluster, ids["cluster"]["anna"])
        assert anna.face_count == 1 and anna.cover_face_id == ids["face"][2]
        assert s.get(Cluster, ids["cluster"]["ben"]) is None
        assert s.get(Cluster, ids["cluster"]["free"]) is None
        # named persons stay, with their references (the picture's face id is dropped)
        assert set(s.scalars(select(Person.name))) == {"Anna", "Ben"}
        ref = s.scalar(select(PersonReference))
        assert ref.person_id == ids["person"]["ben"] and ref.face_id is None
        # pets: the named one stays (no photos left), the empty unnamed one goes
        rex = s.get(Pet, ids["pet"]["rex"])
        assert rex.name == "Rex" and rex.animal_count == 0 and rex.cover_animal_id is None
        assert s.get(Pet, ids["pet"]["loose"]) is None
        assert s.get(Pet, ids["pet"]["shared"]).animal_count == 1
        # places: without photos they go, unless the user named them
        assert s.get(Place, ids["place"]["old"]) is not None
        assert s.get(Place, ids["place"]["harbour"]) is None
        assert s.get(Place, ids["place"]["mine"]) is not None
    assert (result.clusters_removed, result.pets_removed, result.places_removed) == (2, 1, 1)
    # cached pictures of the removed ids are gone, the others stay
    assert not list(settings.thumbs_dir.rglob(f"{img[1]}_*.webp"))
    assert len(list(settings.thumbs_dir.rglob(f"{img[2]}_*.webp"))) == 2
    assert not face_crop_path(settings.face_crops_dir, ids["face"][1]).exists()
    assert face_crop_path(settings.face_crops_dir, ids["face"][2]).exists()
    assert not list((settings.cache_dir / "aligned").rglob("*.webp"))
    for animal_id in ids["animal"][img[1]]:
        assert not face_crop_path(settings.animal_crops_dir, animal_id).exists()
    for animal_id in ids["animal"][img[2]]:
        assert face_crop_path(settings.animal_crops_dir, animal_id).exists()


def test_only_missing_photos_are_removed(db, settings, missing_library):
    img = missing_library["img"]
    result = removal.remove_missing(db, settings, [img[2], 999])
    assert result.removed == 0 and result.unknown == 2
    assert count(db, Image) == 6


def test_removing_the_newest_photo_renews_the_library_id(db, settings):
    with db.write() as s:
        for n in range(1, 4):
            add_image(s, n, path=f"/gone/{n}.jpg", missing=True)
    assert transfer.library_id(db) == "0"
    result = removal.remove_missing(db, settings, [1])  # not the newest: no id comes back
    assert result.removed == 1 and not result.new_library_id
    assert transfer.library_id(db) == "0"
    result = removal.remove_missing(db, settings, [3])  # the next photo would get id 3
    assert result.new_library_id and transfer.library_id(db) not in ("0", "")


def test_removal_stops_when_a_pipeline_starts(db, settings, missing_library, monkeypatch):
    monkeypatch.setattr(removal, "CHUNK", 1)
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 1:
            raise transfer.TransferError(transfer.busy_reason(db, runner_running=True))

    with pytest.raises(transfer.TransferError):
        removal.remove_missing(db, settings, check=check)
    assert count(db, Image) == 5  # only the first chunk was removed


# --------------------------------------------------------------------------- remove: web
def test_remove_one_missing_photo(library, photos):
    c, db = library, library.ctx.db
    first, second = lose(c, photos, "IMG_20210100_120000.jpg", "IMG_20210101_120000.jpg")
    page = ok(c.get("/database/missing")).text
    assert page.count('action="/database/missing/remove"') == 2
    assert 'action="/database/missing/remove-all"' in page
    assert "Remove all missing photos (2)" in page
    assert 'data-confirm="Remove all 2 missing photos from Mira?' in page
    assert 'name="expected" value="2"' in page
    assert "No file is deleted" in page
    # from the Missing files page
    r = c.post(
        "/database/missing/remove",
        data={"id": first},
        headers={"Referer": "http://testserver/database/missing?page=1"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/database/missing?page=1")
    assert "was removed from the library" in ok(c.get(r.headers["location"])).text
    with db.session() as s:
        assert s.get(Image, first) is None
    # from the photo's own page, which is gone afterwards
    page = ok(c.get(f"/images/{second}")).text
    assert "Remove from library" in page and "The file of this photo is missing" in page
    r = c.post(
        "/database/missing/remove",
        data={"id": second},
        headers={"Referer": f"http://testserver/images/{second}?q=1"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/database/missing?msg=")
    assert count(db, Image) == N_PHOTOS - 2
    assert "No missing files" in ok(c.get("/database/missing")).text
    job = c.ctx.jobs.history[0]
    assert job.name == "remove-missing" and "Missing photos removed: 1" in str(job.message)
    # a photo that is not missing cannot be removed this way
    with db.session() as s:
        present = s.scalar(select(Image.id))
    page = ok(c.post("/database/missing/remove", data={"id": present})).text
    assert "is not missing. Only missing photos can be removed here." in page
    assert "Remove from library" not in ok(c.get(f"/images/{present}")).text
    assert count(db, Image) == N_PHOTOS - 2


def test_a_photo_whose_file_is_back_is_not_removed(library, photos):
    c, db = library, library.ctx.db
    (image_id,) = lose(c, photos, "IMG_20210102_120000.jpg")
    make_image(photos / "trip" / "IMG_20210102_120000.jpg", seed=2)  # the drive is back
    page = ok(c.post("/database/missing/remove", data={"id": image_id})).text
    assert "was not removed: its file is there again." in page
    with db.session() as s:
        assert s.get(Image, image_id).missing is False


def test_remove_all_missing_photos(library, photos):
    c, db = library, library.ctx.db
    names = [f"IMG_2021010{i}_120000.jpg" for i in range(3)]
    gone = lose(c, photos, *names)
    with db.session() as s:
        anna = s.scalar(select(Person.id).where(Person.name == "Anna"))
        back = s.scalar(select(Image.id).where(Image.filename == names[2]))
    make_image(photos / "trip" / names[2], seed=2)  # one file came back meanwhile
    r = c.post("/database/missing/remove-all", data={"expected": 3}, follow_redirects=False)
    assert r.status_code == 303 and "Removing+3+missing+photos" in r.headers["location"]
    wait_jobs(c.ctx)
    job = c.ctx.jobs.history[0]
    assert job.name == "remove-missing" and job.status == "done", job.message
    assert str(job.message) == (
        "Missing photos removed: 2; skipped because their file is there again: 1"
    )
    with db.session() as s:
        left = set(s.scalars(select(Image.id)))
        assert left & set(gone) == {back} and len(left) == N_PHOTOS - 2
        assert not s.scalar(select(func.count()).select_from(Image).where(Image.missing))
        assert s.get(Person, anna).name == "Anna"  # named persons stay
        # no cluster without faces, no face without a photo
        for cluster in s.scalars(select(Cluster)):
            assert s.scalar(
                select(func.count()).select_from(Face).where(Face.cluster_id == cluster.id)
            )
    page = ok(c.get("/database/missing")).text
    assert "Removing missing photos" in page and "Missing photos removed: 2" in page
    ok(c.get("/images"))
    ok(c.get("/faces"))
    ok(c.get(f"/persons/{anna}"))


def test_remove_is_refused_while_busy_and_protected(library, photos):
    c, ctx = library, library.ctx
    (image_id,) = lose(c, photos, "IMG_20210103_120000.jpg")
    # the confirmed number must still be the number of missing photos
    page = ok(c.post("/database/missing/remove-all", data={"expected": 7})).text
    assert "The list of missing files has changed." in page
    # a job runs
    ctx.jobs.current = Job(name="index")
    page = ok(c.post("/database/missing/remove", data={"id": image_id})).text
    assert "Another job is running. Wait until it has finished." in page
    page = ok(c.post("/database/missing/remove-all", data={"expected": 1})).text
    assert "Another job is running. Wait until it has finished." in page
    ctx.jobs.current = None
    # a pipeline runs (here: in another process)
    pipeline_alive(ctx.db)
    page = ok(c.get("/database/missing")).text
    assert "Stop the pipeline first." in page and re.search(
        r'<button class="small danger" disabled', page
    )
    assert "Stop the pipeline first." in ok(c.get(f"/images/{image_id}")).text
    page = ok(c.post("/database/missing/remove", data={"id": image_id})).text
    assert "Stop the pipeline first." in page
    page = ok(c.post("/database/missing/remove-all", data={"expected": 1})).text
    assert "Stop the pipeline first." in page
    pipeline_stopped(ctx.db)
    # state-changing requests: POST only, from this UI only
    assert c.get("/database/missing/remove").status_code == 405
    r = c.post(
        "/database/missing/remove", data={"id": image_id}, headers={"Origin": "http://evil.example"}
    )
    assert r.status_code == 403
    r = c.post(
        "/database/missing/remove-all",
        data={"expected": 1},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403
    assert count(ctx.db, Image) == N_PHOTOS


def test_pipeline_cannot_start_while_photos_are_removed(library):
    library.ctx.jobs.current = Job(name="remove-missing")
    page = ok(library.post("/actions/run", data={"stages": ["faces"]})).text
    assert "Wait until “Removing missing photos” has finished." in page
    assert not library.ctx.runner.running
    library.ctx.jobs.current = None


# =========================================================================== clear
def fill_extras(db) -> None:
    """Data that is not hanging off photos: collections, aliases, places, runs."""
    with db.write() as s:
        s.add(Collection(name="Holidays", query="t=event_type:vacation"))
        s.add(PropertyAlias(key="objects", from_value="auto", to_value="car"))
        s.add(Place(name="Somewhere", lat=1.0, lon=2.0, user_named=True, source="user"))
        s.add(Pet(name="Rex", species="dog", locked=True))
        s.add(Run(stages="faces", batch_size=4, status="completed"))


def test_clear_needs_both_confirmations(library):
    c, db = library, library.ctx.db
    page = ok(c.get("/database")).text
    assert "Danger zone" in page and 'href="/database/clear"' in page
    page = ok(c.get("/database/clear")).text
    assert "Step 1 of 2" in page and 'name="understood" value="yes" required' in page
    assert "Your photo files: Mira never deletes, moves or changes them." in page
    assert "vision_tagger.sqlite3.bak-clear-…" in page and str(c.ctx.settings.data_dir) in page
    # without the ticked box: back to step 1
    page = ok(c.get("/database/clear?step=2")).text
    assert "Step 1 of 2" in page and "Tick the box to confirm" in page
    page = ok(c.get("/database/clear?step=2&understood=yes")).text
    assert "Step 2 of 2" in page and f'data-expected="{N_PHOTOS}"' in page
    assert f'in the library: <b class="clear-expected">{N_PHOTOS}</b>' in page
    # the final request checks both again
    r = c.post("/database/clear", data={"photos": str(N_PHOTOS)}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/database/clear?err=")
    for typed in ("", "4", "DELETE", f"{N_PHOTOS}0"):
        page = ok(c.post("/database/clear", data={"understood": "yes", "photos": typed})).text
        assert "The number does not match." in page and "Step 2 of 2" in page
    assert not c.ctx.jobs.history and count(db, Image) == N_PHOTOS
    # GET does nothing, other sites cannot post
    r = c.post(
        "/database/clear",
        data={"understood": "yes", "photos": str(N_PHOTOS)},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403
    assert count(db, Image) == N_PHOTOS


def test_clear_is_refused_while_busy(library):
    c, ctx = library, library.ctx
    data = {"understood": "yes", "photos": str(N_PHOTOS)}
    ctx.jobs.current = Job(name="index")
    page = ok(c.get("/database/clear")).text
    assert "Another job is running." in page and re.search(
        r"<button class=\"danger\" disabled", page
    )
    page = ok(c.post("/database/clear", data=data)).text
    assert "Another job is running. Wait until it has finished." in page
    ctx.jobs.current = None
    pipeline_alive(ctx.db)
    assert "Stop the pipeline first." in ok(c.get("/database")).text
    page = ok(c.post("/database/clear", data=data)).text
    assert "Stop the pipeline first." in page
    pipeline_stopped(ctx.db)
    assert not ctx.jobs.history and count(ctx.db, Image) == N_PHOTOS


def test_clear_empties_the_library_and_keeps_a_backup(library, settings, photos):
    c, db = library, library.ctx.db
    fill_extras(db)
    with db.write() as s:
        s.merge(AppSetting(key=ROOTS_KEY, value=[str(photos)]))
        s.merge(AppSetting(key=transfer.LIBRARY_ID_KEY, value="old"))
    before = transfer.library_contents(db)
    assert before["images"] == N_PHOTOS and before["persons"] >= 1 and before["faces"]
    assert before["collections"] == 1 and before["runs"] >= 1 and before["search"]
    page = ok(c.get("/database/clear")).text
    for value in (before["faces"], before["properties"], before["search"]):
        assert f'<td class="num">{value}</td>' in page
    ok(c.get("/media/thumb/1"))
    with db.session() as s:
        face_id = s.scalar(select(Face.id))
    ok(c.get(f"/media/face/{face_id}"))
    c.ctx.search  # noqa: B018 - creates the search engine (in-memory index)

    r = c.post(
        "/database/clear",
        data={"understood": "yes", "photos": f" {N_PHOTOS} "},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/database?msg=Clearing")
    wait_jobs(c.ctx)
    job = c.ctx.jobs.history[0]
    assert job.name == "clear" and job.status == "done", job.message
    backups = list(settings.data_dir.glob("vision_tagger.sqlite3.bak-clear-*"))
    assert len(backups) == 1
    assert f"Library cleared: {N_PHOTOS} photos removed from Mira" in str(job.message)
    assert str(backups[0]) in str(job.message)
    # everything is gone ...
    after = transfer.library_contents(db)
    assert set(after.values()) == {0}, after
    for model in (StageStatus, PropertyAlias, PersonReference, Animal, Pet):
        assert count(db, model) == 0, model
    # ... except the library folders; the library has a new id (browsers drop their caches)
    assert stored_roots(db, settings) == [str(photos)]
    assert transfer.library_id(db) not in ("old", "0")
    assert c.ctx._search is None
    # the backup has the old data
    con = sqlite3.connect(backups[0])
    try:
        assert con.execute("SELECT count(*) FROM images").fetchone()[0] == N_PHOTOS
        assert con.execute("SELECT count(*) FROM persons WHERE name = 'Anna'").fetchone()[0] == 1
    finally:
        con.close()
    # caches keyed by ids are gone, the photo files are not touched
    assert not [p for p in settings.thumbs_dir.rglob("*") if p.is_file()]
    assert not [p for p in settings.face_crops_dir.rglob("*") if p.is_file()]
    assert len(list((photos / "trip").iterdir())) == N_PHOTOS
    assert not list(transfer.transfer_dir(settings).iterdir())  # temporary file removed
    con = sqlite3.connect(settings.db_path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()
    # the Database page says so, with the backup
    page = ok(c.get("/database")).text
    assert "Library cleared" in page and backups[0].name in page
    # the app keeps working, and a scan builds the library again
    for url in ("/", "/images", "/faces", "/persons", "/collections", "/map", "/database/missing"):
        ok(c.get(url))
    assert "Pipeline idle" in ok(c.get("/partials/activity")).text
    r = c.post("/actions/index", follow_redirects=False)
    assert r.status_code == 303
    wait_jobs(c.ctx)
    assert f"Index: {N_PHOTOS} new" in str(c.ctx.jobs.history[0].message)
    assert count(db, Image) == N_PHOTOS
    ok(c.get("/media/thumb/1"))


def test_clear_keeps_the_configured_folders_when_none_were_saved(db, settings):
    with db.write() as s:
        add_image(s, 1)
    result = transfer.clear_library(db, settings)
    assert result.images == 1 and result.roots is None and result.backup.exists()
    with db.session() as s:
        assert s.get(AppSetting, ROOTS_KEY) is None
    assert stored_roots(db, settings) == [str(p) for p in settings.paths.image_roots]


def test_cli_db_clear(library, settings, monkeypatch):
    from vision_tagger import cli

    monkeypatch.setattr(cli, "load_settings", lambda _config=None: settings)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["db", "clear"], input="n\n")
    assert r.exit_code == 1 and "Nothing was deleted." in r.output
    assert f"{N_PHOTOS} photos" in r.output and "bak-clear-" in r.output
    r = runner.invoke(cli.app, ["db", "clear"], input="y\n4\n")
    assert r.exit_code == 1 and "The number does not match." in r.output
    r = runner.invoke(cli.app, ["db", "clear", "--yes"], input="12\n")
    assert r.exit_code == 1
    assert count(library.ctx.db, Image) == N_PHOTOS
    r = runner.invoke(cli.app, ["db", "clear"], input=f"y\n{N_PHOTOS}\n")
    assert r.exit_code == 0, r.output
    assert f"cleared: {N_PHOTOS} photos removed from Mira" in r.output and "backup:" in r.output
    assert count(library.ctx.db, Image) == 0
    # scripts: --yes twice answers both questions
    with library.ctx.db.write() as s:
        add_image(s, 1)
    r = runner.invoke(cli.app, ["db", "clear", "--yes", "--yes"])
    assert r.exit_code == 0, r.output
    assert count(library.ctx.db, Image) == 0
    pipeline_alive(library.ctx.db)
    r = runner.invoke(cli.app, ["db", "clear", "-yy"])
    assert r.exit_code == 1 and "Stop the pipeline first." in r.output


# =========================================================================== reveal
def reveal_forms(page: str) -> list[tuple[str, str]]:
    return re.findall(
        r'action="/reveal"[^>]*>\s*<input type="hidden" name="kind" value="(\w+)">'
        r'<input type="hidden" name="id" value="(\d+)">',
        page,
    )


def test_reveal_buttons_on_this_computer(local, settings, photos, launched):
    c = local
    with c.ctx.db.write() as s:
        s.merge(AppSetting(key=ROOTS_KEY, value=[str(photos), str(photos / "trip")]))
    page = ok(c.get("/images/1")).text
    assert ("image", "1") in reveal_forms(page) and "Show in folder" in page
    page = ok(c.get("/import")).text
    assert reveal_forms(page) == [("root", "0"), ("root", "1")] and "Open folder" in page
    page = ok(c.get("/database")).text
    assert reveal_forms(page) == [("database", "0"), ("data", "0")]
    assert "Show database file" in page and "Open data folder" in page
    assert 'hx-post="/reveal" hx-swap="none"' in page
    # the paths come from the database and the settings, never from the request
    with c.ctx.db.session() as s:
        image_path = s.get(Image, 1).path
    for data in (
        {"kind": "image", "id": "1", "path": "/etc/passwd"},
        {"kind": "root", "id": "1"},
        {"kind": "database", "id": "0"},
        {"kind": "data"},
    ):
        r = c.post("/reveal", data=data, headers={"HX-Request": "true"})
        assert r.status_code == 204, r.text
    assert [(a.command, a.startfile) for a in launched] == [
        (["xdg-open", str(Path(image_path).parent)], None),
        (["xdg-open", str(photos / "trip")], None),
        (["xdg-open", str(settings.db_path.parent)], None),
        (["xdg-open", str(settings.data_dir)], None),
    ]
    # without HTMX: back to the page, with a note
    r = c.post(
        "/reveal",
        data={"kind": "data"},
        headers={"Referer": "http://127.0.0.1:8765/database"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/database?msg=Opened")
    # anything else is refused
    assert c.post("/reveal", data={"kind": "path", "id": "/etc"}).status_code == 400
    assert c.post("/reveal", data={"kind": "image", "id": "../1"}).status_code == 400
    assert c.post("/reveal", data={"kind": "image", "id": "999"}).status_code == 404
    assert c.post("/reveal", data={"kind": "root", "id": "2"}).status_code == 404
    assert c.get("/reveal").status_code == 405
    r = c.post("/reveal", data={"kind": "data"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert len(launched) == 5


def test_reveal_never_contacts_an_unknown_network_share(local, launched, monkeypatch):
    c = local
    with c.ctx.db.write() as s:  # e.g. from an imported library file
        s.get(Image, 1).path = "//attacker.example/share/a.jpg"
    looked: list[str] = []
    monkeypatch.setattr(Path, "is_file", lambda self: looked.append(str(self)) or True)
    r = c.post("/reveal", data={"kind": "image", "id": "1"}, headers={"HX-Request": "true"})
    assert r.status_code == 204 and "err=" in r.headers["HX-Redirect"]
    assert looked == [] and launched == []


def test_reveal_only_on_the_computer_mira_runs_on(settings, photos, tmp_path, launched):
    make_image(photos / "a.jpg", seed=1)
    app = create_app(settings)
    with TestClient(app) as c:  # another computer (and another host name)
        c.app.state.ctx.file_manager.launcher = launched.append
        Indexer(c.app.state.ctx.db, settings).run()
        for url in ("/images/1", "/import", "/database"):
            assert "/reveal" not in ok(c.get(url)).text
        assert c.post("/reveal", data={"kind": "data"}).status_code == 403
    cases = [
        ({"client": ("192.168.1.20", 5000), "base_url": "http://127.0.0.1:8765"}, {}),
        ({"client": ("127.0.0.1", 5000), "base_url": "http://testserver"}, {}),  # a proxy's
        ({**LOCAL}, {"X-Forwarded-For": "203.0.113.9"}),
        ({**LOCAL}, {"Forwarded": "for=203.0.113.9"}),
    ]
    for kw, headers in cases:
        with TestClient(create_app(settings), **kw) as c:
            c.app.state.ctx.file_manager.launcher = launched.append
            assert "/reveal" not in ok(c.get("/database", headers=headers)).text, kw
            r = c.post("/reveal", data={"kind": "data"}, headers=headers)
            assert r.status_code == 403, (kw, headers)
    for client in (("::1", 5000), ("::ffff:127.0.0.1", 5000), ("127.0.0.2", 5000)):
        with TestClient(create_app(settings), base_url="http://[::1]:8765", client=client) as c:
            c.app.state.ctx.file_manager.launcher = launched.append
            assert "/reveal" in ok(c.get("/database")).text, client
    assert launched == []
    # remote mode: a server somewhere else, even when reached through a tunnel
    remote = Settings(
        mode="remote",
        remote={"upload_dir": tmp_path / "library", "auto_process": False},
        paths={"data_dir": tmp_path / "remote-data"},
        vlm={"backend": "fake"},
    )
    remote.ensure_dirs()
    with TestClient(create_app(remote), **LOCAL) as c:
        c.app.state.ctx.file_manager.launcher = launched.append
        assert "/reveal" not in ok(c.get("/database")).text
        assert c.post("/reveal", data={"kind": "data"}).status_code == 403
    assert launched == []


def test_no_buttons_without_a_desktop(settings, photos, monkeypatch, launched):
    from vision_tagger.web import common

    # a Linux server without a graphical session (reached through an SSH tunnel)
    monkeypatch.setattr(common, "has_desktop", lambda: reveal.has_desktop("linux", {}))
    with TestClient(create_app(settings), **LOCAL) as c:
        assert "/reveal" not in ok(c.get("/database")).text
        assert c.post("/reveal", data={"kind": "data"}).status_code == 403
    assert reveal.has_desktop("win32", {}) and reveal.has_desktop("darwin", {})
    assert reveal.has_desktop("linux", {"WAYLAND_DISPLAY": "wayland-0"})
    assert not reveal.has_desktop("linux", {})


def test_a_missing_file_cannot_be_shown(local, photos, launched):
    c = local
    (image_id,) = lose(c, photos, "IMG_20210104_120000.jpg")
    page = ok(c.get(f"/images/{image_id}")).text
    assert "The file is not there, so it cannot be shown in its folder." in page
    assert ("image", str(image_id)) not in reveal_forms(page)
    r = c.post("/reveal", data={"kind": "image", "id": image_id}, headers={"HX-Request": "true"})
    assert r.status_code == 204 and "err=The+file+is+not+there" in r.headers["hx-redirect"]
    page = ok(c.post("/reveal", data={"kind": "image", "id": image_id})).text
    assert "The file is not there:" in page
    with c.ctx.db.write() as s:
        s.merge(AppSetting(key=ROOTS_KEY, value=[str(photos / "gone")]))
    page = ok(c.post("/reveal", data={"kind": "root", "id": 0})).text
    assert "The folder is not there:" in page
    assert launched == []


def test_file_manager_that_cannot_start(local, monkeypatch):
    def broken(_action):
        raise reveal.RevealError("xdg-open: not found")

    local.ctx.file_manager.launcher = broken
    page = ok(local.post("/reveal", data={"kind": "data"})).text
    assert "The file manager could not be opened: xdg-open: not found" in page


def test_file_manager_commands(monkeypatch, tmp_path):
    monkeypatch.delenv("SYSTEMROOT", raising=False)
    photo = tmp_path / "My trip" / "a, b.jpg"  # spaces and a comma
    folder = photo.parent
    plan = reveal.plan
    # Windows: one command line, the path quoted after "/select," (explorer's own parsing)
    assert plan(photo, select=True, platform="win32") == reveal.Launch(
        command=f'"C:\\Windows\\explorer.exe" /select,"{photo}"'
    )
    monkeypatch.setenv("SYSTEMROOT", "D:\\Win")
    assert plan(photo, select=True, platform="win32").command.startswith(
        '"D:\\Win\\explorer.exe" /select,'
    )
    assert plan(folder, select=False, platform="win32") == reveal.Launch(startfile=str(folder))
    assert plan(photo, select=True, platform="darwin").command == ["open", "-R", str(photo)]
    assert plan(folder, select=False, platform="darwin").command == ["open", str(folder)]
    # Linux can't select a file: its folder opens
    assert plan(photo, select=True, platform="linux").command == ["xdg-open", str(folder)]
    assert plan(folder, select=False, platform="linux").command == ["xdg-open", str(folder)]


def test_file_manager_follows_the_platform(monkeypatch, tmp_path):
    calls: list[reveal.Launch] = []
    manager = reveal.FileManager(calls.append)
    monkeypatch.setattr(sys, "platform", "darwin")
    manager.show(tmp_path / "a.jpg", select=True)
    monkeypatch.setattr(sys, "platform", "linux")
    manager.show(tmp_path / "a.jpg", select=True)
    assert [a.command for a in calls] == [
        ["open", "-R", str(tmp_path / "a.jpg")],
        ["xdg-open", str(tmp_path)],
    ]


def test_launch_is_detached_and_gets_the_childs_environment(monkeypatch):
    started = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            started.update(command=command, **kwargs)

        def wait(self):
            started["waited"] = True

    monkeypatch.setattr(reveal.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/mira/_internal")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/lib")
    reveal.launch(reveal.Launch(command=["xdg-open", "/photos"]))
    assert started["command"] == ["xdg-open", "/photos"]
    assert started["env"]["LD_LIBRARY_PATH"] == "/usr/lib"  # not the bundle's libraries
    assert started["stdin"] is reveal.subprocess.DEVNULL
    if sys.platform != "win32":
        assert started["start_new_session"] is True

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("xdg-open")

    monkeypatch.setattr(reveal.subprocess, "Popen", missing)
    with pytest.raises(reveal.RevealError):
        reveal.launch(reveal.Launch(command=["xdg-open", "/photos"]))


def test_file_manager_does_not_inherit_mira_s_libraries(monkeypatch, tmp_path):
    own = tmp_path / "venv" / "lib" / "python3.12"
    bundled = own / "site-packages" / "cv2" / ".." / ".." / "lib64"  # what OpenCV prepends
    other = tmp_path / "opt" / "lib"
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join([str(bundled), str(other)]))
    env = reveal.environment([own])
    assert env["LD_LIBRARY_PATH"] == str(other)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(bundled))
    assert "LD_LIBRARY_PATH" not in reveal.environment([own])
    monkeypatch.setenv("LD_LIBRARY_PATH", str(other))
    assert reveal.environment([own]) is None  # nothing to change: inherit
