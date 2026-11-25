from __future__ import annotations

import sqlite3
from datetime import date, datetime

import pytest
from sqlalchemy import select

from vision_tagger.db.models import Cluster, Image, ImageProperty, Person, StageStatus, Status
from vision_tagger.db.session import Database
from vision_tagger.faces.ages import (
    analyse_person,
    check_image_date,
    correct_image_date,
    estimate_birth_year,
    person_faces,
)

from .helpers import add_face, add_image, person_centers

# (n, recorded_at, date_source, face age) for a person born in 1990
PHOTOS = [
    (1, datetime(2010, 6, 1), "exif", 20),
    (2, datetime(2015, 6, 1), "exif", 25),
    (3, datetime(2018, 6, 1), "exif", 27),
    (4, datetime(2020, 6, 1), "filename", 31),
    (5, datetime(2000, 1, 1), "exif", 33),  # camera clock reset to 2000
    (6, datetime(2023, 3, 3), "mtime", 8),  # scanned childhood photo: only the file date
]


@pytest.fixture
def anna(db):
    center = person_centers(1)[0]
    with db.write() as s:
        cluster = Cluster(locked=True, person=Person(name="Anna"))
        s.add(cluster)
        s.flush()
        for n, when, source, age in PHOTOS:
            img = add_image(s, n, recorded_at=when)
            img.date_source = source
            img.stages.append(StageStatus(stage="embed", status=Status.DONE))
            s.add(ImageProperty(image_id=img.id, key="year", value=str(when.year), source="exif"))
            add_face(s, img.id, center, cluster_id=cluster.id, age=age)
        return cluster.person_id


def test_birth_year_estimate_is_robust_to_wrong_dates(db, anna):
    with db.session() as s:
        est = estimate_birth_year(person_faces(s, anna))
    assert 1989 <= est.year <= 1991
    assert est.samples >= 4 and est.confidence in ("medium", "high")


def test_analysis_flags_doubtful_dates(db, anna):
    with db.write() as s:
        person = s.get(Person, anna)
        person.birth_year = 1990
        ages = analyse_person(s, person, today=date(2026, 1, 1))
    assert ages.birth_source == "set" and ages.current_age == 36
    flagged = {f.image_id: f for f in ages.suspicious}
    assert set(flagged) == {5, 6}
    assert flagged[5].calendar_age == 10 and flagged[5].suggested_year == 2023
    assert flagged[6].suggested_year == 1998
    ok = next(f for f in ages.faces if f.image_id == 4)
    assert ok.best_age == 30 and not ok.suspicious  # calendar age wins when the date fits


def test_estimate_is_used_when_no_birth_year(db, anna):
    with db.session() as s:
        ages = analyse_person(s, s.get(Person, anna))
    assert ages.birth_source == "estimated"
    assert {f.image_id for f in ages.suspicious} == {5, 6}


def test_image_date_check_and_correction(db, anna):
    with db.write() as s:
        s.get(Person, anna).birth_year = 1990
    with db.session() as s:
        check = check_image_date(s, 6)
        good = check_image_date(s, 2)
    assert check.suspicious and check.suggested_year == 1998
    assert not good.suspicious
    info = next(iter(good.faces.values()))
    assert info.calendar_age == 25 and info.person == "Anna"
    with db.write() as s:
        correct_image_date(s, 6, 1998)
    with db.session() as s:
        img = s.get(Image, 6)
        assert img.recorded_at.year == 1998 and img.date_source == "manual"
        assert (
            s.scalar(
                select(ImageProperty.value).where(
                    ImageProperty.image_id == 6, ImageProperty.key == "year"
                )
            )
            == "1998"
        )
        assert s.get(StageStatus, (6, "embed")).status == Status.PENDING
        assert not check_image_date(s, 6).suspicious
    with db.write() as s, pytest.raises(ValueError):
        correct_image_date(s, 6, 1200)


def test_manual_dates_survive_reindex(db, anna):
    from types import SimpleNamespace

    from vision_tagger.indexer import Indexer

    with db.write() as s:
        correct_image_date(s, 6, 1998)
        img = s.get(Image, 6)
        info = SimpleNamespace(
            width=1,
            height=1,
            recorded_at=datetime(2024, 1, 1),
            date_source="mtime",
            camera=None,
            meta={},
        )
        Indexer._apply_info(img, info)
        assert img.recorded_at.year == 1998 and img.date_source == "manual"


def test_old_database_gets_new_columns(tmp_path):
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE persons (id INTEGER PRIMARY KEY, name TEXT UNIQUE, notes TEXT, "
        "created_at DATETIME)"
    )
    con.execute("INSERT INTO persons (name) VALUES ('Anna')")
    con.commit()
    con.close()
    db = Database(path)
    db.create_all()
    with db.write() as s:
        p = s.scalar(select(Person))
        assert p.name == "Anna" and p.birth_year is None
        p.birth_year = 1990
    db.dispose()
