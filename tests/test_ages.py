from __future__ import annotations

import itertools
import sqlite3
from datetime import date, datetime

import numpy as np
import pytest
from sqlalchemy import func, select

from vision_tagger.db.models import Cluster, Image, ImageProperty, Person, StageStatus, Status
from vision_tagger.db.session import Database
from vision_tagger.faces.ages import (
    analyse_person,
    check_image_date,
    correct_image_date,
    estimate_birth_year,
    person_faces,
)

from .helpers import add_face, add_image, noisy, person_centers

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


def test_old_database_gets_birth_year_via_migration(tmp_path):
    from vision_tagger.db.migrate import MigrationRequired, upgrade

    from .test_migrations import build_at, url_of

    path = tmp_path / "old.sqlite3"
    build_at(path, "0001", keep_version_table=False)
    con = sqlite3.connect(path)
    con.execute("INSERT INTO persons (name, created_at) VALUES ('Anna', '2024-01-01 00:00:00')")
    con.commit()
    con.close()
    db = Database(path)
    with pytest.raises(MigrationRequired):
        db.create_all()
    db.dispose()
    upgrade(url_of(path), backup=False)
    db = Database(path)
    db.create_all()
    with db.write() as s:
        p = s.scalar(select(Person))
        assert p.name == "Anna" and p.birth_year is None
        p.birth_year = 1990
    db.dispose()


# --------------------------------------------------------------------------- realistic families
from vision_tagger.faces.ages import AgeCalibration, age_sigma, get_calibration  # noqa: E402


def add_person(s, name, faces, birth_year=None, seed=0):
    """faces: (recorded_at, date_source, face age, face_px). Returns the person id."""
    rng = np.random.default_rng(seed)
    center = person_centers(1, seed=seed + 11)[0]
    cluster = Cluster(locked=True, person=Person(name=name, birth_year=birth_year))
    s.add(cluster)
    s.flush()
    for when, source, age, px in faces:
        n = int(s.scalar(select(func.count()).select_from(Image)) or 0) + 1
        img = add_image(s, n, recorded_at=when)
        img.date_source = source
        add_face(s, img.id, noisy(center, rng, 0.3), cluster_id=cluster.id, age=age, face_px=px)
    s.flush()
    return cluster.person_id


def model_age(true_age, rng, bias=0.0):
    """A face age as the model reports it: noisy, more so for adults."""
    return max(0, round(true_age + bias + rng.normal(0, age_sigma(true_age) * 0.8)))


def child_photos(birth, years, rng, per_year=3, source="exif", bias=0.0):
    out = []
    for year in years:
        for k in range(per_year):
            true = year - birth
            out.append((datetime(year, 1 + 3 * k, 10), source, model_age(true, rng, bias), 120))
    return out


def test_children_get_an_exact_birth_year_and_no_false_alarms(db):
    rng = np.random.default_rng(3)
    with db.write() as s:
        pid = add_person(s, "Mia", child_photos(2015, range(2015, 2025), rng))
    with db.session() as s:
        est = estimate_birth_year(person_faces(s, pid))
        ages = analyse_person(s, s.get(Person, pid), today=date(2026, 1, 1))
    assert est.year in (2014, 2015, 2016) and est.confidence == "high"
    assert ages.birth_source == "estimated"
    assert ages.suspicious == []  # correct dates of a growing child are never doubted


def test_wrong_dates_are_found_and_right_ones_kept(db):
    rng = np.random.default_rng(4)
    photos = child_photos(2012, range(2012, 2024), rng)
    photos += [
        (datetime(2000, 1, 1), "exif", 6, 120),  # camera clock reset, before birth
        (datetime(2024, 5, 1), "mtime", 2, 120),  # scan of a baby photo, file date only
    ]
    with db.write() as s:
        pid = add_person(s, "Ben", photos, birth_year=2012)
    with db.session() as s:
        ages = analyse_person(s, s.get(Person, pid), today=date(2026, 1, 1))
    flagged = sorted((f.recorded_at.year, f.suggested_year) for f in ages.suspicious)
    assert [year for year, _ in flagged] == [2000, 2024]  # exactly the two wrong dates
    assert abs(flagged[0][1] - 2018) <= 1 and abs(flagged[1][1] - 2014) <= 1
    assert all(f.calendar_age == f.recorded_at.year - 2012 for f in ages.faces)


def test_file_dates_do_not_outvote_camera_dates(db):
    """Many scanned childhood photos (all with this year's file date) plus a few camera photos."""
    rng = np.random.default_rng(5)
    scans = [(datetime(2025, 3, 1), "mtime", model_age(a, rng), 120) for a in range(2, 12)] * 4
    camera = child_photos(1985, [2015, 2018, 2021, 2023], rng, per_year=3)
    with db.write() as s:
        pid = add_person(s, "Dad", scans + camera)
    with db.session() as s:
        est = estimate_birth_year(person_faces(s, pid))
    assert abs(est.year - 1985) <= 3  # not ~2018, which the scans alone would suggest


def test_tiny_faces_with_absurd_ages_barely_count(db):
    rng = np.random.default_rng(6)
    photos = child_photos(2010, [2014, 2016, 2018, 2020], rng, per_year=1)
    photos += [(datetime(2016, 1, 1), "exif", 45, 12)] * 2  # blurry background faces
    with db.write() as s:
        pid = add_person(s, "Lea", photos)
    with db.session() as s:
        est = estimate_birth_year(person_faces(s, pid))
    assert abs(est.year - 2010) <= 1


def test_birth_years_teach_the_model_its_bias(db):
    """Two people with a birth year show the model calls adults ~7 years too young; a third
    person without a birth year is corrected accordingly."""
    rng = np.random.default_rng(7)
    with db.write() as s:
        add_person(s, "Ann", child_photos(1980, range(2005, 2020), rng, bias=-7), 1980, seed=1)
        add_person(s, "Tom", child_photos(1975, range(2003, 2018), rng, bias=-7), 1975, seed=2)
        eva = child_photos(1978, range(2006, 2021), rng, bias=-7)
        pid = add_person(s, "Eva", eva, seed=3)
    with db.session() as s:
        calibration = get_calibration(s)
        assert calibration.active and calibration.samples >= 60
        assert 4 <= calibration.apply(30.0) - 30.0 <= 8
        corrected = estimate_birth_year(person_faces(s, pid))
        raw = estimate_birth_year(person_faces(s, pid, AgeCalibration()))
    assert abs(corrected.year - 1978) <= 1
    assert raw.year - 1978 >= 5  # without the birth years the estimate is years off


def test_calibration_needs_enough_examples_and_stays_monotonic():
    assert not AgeCalibration.fit([(10.0, 12.0, 1.0)] * 5).active
    pairs = [(float(p), float(p) + 6, 1.0) for p in range(20, 40)] * 2
    pairs += [(float(p), float(p) - 6, 1.0) for p in range(45, 60)] * 2
    cal = AgeCalibration.fit(pairs)
    grid = [cal.apply(float(a)) for a in range(0, 90)]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(grid))


def test_manual_dates_are_never_doubted(db):
    rng = np.random.default_rng(8)
    photos = child_photos(2012, range(2014, 2020), rng)
    photos.append((datetime(2001, 6, 1), "manual", 7, 120))  # the user said so
    with db.write() as s:
        pid = add_person(s, "Max", photos, birth_year=2012)
    with db.session() as s:
        ages = analyse_person(s, s.get(Person, pid))
    assert ages.suspicious == []
