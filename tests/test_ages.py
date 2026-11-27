from __future__ import annotations

import itertools
import re
import sqlite3
from datetime import date, datetime

import numpy as np
import pytest
from sqlalchemy import func, select

from vision_tagger.db.models import Cluster, Image, ImageProperty, Person, StageStatus, Status
from vision_tagger.db.session import Database
from vision_tagger.faces.ages import (
    age_label,
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


# --------------------------------------------------------------------------- manual ages
def _face_of(s, image_id):
    from vision_tagger.db.models import Face

    return s.scalar(select(Face).where(Face.image_id == image_id))


def test_manual_age_keeps_the_date_and_replaces_the_estimate(db, anna):
    from vision_tagger.faces.ages import age_label, set_face_age

    with db.write() as s:
        s.get(Person, anna).birth_year = 1990
    with db.write() as s:
        assert check_image_date(s, 6).suspicious  # 2023, but the face looks 8
        set_face_age(s, _face_of(s, 6).id, 33)  # the date is right: Anna is 33 there
    with db.session() as s:
        check = check_image_date(s, 6)
        assert not check.suspicious
        info = next(iter(check.faces.values()))
        assert info.manual and info.face_age == 33 and age_label(info) == "33"
        assert s.get(Image, 6).date_source == "mtime"  # the date is untouched
        ages = analyse_person(s, s.get(Person, anna))
        face = next(f for f in ages.faces if f.image_id == 6)
        assert face.manual and not face.suspicious and face.best_age == 33
        assert {f.image_id for f in ages.suspicious} == {5}
    # a manual age that contradicts the date flags the date (the user said how old Anna is)
    with db.write() as s:
        set_face_age(s, _face_of(s, 2).id, 12)
    with db.session() as s:
        check = check_image_date(s, 2)
        assert check.suspicious and check.suggested_year == 2002
    # back to the estimate
    with db.write() as s:
        set_face_age(s, _face_of(s, 6).id, None)
        set_face_age(s, _face_of(s, 2).id, None)
    with db.session() as s:
        assert check_image_date(s, 6).suspicious and not check_image_date(s, 2).suspicious
    with db.write() as s, pytest.raises(ValueError):
        set_face_age(s, _face_of(s, 6).id, 300)
    with db.write() as s, pytest.raises(LookupError):
        set_face_age(s, 999_999, 3)


def test_manual_ages_count_most_for_the_birth_year(db, anna):
    from vision_tagger.faces.ages import set_face_age

    with db.write() as s:
        for n in (1, 2, 3):  # 2010, 2015, 2018: Anna really was born in 1992
            when = s.get(Image, n).recorded_at
            set_face_age(s, _face_of(s, n).id, when.year - 1992)
    with db.session() as s:
        est = estimate_birth_year(person_faces(s, anna))
    assert est.year == 1992


def test_manual_age_survives_redetection(db, anna):
    from vision_tagger.db.models import Face
    from vision_tagger.faces.clustering import replace_faces

    with db.write() as s:
        old = _face_of(s, 6)
        old.manual_age = 33
        fresh = Face(
            x1=old.x1 + 2,
            y1=old.y1,
            x2=old.x2 + 2,
            y2=old.y2,
            det_score=0.9,
            embedding=old.embedding,
            age=9,
            face_px=100,
        )
        replace_faces(s, 6, [fresh])
    with db.session() as s:
        face = _face_of(s, 6)
        assert face.age == 9 and face.manual_age == 33


def test_sort_by_age_of_one_person(db, anna):
    from vision_tagger.query import ImageFilter, all_image_ids, neighbours, page_images, sort_ages

    with db.write() as s:
        s.get(Person, anna).birth_year = 1990
        add_image(s, 50, recorded_at=datetime(2021, 1, 1))  # without Anna
    f = ImageFilter(persons=[anna], sort="age_asc")
    with db.session() as s:
        # doubtful dates (5: camera clock 2000, 6: scanned childhood photo) go by the face
        assert all_image_ids(s, f) == [6, 1, 2, 3, 4, 5]
        assert all_image_ids(s, f.copy(sort="age_desc")) == [5, 4, 3, 2, 1, 6]
        assert [img.id for img in page_images(s, f, 1, 2)] == [1, 2]
        assert neighbours(s, f, 1) == (6, 2)
        ages = sort_ages(s, f)
        assert ages[6] == (8, True) and ages[4] == (30, False)
        # without exactly one person there is no age to sort by
        assert ImageFilter(sort="age_asc").sort == "date_desc"
        assert ImageFilter.from_params({"sort": "age_asc"}).sort == "date_desc"
        assert ImageFilter(persons=[anna], sort="age_asc").copy(persons=[]).sort == "date_desc"
        assert sort_ages(s, ImageFilter(persons=[anna])) is None
        # an unnamed cluster sorts by its face ages
        cluster_id = s.scalar(select(Cluster.id).where(Cluster.person_id == anna))
        by_cluster = ImageFilter(clusters=[cluster_id], sort="age_asc")
        assert all_image_ids(s, by_cluster) == [6, 1, 2, 3, 4, 5]


def _child_photo(s, anna, n=40, when=datetime(2000, 6, 1, 15, 12, 9), face_age=30):
    """Anna (born 1990) at ten, on a camera date, but the face model thinks she is 30."""
    cluster_id = s.scalar(select(Cluster.id).where(Cluster.person_id == anna))
    img = add_image(s, n, recorded_at=when)
    add_face(s, img.id, person_centers(1)[0], cluster_id=cluster_id, age=face_age)
    return cluster_id, img.id


def test_set_birth_year_and_camera_date_give_the_age(db, anna):
    from vision_tagger.query import ImageFilter, sort_ages

    with db.write() as s:
        s.get(Person, anna).birth_year = 1990
        cluster_id, child = _child_photo(s, anna)
    with db.session() as s:
        ages = sort_ages(s, ImageFilter(persons=[anna], sort="age_asc"))
        assert ages[child] == (10, False)  # birth year + date, not the face's 30
        assert ages[1] == (20, False) and ages[2] == (25, False)
        # untrustworthy dates still go by the face: a camera's default date, a file time
        assert ages[5] == (33, True) and ages[6] == (8, True)
        # filtering by Anna's cluster instead of Anna: her birth year all the same
        by_cluster = sort_ages(s, ImageFilter(clusters=[cluster_id], sort="age_asc"))
        assert by_cluster[child] == (10, False) and by_cluster[6] == (8, True)
        # the face still disagrees with the date: the person page keeps the hint
        flagged = {f.image_id for f in analyse_person(s, s.get(Person, anna)).suspicious}
        assert child in flagged


def test_a_default_date_is_doubted_on_the_photo_page(db, anna):
    with db.write() as s:
        s.get(Person, anna).birth_year = 1990
        _cluster, photo = _child_photo(
            s, anna, n=41, when=datetime(2000, 12, 31, 22, 0), face_age=24
        )
    with db.session() as s:
        check = check_image_date(s, photo)
    assert check.suspicious and not check.date_trusted
    (info,) = check.faces.values()
    assert age_label(info, check.suspicious, check.date_trusted) == "~24"


@pytest.fixture
def web(settings, db, anna):
    from fastapi.testclient import TestClient

    from vision_tagger.web.app import create_app

    with db.write() as s:
        s.get(Person, anna).birth_year = 1990
    with TestClient(create_app(settings)) as c:
        yield c


def test_gallery_offers_the_age_sort_for_one_person(web, anna):
    page = web.get("/images").text
    assert 'value="age_asc"' not in page
    page = web.get(f"/images?p={anna}&sort=age_asc").text
    assert 'value="age_asc" selected' in page and 'value="age_desc"' in page
    ids = [int(i) for i in re.findall(r'data-id="(\d+)"', page)]
    assert ids == [6, 1, 2, 3, 4, 5]
    assert '<span class="agebadge est"' in page and ">~8</span>" in page
    page = web.get(f"/images/1?p={anna}&sort=age_asc").text
    assert f'href="/images/6?p={anna}&amp;sort=age_asc"' in page  # previous: the younger one


def test_image_page_shows_the_calendar_age_for_a_camera_date(web, anna):
    with web.app.state.ctx.db.write() as s:
        _cluster, child = _child_photo(s, anna)
    page = web.get(f"/images/{child}").text
    assert "Anna · 10</span>" in page  # the face box: her real age, not ~30
    assert "would be 10, looks about 30" in page  # the hint that face and date disagree
    page = web.get(f"/images?p={anna}&sort=age_asc").text
    assert '<span class="agebadge "' in page and ">10</span>" in page


def test_image_page_offers_both_fixes(web, anna):
    with web.app.state.ctx.db.session() as s:
        face_id = _face_of(s, 6).id
    page = web.get("/images/6").text
    assert "Set year to 1998" in page and "Anna is 33 here" in page
    r = web.post(
        f"/images/6/faces/{face_id}/age",
        data={"age": "33"},
        headers={"referer": "http://testserver/images/6"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = web.get(r.headers["location"]).text
    assert "This date looks wrong" not in page and "Age set to 33." in page
    assert "Automatic" in page
    with web.app.state.ctx.db.session() as s:
        assert _face_of(s, 6).manual_age == 33 and s.get(Image, 6).date_source == "mtime"
    # the person page lists the doubtful photo 5 with both fixes, 6 no longer
    page = web.get(f"/persons/{anna}").text
    assert "Keep date, age 10" in page and "Set to 2023" in page
    # reset, wrong face, nonsense
    web.post(f"/images/6/faces/{face_id}/age", data={"reset": "1"})
    with web.app.state.ctx.db.session() as s:
        assert _face_of(s, 6).manual_age is None
    r = web.post(f"/images/5/faces/{face_id}/age", data={"age": "3"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
    r = web.post(f"/images/6/faces/{face_id}/age", data={"age": "abc"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
