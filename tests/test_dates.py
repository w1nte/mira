"""Date groups (``vision_tagger.dates``) and the "Dates" page."""

from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vision_tagger.dates import (
    Photo,
    apply_dates,
    default_date_reason,
    group_photos,
    mark_default_dates,
    planned_dates,
)
from vision_tagger.db.models import Image, ImageProperty, StageStatus, Status
from vision_tagger.web.app import create_app

from .helpers import add_image

JULY = datetime(2015, 7, 4, 14, 0)


def photo(n: int, name: str, when: datetime | None, source: str = "exif", **kw) -> Photo:
    return Photo(
        id=n,
        folder=kw.get("folder", "2015/summer"),
        filename=name,
        recorded_at=when,
        date_source=source,
        camera=kw.get("camera", "Canon"),
        phash=kw.get("phash", f"{n:016x}"),
    )


def by_member(groups):
    return {p.id: g for g in groups for p in g.photos}


def test_one_event_and_a_copy_from_a_messenger():
    photos = [photo(i, f"IMG_010{i}.jpg", JULY.replace(minute=i)) for i in range(1, 5)]
    # the same camera's next shot, but sent through a messenger: only the file time is left
    photos.append(photo(5, "IMG_0105.jpg", datetime(2023, 3, 1, 9, 0), "mtime", camera=None))
    groups = group_photos(photos)
    assert len(groups) == 1
    g = groups[0]
    assert g.status == "mixed" and g.needs_attention and g.off == {5}
    assert g.date == JULY.replace(minute=2)  # lower median of the trustworthy dates
    assert g.suggestions == {5: JULY.replace(minute=4)}  # its neighbour IMG_0104


def test_consistent_event_needs_no_look():
    photos = [photo(i, f"DSC{i:04d}.jpg", JULY.replace(hour=10 + i)) for i in range(4)]
    (g,) = group_photos(photos)
    assert g.status == "ok" and not g.off and not g.suggestions


def test_trustworthy_dates_far_apart_stay_separate():
    # consecutive numbers, but Christmas and a week later: two events, nothing is "wrong"
    photos = [
        photo(1, "IMG_0100.jpg", datetime(2015, 12, 24, 18)),
        photo(2, "IMG_0101.jpg", datetime(2015, 12, 24, 19)),
        photo(3, "IMG_0102.jpg", datetime(2015, 12, 31, 23, 4, 51)),
        photo(4, "IMG_0103.jpg", datetime(2015, 12, 31, 23, 31, 7)),
    ]
    groups = by_member(group_photos(photos))
    assert groups[1] is groups[2] and groups[3] is groups[4] and groups[1] is not groups[3]
    assert all(g.status == "ok" for g in groups.values())


def test_scans_with_file_times_only():
    photos = [
        photo(i, f"scan_{i:03d}.jpg", datetime(2024, 1, 5, 20, i), "mtime", camera=None)
        for i in range(1, 4)
    ]
    photos.append(photo(9, "unrelated.jpg", datetime(2024, 1, 5, 20, 2), "mtime", camera=None))
    groups = by_member(group_photos(photos))
    assert groups[1] is groups[2] is groups[3]
    assert groups[1].status == "unreliable" and groups[1].date is None
    assert groups[9].size == 1  # file times close together alone don't link


def test_copies_are_grouped_across_folders():
    photos = [
        photo(1, "IMG_0001.jpg", JULY, phash="abcdef0123456789"),
        photo(2, "IMG_0002.jpg", JULY.replace(minute=5)),
        photo(3, "IMG-WA0007.jpg", datetime(2022, 1, 1), "mtime", folder="WhatsApp",
              phash="abcdef0123456789", camera=None),
    ]  # fmt: skip
    g = by_member(group_photos(photos))[3]
    assert {p.id for p in g.photos} == {1, 2, 3} and g.folders == ["2015/summer", "WhatsApp"]
    assert g.off == {3} and g.suggestions[3] == g.date  # no trusted neighbour in its folder


def test_look_alike_neighbours_join_unless_both_dates_are_trusted():
    a, b = np.zeros(4, np.float32), np.zeros(4, np.float32)
    a[0] = b[0] = 1.0
    vectors = {1: a, 2: b, 3: a}.get
    photos = [
        photo(1, "beach.jpg", JULY),
        photo(2, "beach copy edited.jpg", datetime(2021, 6, 1), "mtime"),
        photo(3, "sunset.jpg", datetime(2019, 7, 4)),  # same look, but a different year
    ]
    groups = by_member(group_photos(photos, vectors))
    assert groups[1] is groups[2] and groups[2].off == {2}
    assert groups[3] is not groups[1]
    assert by_member(group_photos(photos))[2].size == 1  # without vectors: no link


def test_a_date_set_by_hand_is_never_contradicted():
    photos = [photo(i, f"IMG_000{i}.jpg", JULY.replace(minute=i)) for i in range(1, 4)]
    photos.append(photo(4, "IMG_0004.jpg", datetime(2015, 7, 2, 20), "manual"))
    (g,) = group_photos(photos)  # linked by name, and more than a day before the others
    assert 4 not in g.off and g.status == "ok"


@pytest.mark.parametrize(
    ("when", "reason"),
    [
        (datetime(2000, 12, 31, 22, 0), "epoch"),  # 2001-01-01 00:00 UTC, seen in UTC-2
        (datetime(2000, 1, 1, 0, 0), "epoch"),
        (datetime(2000, 1, 2, 9, 14, 37), "epoch"),  # a reset clock running on
        (datetime(1970, 1, 1, 1, 0), "epoch"),
        (datetime(1969, 12, 31, 19, 0), "epoch"),
        (datetime(1980, 1, 1, 0, 0, 2), "epoch"),
        (datetime(2012, 1, 1, 0, 0), "new_year"),  # a maker's default, any year
        (datetime(2011, 12, 31, 23, 45), "new_year"),
        (datetime(2040, 5, 1), "future"),
        (datetime(2011, 12, 31, 23, 59, 58), None),  # fireworks: a real second
        (datetime(2012, 1, 1, 0, 7), None),
        (datetime(2000, 3, 1, 0, 0), None),
        (datetime(2015, 7, 4, 14, 0), None),
        (None, None),
    ],
)
def test_default_date_reason(when, reason):
    assert default_date_reason(when, now=datetime(2026, 11, 27)) == reason


def test_repeated_dates_in_unrelated_photos_are_defaults():
    noon = datetime(2007, 6, 1, 12, 0)
    photos = [
        photo(1, "a.jpg", noon, folder="x"),
        photo(2, "b.jpg", noon, folder="y"),
        photo(3, "c.jpg", noon, folder="z"),
        photo(4, "d.jpg", noon, "manual", folder="z"),  # set by hand: never a default
        # a burst: the same second, but one folder
        photo(5, "e.jpg", datetime(2008, 5, 5, 10, 0), folder="burst"),
        photo(6, "f.jpg", datetime(2008, 5, 5, 10, 0), folder="burst"),
        photo(7, "g.jpg", datetime(2008, 5, 5, 10, 0), folder="burst"),
        # copies: the same pHash in three folders
        photo(8, "h.jpg", datetime(2009, 9, 9, 9, 0), folder="p", phash="ab" * 8),
        photo(9, "h.jpg", datetime(2009, 9, 9, 9, 0), folder="q", phash="ab" * 8),
        photo(10, "h.jpg", datetime(2009, 9, 9, 9, 0), folder="r", phash="ab" * 8),
    ]
    mark_default_dates(photos)
    assert {p.id: p.default for p in photos if p.default} == {1: "repeated", 2: "repeated",
                                                              3: "repeated"}  # fmt: skip
    assert photos[3].trusted  # the manual date stays trustworthy


def test_a_default_date_in_a_group_gets_its_neighbours_date():
    photos = [photo(i, f"IMG_010{i}.jpg", JULY.replace(minute=i)) for i in range(1, 5)]
    photos.append(photo(5, "IMG_0105.jpg", datetime(2000, 12, 31, 22, 0)))  # EXIF, but a reset
    (g,) = group_photos(photos)
    assert g.date == JULY.replace(minute=2)  # the default date doesn't count
    assert g.off == {5} and g.suggestions == {5: JULY.replace(minute=4)}
    assert [p.id for p in g.defaults] == [5]


def test_a_reset_camera_batch_has_no_date_to_suggest():
    photos = [photo(i, f"DSC_00{i}.jpg", datetime(2000, 1, 1, 0, 3 * i, 11)) for i in range(1, 5)]
    (g,) = group_photos(photos)  # grouped by file name: their dates don't count
    assert g.status == "unreliable" and g.off == {1, 2, 3, 4} and not g.suggestions
    assert not g.loose


def test_lone_default_dates_are_collected():
    photos = [
        photo(1, "party.jpg", datetime(2000, 12, 31, 22, 0), folder="a"),
        photo(2, "beach.jpg", datetime(1970, 1, 1, 1, 0), folder="b"),
        photo(3, "fine.jpg", datetime(2015, 7, 4), folder="c"),
        photo(4, "set.jpg", datetime(2000, 1, 1), "manual", folder="d"),
    ]
    groups = group_photos(photos)
    assert groups[0].loose and {p.id for p in groups[0].photos} == {1, 2}
    assert groups[0].needs_attention and not groups[0].suggestions
    assert all(not g.loose and g.size == 1 for g in groups[1:])


def test_planned_dates_set_and_shift():
    photos = [
        photo(1, "a.jpg", datetime(2000, 1, 1, 0, 5), "exif"),
        photo(2, "b.jpg", datetime(2000, 1, 1, 2, 35), "exif"),
        photo(3, "c.jpg", None, "mtime"),
    ]
    day = date(2016, 8, 20)
    plan = planned_dates(photos, day)
    assert plan == {
        1: datetime(2016, 8, 20, 0, 5),
        2: datetime(2016, 8, 20, 2, 35),
        3: datetime(2016, 8, 20, 12, 0),
    }
    assert set(planned_dates(photos, day, time(9, 30)).values()) == {datetime(2016, 8, 20, 9, 30)}
    # a camera clock reset to 2000: the first photo gets the date, the others keep their spacing
    shifted = planned_dates(photos, day, time(10, 0), shift=True)
    assert shifted == {
        1: datetime(2016, 8, 20, 10, 0),
        2: datetime(2016, 8, 20, 12, 30),
        3: datetime(2016, 8, 20, 10, 0),
    }
    assert planned_dates([], day) == {}


def test_apply_dates_marks_them_manual(db):
    with db.write() as s:
        add_image(s, 1)
        undated = add_image(s, 2, recorded_at=None)
        undated.date_source = "mtime"
        s.add(ImageProperty(image_id=1, key="year", value="2020", source="exif"))
        for n in (1, 2):
            s.add(StageStatus(image_id=n, stage="embed", status=Status.DONE))
    with db.write() as s:
        assert apply_dates(s, {1: datetime(2011, 5, 6, 7, 8), 2: datetime(2012, 1, 1, 12)}) == 2
    with db.session() as s:
        imgs = {i.id: i for i in s.scalars(select(Image))}
        assert imgs[1].recorded_at == datetime(2011, 5, 6, 7, 8)
        assert {i.date_source for i in imgs.values()} == {"manual"}
        query = select(ImageProperty.image_id, ImageProperty.value)
        years = {i: v for i, v in s.execute(query.where(ImageProperty.key == "year"))}
        assert years == {1: "2011", 2: "2012"}
        embed = s.scalars(select(StageStatus.status).where(StageStatus.stage == "embed")).all()
        assert set(embed) == {Status.PENDING}  # the search document mentions the date
    with pytest.raises(ValueError), db.write() as s:
        apply_dates(s, {1: datetime(2010, 1, 1), 2: datetime(1500, 1, 1)})
    with db.session() as s:
        assert s.get(Image, 1).recorded_at.year == 2011  # nothing changed


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        with app.state.ctx.db.write() as s:
            for i in range(1, 5):
                img = add_image(s, i, folder="2015/summer", recorded_at=JULY.replace(minute=i))
                img.filename = f"IMG_010{i}.jpg"
            copy = add_image(s, 5, folder="2015/summer", recorded_at=datetime(2023, 3, 1))
            copy.filename, copy.date_source = "IMG_0105.jpg", "mtime"
            other = add_image(s, 6, folder="misc", recorded_at=datetime(2019, 3, 1, 10))
            other.filename = "lonely.jpg"
        c.ctx = app.state.ctx
        yield c


def recorded(client, image_id):
    with client.ctx.db.session() as s:
        img = s.get(Image, image_id)
        return img.recorded_at, img.date_source


def test_dates_page_lists_groups_that_need_a_look(client):
    r = client.get("/dates")
    assert r.status_code == 200
    assert "1 date doesn\u2019t fit" in r.text
    assert 'name="suggested" value="5=2015-07-04T14:04:00"' in r.text
    assert 'value="5" form="dg-1" aria-label="Select IMG_0105.jpg" checked' in r.text
    assert "lonely.jpg" not in r.text  # a group of one is not listed
    assert 'href="/dates"' in client.get("/").text  # in the sidebar
    # a filter narrows the photos that are grouped
    assert "No photos belong together yet." in client.get("/dates?folder=misc").text


def test_lone_default_date_is_listed(client):
    with client.ctx.db.write() as s:
        img = add_image(s, 7, folder="old", recorded_at=datetime(2000, 12, 31, 22, 0))
        img.filename = "party.jpg"
    page = client.get("/dates").text
    assert "Default dates" in page and "party.jpg" in page and "default date" in page
    assert 'value="7" form="dg-7" aria-label="Select party.jpg">' in page  # not pre-ticked
    client.post("/dates/apply", data={"mode": "set", "image_ids": ["7"], "day": "2001-06-02"})
    assert recorded(client, 7) == (datetime(2001, 6, 2, 22, 0), "manual")
    assert "party.jpg" not in client.get("/dates").text


def test_use_suggested_dates(client):
    data = {"mode": "suggested", "image_ids": ["5"], "suggested": ["5=2015-07-04T14:04:00"]}
    r = client.post("/dates/apply", data=data)
    assert r.status_code == 200 and "Date corrected for 1 photo." in r.text
    assert recorded(client, 5) == (datetime(2015, 7, 4, 14, 4), "manual")
    assert "All groups have dates that agree." in client.get("/dates").text
    assert "Dates agree" in client.get("/dates?show=all").text
    # a suggestion for a photo that was not ticked is ignored
    data = {"mode": "suggested", "image_ids": ["1"], "suggested": ["5=2001-01-01T00:00:00"]}
    assert "None of the selected photos" in client.post("/dates/apply", data=data).text
    assert recorded(client, 5)[0].year == 2015


def test_set_and_shift_dates(client):
    r = client.post(
        "/dates/apply", data={"mode": "set", "image_ids": ["1", "2"], "day": "2014-02-03"}
    )
    assert "Dates corrected for 2 photos." in r.text
    assert recorded(client, 1) == (datetime(2014, 2, 3, 14, 1), "manual")  # keeps its time
    data = {"mode": "shift", "image_ids": ["3", "4"], "day": "2014-02-03", "at": "09:00"}
    client.post("/dates/apply", data=data)
    assert recorded(client, 3)[0] == datetime(2014, 2, 3, 9, 0)
    assert recorded(client, 4)[0] == datetime(2014, 2, 3, 9, 1)


def test_set_date_from_the_gallery_selection(client):
    page = client.get("/images").text
    assert 'formaction="/dates/apply"' in page and 'href="/dates"' in page
    r = client.post(
        "/dates/apply",
        data={"mode": "set", "image_ids": ["6"], "day": "1999-12-31", "value": "yes"},
        headers={"Referer": "http://testserver/images?folder=misc"},
    )
    assert r.url.path == "/images" and "Date corrected for 1 photo." in r.text
    assert recorded(client, 6) == (datetime(1999, 12, 31, 10, 0), "manual")


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"mode": "set", "day": "2014-02-03"}, "Select at least one photo."),
        ({"mode": "set", "image_ids": ["1"], "day": ""}, "Enter a date."),
        ({"mode": "set", "image_ids": ["1"], "day": "2014-02-03", "at": "noon"}, "Enter a time"),
        ({"mode": "set", "image_ids": ["1"], "day": "1500-01-01"}, "not plausible"),
    ],
)
def test_wrong_input_changes_nothing(client, data, error):
    r = client.post("/dates/apply", data=data)
    assert error in r.text
    assert recorded(client, 1) == (JULY.replace(minute=1), "exif")


def test_apply_needs_a_post_from_this_ui(client):
    data = {"mode": "set", "image_ids": ["1"], "day": "2014-02-03"}
    r = client.post("/dates/apply", data=data, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert recorded(client, 1)[1] == "exif"
