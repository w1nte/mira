from __future__ import annotations

from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vision_tagger import library
from vision_tagger.db.models import Cluster, Collection, ImageProperty, Person
from vision_tagger.web.app import create_app

from .helpers import add_face, add_image, person_centers


@pytest.fixture
def filled(db):
    today = date.today()
    with db.write() as s:
        cluster = Cluster(locked=True, person=Person(name="Anna"))
        unnamed = Cluster(locked=False)
        s.add_all([cluster, unnamed])
        s.flush()
        for n in range(1, 7):
            img = add_image(s, n, recorded_at=datetime(2015 + n, today.month, today.day, 12))
            add_face(s, img.id, person_centers(1)[0], cluster_id=cluster.id)
            if n <= 4:
                s.add(
                    ImageProperty(image_id=img.id, key="location_type", value="beach", source="vlm")
                )
        img = add_image(s, 7, recorded_at=datetime(2001, 1, 2))
        s.add(ImageProperty(image_id=img.id, key="event_type", value="birthday", source="vlm"))
    return db


def test_stats_memories_and_suggestions(filled):
    with filled.session() as s:
        st = library.stats(s)
        assert st.photos == 7 and st.people == 1 and st.unnamed_clusters == 1
        assert st.first_year == 2001 and st.with_faces == 6 and st.places == 1
        memories = library.on_this_day(s)
        assert [m.year for m in memories] == [2021, 2020, 2019, 2018, 2017, 2016]
        assert memories[0].years_ago == date.today().year - 2021
        suggested = {c.name: c for c in library.suggested(s)}
    assert suggested["Anna"].count == 6 and suggested["Anna"].query == "p=1"
    assert suggested["At the beach"].count == 4 and len(suggested["At the beach"].covers) == 4
    assert "Birthdays" not in suggested  # only one photo: not worth a collection


def test_saved_collections(filled):
    with filled.write() as s:
        a = library.add(s, "  Beach with Anna ", "p=1&t=location_type:beach&bogus=1")
        b = library.add(s, "Old", "to=2005")
    assert a.query == "p=1&t=location_type%3Abeach"  # normalised, unknown parameters dropped
    with filled.write() as s:
        library.move(s, b.id, -1)
        library.rename(s, a.id, "Anna at the sea")
    with filled.session() as s:
        views = library.saved(s)
    assert [v.name for v in views] == ["Old", "Anna at the sea"]
    assert views[1].count == 4 and views[0].count == 1
    with filled.write() as s, pytest.raises(ValueError):
        library.add(s, "   ", "")
    with filled.write() as s:
        library.delete(s, a.id)
    with filled.session() as s:
        assert [c.name for c in s.scalars(select(Collection))] == ["Old"]


@pytest.fixture
def client(settings, filled):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_home_collections_and_pipeline_pages(client):
    page = client.get("/").text
    assert "Your photos" in page and "On this day" in page and "At the beach" in page
    assert "face group has no name yet" in page  # nudge towards naming
    assert "Run pipeline" not in page  # the technical part moved
    pipeline = client.get("/pipeline").text
    assert "Run pipeline" in pipeline and 'hx-get="/partials/status"' in pipeline

    page = client.get("/images?t=location_type:beach").text
    assert "Save as collection" in page
    r = client.post(
        "/collections",
        data={"name": "Sea", "query": "t=location_type:beach"},
        headers={"Referer": "http://testserver/images?t=location_type:beach"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/images?")
    assert "Sea" in client.get("/").text
    page = client.get("/collections").text
    assert "Your collections" in page and 'action="/collections/1/rename"' in page
    client.post("/collections/1/rename", data={"name": "Ocean"})
    assert "Ocean" in client.get("/collections").text
    client.post("/collections/1/delete")
    assert "No saved collections yet" in client.get("/collections").text
    r = client.post("/collections", data={"name": "x"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
