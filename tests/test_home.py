from __future__ import annotations

import re
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


def test_old_collections_keep_working_and_alternatives_are_saved(filled):
    """Images 1-6: Anna, 2016-2021, 1-4 at the beach; image 7: a birthday in 2001."""
    with filled.write() as s:  # saved before alternatives existed, never normalised
        s.add(Collection(name="Any", query="t=location_type:beach&t=event_type:birthday&tm=any"))
        s.add(Collection(name="All", query="p=1&t=location_type%3Abeach&from=2017"))
        s.add(Collection(name="Place", query="place=1"))
    with filled.session() as s:
        views = {v.name: v for v in library.saved(s)}
    assert views["Any"].count == 5 and views["All"].count == 3 and views["Place"].count == 0
    with filled.write() as s:
        saved = library.add(
            s, "Or", "t=location_type:beach|event_type:birthday&d=2016..2017&d=2001"
        )
    assert saved.query == "t=location_type%3Abeach%7Cevent_type%3Abirthday&d=2016..2017&d=2001"
    with filled.session() as s:
        views = {v.name: v for v in library.saved(s)}
    assert views["Or"].count == 3  # images 1, 2 (beach in 2016 / 2017) and 7 (birthday 2001)


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


DE = {"Accept-Language": "de-DE,de;q=0.9"}


def test_gallery_builds_alternatives_from_the_facets(client, filled):
    with filled.write() as s:
        s.add(ImageProperty(image_id=5, key="location_type", value="mountain", source="vlm"))
        s.add(ImageProperty(image_id=6, key="location_type", value="city", source="vlm"))
        for tag in ("sun", "sand"):  # a key with several values per photo
            s.add(ImageProperty(image_id=1, key="tags", value=tag, source="vlm"))
    page = client.get("/images?t=location_type:beach").text
    assert 'class="or-toggle"' in page and 'x-data="{ alt: false }"' in page
    # the "or" list offers the other values (counted without the condition on the key)
    assert 'href="/images?t=location_type%3Abeach%7Clocation_type%3Amountain"' in page
    page = client.get("/images?t=location_type:beach|location_type:mountain").text
    assert "5 images match the filter" in page
    assert re.search(
        r'Location type:</span>.*beach.*<span class="or">or</span>.*mountain', page, re.S
    )
    assert 'x-data="{ alt: true }"' in page  # the key has alternatives: "or" mode is on
    assert 'href="/images?t=location_type%3Amountain"' in page  # remove one alternative
    assert "any property ⇄" in page  # split into plain conditions
    page = client.get("/images?t=location_type:beach|location_type:mountain", headers=DE).text
    assert '<span class="or">oder</span>' in page and "5 Bilder" in page
    # Ctrl+click on "sand": "sun or sand" instead of "sun and sand"
    page = client.get("/images?t=tags:sun").text
    sand = 'href="/images?t=tags%3Asun&amp;t=tags%3Asand"'
    assert f'{sand} data-or-href="/images?t=tags%3Asun%7Ctags%3Asand"' in page
    page = client.get("/images?t=location_type:beach&t=event_type:birthday").text
    assert "No images match." in page and "all properties ⇄" in page
    # dates as alternatives
    page = client.get("/images?d=2016&d=2021").text
    assert "2 images match the filter" in page
    assert re.search(r'2016.*<span class="or">or</span>.*2021', page, re.S)
    # properties page: "any selected" is one OR condition, the other filters stay
    r = client.get(
        "/images/add?p=1&pick=location_type:beach&pick=event_type:birthday&combine=any",
        follow_redirects=False,
    )
    assert r.headers["location"] == "/images?p=1&t=location_type%3Abeach%7Cevent_type%3Abirthday"
    r = client.get(
        "/images/add?p=1&pick=location_type:beach&pick=event_type:birthday&combine=all",
        follow_redirects=False,
    )
    assert r.headers["location"] == "/images?p=1&t=location_type%3Abeach&t=event_type%3Abirthday"
    # save the OR filter as a collection and open it again
    query = "t=location_type:beach|location_type:mountain"
    r = client.post(
        "/collections",
        data={"name": "Sea or hills", "query": query},
        headers={"Referer": f"http://testserver/images?{query}"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = client.get("/collections").text
    link = "/images?t=location_type%3Abeach%7Clocation_type%3Amountain"
    assert f'href="{link}"' in page and "5 photos" in page
    assert "5 images match the filter" in client.get(link).text


def test_gallery_persons_or(client, filled):
    with filled.write() as s:
        ben = Cluster(locked=True, person=Person(name="Ben"))
        s.add(ben)
        s.flush()
        add_face(s, 7, person_centers(2)[1], cluster_id=ben.id)
        ben_id = ben.person_id
    page = client.get("/images?p=1").text
    # one person chosen: Ben can be added as an alternative ("Anna or Ben")
    assert f'href="/images?p=1&amp;p={ben_id}&amp;pm=any"' in page
    page = client.get(f"/images?p=1&p={ben_id}&pm=any").text
    assert "7 images match the filter" in page
    assert re.search(r'Anna.*<span class="or">or</span>.*Ben', page, re.S)
    assert "any of these people ⇄" in page
