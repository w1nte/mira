"""Place / area filters, the map API, the map page and the place actions in the web UI."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.datastructures import QueryParams

from vision_tagger.db.models import Image, Place
from vision_tagger.geo.mapdata import heat_points
from vision_tagger.indexer import Indexer
from vision_tagger.query import ImageFilter, all_image_ids, count_images
from vision_tagger.web.app import create_app

from .conftest import make_image
from .test_geo import BERLIN, EIFFEL, write_geonames

# folder -> positions of its photos
LIBRARY = {
    "home": [BERLIN, (BERLIN[0] + 0.0004, BERLIN[1]), (BERLIN[0], BERLIN[1] + 0.0004)],
    "trip": [EIFFEL, (EIFFEL[0] + 0.0003, EIFFEL[1])],
    "fiji": [(-16.5, 179.9995)],  # next to the date line
    "scans": [None, None],  # no GPS
}


@pytest.fixture
def client(settings, photos, tmp_path):
    settings.geo.cities_file = write_geonames(tmp_path / "cities500.txt")
    seed = 0
    for folder, positions in LIBRARY.items():
        for pos in positions:
            seed += 1
            make_image(photos / folder / f"IMG_{seed}.jpg", seed=seed, gps=pos)
    app = create_app(settings)
    with TestClient(app) as c:
        ctx = app.state.ctx
        Indexer(ctx.db, settings).run()  # finds the places at the end
        c.ctx = ctx
        yield c


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def _places(client) -> dict[str, Place]:
    with client.ctx.db.session() as s:
        return {p.name: p for p in s.scalars(select(Place))}


def _filenames(client, f: ImageFilter) -> set[str]:
    with client.ctx.db.session() as s:
        ids = all_image_ids(s, f)
        return set(s.scalars(select(Image.filename).where(Image.id.in_(ids))))


# --------------------------------------------------------------------------- filters
def test_place_and_bbox_parameters_round_trip():
    f = ImageFilter.from_params(QueryParams("place=12&bbox=13.3,52.4,13.5,52.6&p=3"))
    assert f.places == [12] and f.bbox == (13.3, 52.4, 13.5, 52.6)
    assert f.query_string() == "p=3&place=12&bbox=13.3%2C52.4%2C13.5%2C52.6"
    assert ImageFilter.from_params(QueryParams(f.query_string())) == f
    assert f.query_string(places=[], bbox=None) == "p=3"
    assert not ImageFilter(places=[1]).is_empty and not ImageFilter(bbox=(0, 0, 1, 1)).is_empty
    for bad in ("1,2,3", "a,b,c,d", "0,10,1,5", "0,-91,1,1", "200,0,201,1", "nan,0,1,1"):
        assert ImageFilter.from_params(QueryParams(f"bbox={bad}")).bbox is None, bad
    assert ImageFilter.from_params(QueryParams("place=abc")).places == []
    long = ImageFilter(bbox=(13.123456789, 52.0, 13.5, 52.6)).query_string()
    assert "13.12346" in long  # five decimals (~1 m) are plenty


def test_place_and_bbox_filters_select_photos(client):
    places = _places(client)
    berlin = places["Berlin, Germany"]
    assert _filenames(client, ImageFilter(places=[berlin.id])) == {
        "IMG_1.jpg",
        "IMG_2.jpg",
        "IMG_3.jpg",
    }
    around_paris = ImageFilter(bbox=(2.0, 48.0, 3.0, 49.0))
    assert _filenames(client, around_paris) == {"IMG_4.jpg", "IMG_5.jpg"}
    across_dateline = ImageFilter(bbox=(179.0, -17.0, -179.0, -16.0))  # west > east
    assert _filenames(client, across_dateline) == {"IMG_6.jpg"}
    with client.ctx.db.session() as s:
        assert count_images(s, ImageFilter(bbox=(-10.0, 40.0, 20.0, 60.0))) == 5
        assert count_images(s, ImageFilter(places=[999_999])) == 0


def test_gallery_shows_place_chips_and_facet(client):
    berlin = _places(client)["Berlin, Germany"]
    page = ok(client.get(f"/images?place={berlin.id}")).text
    assert "Berlin, Germany" in page and "3 images" in page
    assert f'href="/map?place={berlin.id}"' in page  # the gallery links to its map
    page = ok(client.get("/images?bbox=2,48,3,49")).text
    assert "Map area" in page and "2 images" in page
    page = ok(client.get("/images")).text
    assert "Paris, France" in page  # the place facet (image property "place")
    page = ok(client.get("/images?t=place:Paris%2C%20France")).text
    assert "2 images" in page
    page = ok(client.get("/images?t=country:Germany")).text
    assert "3 images" in page


def test_places_as_alternatives(client):
    places = _places(client)
    berlin, paris = places["Berlin, Germany"], places["Paris, France"]
    assert _filenames(client, ImageFilter(places=[berlin.id, paris.id])) == {
        f"IMG_{n}.jpg" for n in range(1, 6)
    }
    page = ok(client.get(f"/images?place={berlin.id}&place={paris.id}")).text
    assert "5 images" in page
    assert '<span class="alt">Berlin, Germany' in page and '<span class="or">or</span>' in page
    assert f'href="/images?place={paris.id}"' in page  # removes Berlin
    # the same with the place properties (names with commas)
    page = ok(client.get("/images?t=place:Berlin, Germany|place:Paris, France")).text
    assert "5 images" in page


# --------------------------------------------------------------------------- API
def test_points_are_aggregated_and_filtered(client):
    data = ok(client.get("/api/geo/points")).json()
    assert data["photos"] == 6  # the two scans have no position
    assert len(data["points"]) < 6  # photos within ~100 m share a cell
    assert all(len(p) == 3 for p in data["points"])
    assert data["bounds"][0][0] == pytest.approx(-16.5) and data["max"] >= 2
    hits = client.ctx.cache.hits
    ok(client.get("/api/geo/points"))  # whole library: cached
    assert client.ctx.cache.hits == hits + 1
    trip = ok(client.get("/api/geo/points?folder=trip")).json()
    assert trip["photos"] == 2
    assert {round(p[0]) for p in trip["points"]} == {49}
    assert ok(client.get("/api/geo/points?folder=scans")).json() == {
        "points": [],
        "photos": 0,
        "max": 0,
        "decimals": 3,
        "bounds": None,
    }


def test_heat_points_get_coarser_for_huge_libraries(db):
    from .helpers import add_image

    with db.write() as s:
        for n in range(60):
            add_image(s, n, lat=10 + n * 0.01, lon=20.0)  # 60 cells 1 km apart
    with db.session() as s:
        fine = heat_points(s, ImageFilter())
        coarse = heat_points(s, ImageFilter(), max_points=10)
    assert len(fine["points"]) == 60 and fine["decimals"] == 3
    assert len(coarse["points"]) <= 10 and coarse["decimals"] < 3
    assert coarse["photos"] == fine["photos"] == 60


def test_places_api(client):
    places = ok(client.get("/api/geo/places")).json()["places"]
    assert [(p["name"], p["count"]) for p in places][:2] == [
        ("Berlin, Germany", 3),
        ("Paris, France", 2),
    ]
    berlin = places[0]
    assert set(berlin) == {"id", "name", "lat", "lon", "radius_m", "user_named", "count"}
    detail = ok(client.get(f"/api/geo/places/{berlin['id']}?folder=home")).json()
    assert detail["count"] == 3 and len(detail["images"]) == 3
    assert detail["gallery"] == f"/images?folder=home&place={berlin['id']}"
    assert (detail["locality"], detail["country"]) == ("Berlin", "Germany")
    only_trip = ok(client.get("/api/geo/places?folder=trip")).json()["places"]
    assert [p["name"] for p in only_trip] == ["Paris, France"]
    assert client.get("/api/geo/places/999999").status_code == 404


# --------------------------------------------------------------------------- pages
def test_map_page_renders(client):
    page = ok(client.get("/map")).text
    assert 'id="map"' in page and "/static/vendor/leaflet/leaflet.js" in page
    assert "/static/vendor/leaflet.heat/leaflet-heat.js" in page
    assert 'data-points="/api/geo/points"' in page
    assert 'data-tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png"' in page
    assert "OpenStreetMap&lt;/a&gt; contributors" in page  # attribution (in an attribute)
    assert "Photos in this area" in page and "Find places" in page
    assert "Berlin, Germany" in page and "Paris, France" in page  # the places list
    assert "6 photos with a GPS position" in page
    assert 'id="map-i18n"' in page
    assert 'href="/map"' in page and 'aria-current="page"' in page  # nav link, active

    page = ok(client.get("/map?folder=trip")).text
    assert 'data-points="/api/geo/points?folder=trip"' in page
    assert "Paris, France" in page and "Berlin, Germany" not in page
    assert "2 photos match the filter" in page

    page = ok(client.get("/map?folder=scans")).text
    assert 'id="map"' not in page and "has a GPS position" in page  # empty state

    with client.ctx.db.session() as s:
        image_id = s.scalar(select(Image.id).where(Image.filename == "IMG_4.jpg"))
    page = ok(client.get(f"/map?photo={image_id}")).text
    assert f'data-photo-id="{image_id}"' in page and 'data-photo-lat="48.858' in page


def test_map_page_without_tiles(client):
    client.ctx.settings.geo.tile_url = ""
    page = ok(client.get("/map")).text
    assert 'data-tiles=""' in page and "world-110m.json" in page


def test_photo_page_shows_place(client):
    berlin = _places(client)["Berlin, Germany"]
    with client.ctx.db.session() as s:
        image_id = s.scalar(select(Image.id).where(Image.filename == "IMG_1.jpg"))
        plain = s.scalar(select(Image.id).where(Image.filename == "IMG_7.jpg"))
    page = ok(client.get(f"/images/{image_id}")).text
    assert f'href="/images?place={berlin.id}"' in page and "Berlin, Germany" in page
    assert f'href="/map?photo={image_id}"' in page and "52.51630, 13.37770" in page
    assert "Open on map" not in ok(client.get(f"/images/{plain}")).text


def test_map_script_strings_are_translatable():
    import re

    from vision_tagger.web.common import STATIC_DIR
    from vision_tagger.web.routes.geo import MAP_JS_KEYS

    used = set(re.findall(r'\bt\("([^"]+)"', (STATIC_DIR / "map.js").read_text("utf-8")))
    assert used == set(MAP_JS_KEYS)


def test_vendored_assets_are_served(client):
    for path in (
        "/static/vendor/leaflet/leaflet.js",
        "/static/vendor/leaflet/leaflet.css",
        "/static/vendor/leaflet/LICENSE",
        "/static/vendor/leaflet.heat/leaflet-heat.js",
        "/static/vendor/leaflet.heat/LICENSE",
        "/static/vendor/natural-earth/world-110m.json",
        "/static/map.js",
        "/static/map.css",
    ):
        ok(client.get(path))


# --------------------------------------------------------------------------- actions
def test_rename_place(client):
    berlin = _places(client)["Berlin, Germany"]
    r = client.post(
        f"/places/{berlin.id}/rename",
        data={"name": "Home"},
        headers={"referer": "http://testserver/map?folder=home"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/map?folder=home&msg=")
    assert _places(client)["Home"].user_named
    page = ok(client.get("/images?t=place:Home")).text  # the property follows the name
    assert "3 images" in page

    r = client.post(
        f"/places/{berlin.id}/rename",
        data={"name": ""},
        headers={"accept": "application/json"},
    )
    assert ok(r).json() == {"id": berlin.id, "name": "Berlin, Germany", "user_named": False}
    assert client.post("/places/999999/rename", data={"name": "x"}).status_code == 404
    assert client.get(f"/places/{berlin.id}/rename").status_code == 405  # POST only


def test_rename_is_protected_against_csrf(client):
    berlin = _places(client)["Berlin, Germany"]
    r = client.post(
        f"/places/{berlin.id}/rename",
        data={"name": "pwned"},
        headers={"origin": "https://evil.example"},
    )
    assert r.status_code == 403
    r = client.post(
        f"/places/{berlin.id}/rename",
        data={"name": "pwned"},
        headers={"sec-fetch-site": "cross-site"},
    )
    assert r.status_code == 403
    assert "pwned" not in _places(client)
    r = client.post(  # the map page itself (same origin) may
        f"/places/{berlin.id}/rename",
        data={"name": "Home"},
        headers={"origin": "http://testserver", "accept": "application/json"},
    )
    assert ok(r).json()["name"] == "Home"


def test_place_names_are_escaped(client):
    berlin = _places(client)["Berlin, Germany"]
    evil = '<img src=x onerror="alert(1)">'
    client.post(f"/places/{berlin.id}/rename", data={"name": evil})
    for url in ("/map", f"/images?place={berlin.id}", "/images"):
        page = ok(client.get(url)).text
        assert evil not in page and "&lt;img src=x" in page, url
    assert ok(client.get("/api/geo/places")).json()["places"][0]["name"] == evil  # JSON: raw


def test_find_places_job(client):
    with client.ctx.db.write() as s:
        s.execute(Image.__table__.update().values(place_id=None))
    page = ok(client.get("/map")).text
    assert "6 photos with a GPS position have no place yet." in page
    r = client.post("/places/find", data={"refresh": "1"}, follow_redirects=False)
    assert r.status_code == 303
    for _ in range(100):
        if not client.ctx.jobs.busy:
            break
        time.sleep(0.05)
    job = client.ctx.jobs.history[0]
    assert job.name == "places" and job.status == "done", job.message
    assert job.message.startswith("Places: 3")
    assert "have no place yet" not in ok(client.get("/map")).text
