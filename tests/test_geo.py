"""Places: offline names (GeoNames), the verified download, clustering, OpenStreetMap names
(mocked Nominatim), user names surviving re-clustering. No network access anywhere."""

from __future__ import annotations

import hashlib
import io
import itertools
import tarfile
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from vision_tagger.db.models import (
    GeocodeCache,
    Image,
    ImageProperty,
    Place,
    PlaceSource,
    StageStatus,
    Status,
)
from vision_tagger.geo import coords_name, geo_dir
from vision_tagger.geo import gazetteer as gz
from vision_tagger.geo.nominatim import (
    GeocoderUnavailable,
    Nominatim,
    RateLimiter,
    osm_name,
    user_agent,
)
from vision_tagger.geo.places import cluster_positions, find_places, needs_update, rename_place

from .helpers import add_image

# --------------------------------------------------------------------------- test data
# (geonameid, name, lat, lon, feature code, country, admin1, population)
TOWNS = [
    (2950159, "Berlin", 52.52437, 13.41053, "PPLC", "DE", "16", 3426354),
    (2930544, "Kreuzberg", 52.49973, 13.40338, "PPLX", "DE", "16", 147532),  # city section
    (2950000, "Alt-Cölln", 52.5150, 13.4000, "PPLH", "DE", "16", 0),  # historical
    (2852458, "Potsdam", 52.39886, 13.06566, "PPLA", "DE", "11", 183154),
    (2988507, "Paris", 48.85341, 2.3488, "PPLC", "FR", "11", 2138551),
    (3031137, "Boulogne-Billancourt", 48.83545, 2.24128, "PPL", "FR", "11", 121334),
    (2953301, "Schwangau", 47.57722, 10.73416, "PPLA4", "DE", "02", 3328),
    (2924360, "Füssen", 47.57143, 10.70171, "PPL", "DE", "02", 15000),
    (5128581, "New York City", 40.71427, -74.00597, "PPL", "US", "NY", 8175133),
    (1880252, "Singapore", 1.28967, 103.85007, "PPLC", "SG", "01", 3547809),
]

BERLIN = (52.5163, 13.3777)  # Brandenburg Gate
KREUZBERG = (52.4986, 13.4030)
EIFFEL = (48.8584, 2.2945)
NEUSCHWANSTEIN = (47.5576, 10.7498)


def geonames_line(t) -> str:
    gid, name, lat, lon, fcode, cc, admin, pop = t
    cols = [str(gid), name, name, "", str(lat), str(lon), "P", fcode, cc, "", admin, "", "", ""]
    cols += [str(pop), "", "0", "Europe/Berlin", "2024-01-01"]
    return "\t".join(cols)


def write_geonames(path: Path) -> Path:
    lines = [geonames_line(t) for t in TOWNS]
    lines.append("1\tMount Nowhere\tMount Nowhere\t\t10\t10\tT\tMT\tXX\t\t\t\t\t\t0\t\t\t\t")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def towns(settings, tmp_path):
    """Offline names from a tiny GeoNames dump (``geo.cities_file``)."""
    settings.geo.cities_file = write_geonames(tmp_path / "cities500.txt")
    return settings


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def encode_pbf(towns) -> bytes:
    """``cities.pbf`` the way the all-the-cities package stores it."""

    def string(tag: int, value: str) -> bytes:
        raw = value.encode()
        return _varint(tag << 3 | 2) + _varint(len(raw)) + raw

    def number(tag: int, value: int) -> bytes:
        return _varint(tag << 3) + _varint(value)

    out = bytearray()
    prev_lat = prev_lon = 0
    for gid, name, lat, lon, fcode, cc, admin, pop in towns:
        lat_i, lon_i = round(lat * 1e5), round(lon * 1e5)
        msg = (
            number(1, _zigzag(gid))
            + string(2, name)
            + string(3, cc)
            + string(7, fcode)
            + string(8, admin)
            + number(9, pop)
            + number(10, _zigzag(lon_i - prev_lon))
            + number(11, _zigzag(lat_i - prev_lat))
        )
        prev_lat, prev_lon = lat_i, lon_i
        out += _varint(len(msg)) + msg
    return bytes(out)


def fake_package() -> bytes:
    pbf = encode_pbf(TOWNS)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(gz.SOURCE_MEMBER)
        info.size = len(pbf)
        tf.addfile(info, io.BytesIO(pbf))
    return buf.getvalue()


# --------------------------------------------------------------------------- offline names
def test_offline_names_from_a_geonames_dump(towns):
    gaz = gz.load_gazetteer(towns)
    assert gaz is not None
    assert len(gaz) == len(TOWNS) - 2  # no city sections, no historical places, no mountains
    assert gaz.describe(*BERLIN).name == "Berlin, Germany"
    named = gaz.describe(*KREUZBERG)  # the section is skipped: the city's name
    assert (named.name, named.locality, named.country) == ("Berlin, Germany", "Berlin", "Germany")
    assert named.source == PlaceSource.GEONAMES
    # Boulogne is nearer to the Eiffel Tower, but Paris is much bigger
    assert gaz.describe(*EIFFEL).name == "Paris, France"
    # ... while a photo in a village keeps the village's name
    assert gaz.describe(*NEUSCHWANSTEIN).name == "Schwangau, Germany"
    assert gaz.describe(40.6892, -74.0445).name == "New York City, NY"  # US: state code
    assert gaz.describe(1.2834, 103.8607).name == "Singapore"  # city state
    assert gaz.describe(40.0, -40.0) is None  # mid-Atlantic: nothing within reach
    # converted once, then loaded from disk (and memory)
    stored = geo_dir(towns) / gz.LOCAL_FILE
    assert stored.exists()
    assert gz.load_gazetteer(towns) is gaz


def test_offline_names_from_a_zipped_dump(settings, tmp_path):
    source = write_geonames(tmp_path / "cities500.txt")
    with zipfile.ZipFile(tmp_path / "cities500.zip", "w") as zf:
        zf.writestr("readme.txt", "GeoNames")
        zf.write(source, "cities500.txt")
    settings.geo.cities_file = tmp_path / "cities500.zip"
    gaz = gz.load_gazetteer(settings)
    assert gaz is not None and gaz.describe(*EIFFEL).name == "Paris, France"


def test_no_names_without_data(settings):
    assert gz.load_gazetteer(settings) is None  # download switched off in the tests
    settings.geo.cities_file = Path("/does/not/exist.txt")
    assert gz.load_gazetteer(settings) is None  # logged, never raised


def test_download_is_verified_before_use(settings, monkeypatch):
    package = fake_package()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=package)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings.geo.download_gazetteer = True
    target = geo_dir(settings) / gz.DEFAULT_FILE

    # the real checksum does not match the fake package: refused, nothing stored
    assert gz.load_gazetteer(settings, client=client) is None
    assert calls == [gz.SOURCE_URL] and not target.exists()

    monkeypatch.setattr(gz, "SOURCE_SIZE", len(package) + 1)  # wrong size: refused too
    monkeypatch.setattr(gz, "SOURCE_SHA256", hashlib.sha256(package).hexdigest())
    assert gz.load_gazetteer(settings, client=client) is None and not target.exists()

    monkeypatch.setattr(gz, "SOURCE_SIZE", len(package))
    gaz = gz.load_gazetteer(settings, client=client)
    assert gaz is not None and target.exists()
    assert gaz.describe(*KREUZBERG).name == "Berlin, Germany"
    assert gaz.describe(40.6892, -74.0445).name == "New York City, NY"
    assert len(calls) == 3
    assert gz.load_gazetteer(settings, client=client) is gaz  # downloaded once
    assert len(calls) == 3


def test_parse_pbf_rejects_truncated_data():
    data = encode_pbf(TOWNS[:2])
    assert [r[0] for r in gz.parse_pbf(data)] == ["Berlin", "Kreuzberg"]
    with pytest.raises(ValueError):
        list(gz.parse_pbf(data[:-3]))


# --------------------------------------------------------------------------- clustering
def test_cluster_positions_groups_splits_and_wraps():
    rng = np.random.default_rng(0)
    home = rng.normal(0, 0.002, size=(300, 2)) + BERLIN  # ~200 m spread
    trip = rng.normal(0, 0.002, size=(40, 2)) + EIFFEL
    alone = np.array([[64.1466, -21.9426]])  # one photo in Reykjavik
    road = np.column_stack([47.0 + np.arange(200) * 0.003, np.full(200, 11.0)])  # 66 km
    dateline = np.array([[-16.5, 179.9995], [-16.5, -179.9995]])  # 100 m apart
    pts = np.vstack([home, trip, alone, road, dateline])
    groups = cluster_positions(pts[:, 0], pts[:, 1], eps_km=0.75, max_km=5.0)
    assert sorted(i for g in groups for i in g) == list(range(len(pts)))  # all, once
    by_point = {int(i): gi for gi, g in enumerate(groups) for i in g}
    assert len({by_point[i] for i in range(300)}) == 1  # home: one place
    assert len({by_point[i] for i in range(300, 340)}) == 1
    assert by_point[0] != by_point[300] != by_point[340]
    road_groups = {by_point[i] for i in range(341, 541)}
    assert 6 <= len(road_groups) <= 16  # split into parts of at most 5 km radius
    for gi in road_groups:
        lats = pts[groups[gi], 0]
        assert (lats.max() - lats.min()) * 111.32 <= 10.5
    assert by_point[541] == by_point[542]  # across the date line


# --------------------------------------------------------------------------- places
def _photos(s, positions, start=0):
    out = []
    for n, (lat, lon) in enumerate(positions, start):
        img = add_image(s, n, lat=lat, lon=lon)
        s.add(StageStatus(image_id=img.id, stage="embed", status=Status.DONE))
        out.append(img.id)
    return out


def _props(db, image_id):
    with db.session() as s:
        return dict(
            s.execute(
                select(ImageProperty.key, ImageProperty.value).where(
                    ImageProperty.image_id == image_id, ImageProperty.source == "place"
                )
            ).all()
        )


def _place_of(db, image_id):
    with db.session() as s:
        pid = s.scalar(select(Image.place_id).where(Image.id == image_id))
        return s.get(Place, pid) if pid else None


def test_find_places_names_assigns_and_keeps_ids(db, towns):
    rng = np.random.default_rng(1)
    with db.write() as s:
        berlin = _photos(s, rng.normal(0, 0.001, (6, 2)) + BERLIN)
        paris = _photos(s, rng.normal(0, 0.001, (3, 2)) + EIFFEL, start=10)
        add_image(s, 20)  # no GPS
    assert needs_update(db, towns)
    result = find_places(db, towns)
    assert (result.places, result.new, result.photos, result.changed) == (2, 2, 9, 9)
    home = _place_of(db, berlin[0])
    assert home.name == "Berlin, Germany" and home.source == PlaceSource.GEONAMES
    assert 0 < home.radius_m < 750
    assert _place_of(db, paris[0]).name == "Paris, France"
    assert _props(db, berlin[0]) == {
        "place": "Berlin, Germany",
        "city": "Berlin",
        "country": "Germany",
    }
    with db.session() as s:  # the search documents mention the place: re-embedded
        pending = s.scalars(
            select(StageStatus.image_id).where(
                StageStatus.stage == "embed", StageStatus.status == Status.PENDING
            )
        ).all()
    assert sorted(pending) == sorted(berlin + paris)

    assert not needs_update(db, towns)
    assert find_places(db, towns).skipped

    with db.write() as s:  # a new photo in Berlin joins the place, ids stay
        new = _photos(s, [(BERLIN[0] + 0.001, BERLIN[1])], start=30)[0]
    result = find_places(db, towns)
    assert (result.places, result.new, result.changed) == (2, 0, 1)
    assert _place_of(db, new).id == home.id

    result = find_places(db, towns, refresh=True)  # regroup and rename: nothing moves
    assert (result.new, result.removed, result.changed) == (0, 0, 0)
    assert _place_of(db, berlin[0]).id == home.id

    with db.write() as s:  # GPS removed from a photo (e.g. in another app)
        s.get(Image, berlin[0]).lat = None
        s.get(Image, berlin[0]).lon = None
        for image_id in paris:
            s.get(Image, image_id).lat = None
    result = find_places(db, towns)
    assert result.removed == 1  # Paris has no photos left
    assert _place_of(db, berlin[0]) is None and _props(db, berlin[0]) == {}
    with db.session() as s:
        assert [p.name for p in s.scalars(select(Place))] == ["Berlin, Germany"]


def test_neighbouring_places_with_the_same_name_merge(db, towns):
    # a walk through Berlin, a photo every 1.3 km: too far apart for one group, but one town
    walk = [(52.49 + k * 0.012, 13.40) for k in range(6)]
    far = [(52.52 + k * 0.1, 13.70) for k in range(2)]  # 11 km apart: stays two places
    with db.write() as s:
        ids = _photos(s, walk + far)
    result = find_places(db, towns)
    places = {_place_of(db, i).id for i in ids[:6]}
    assert len(places) == 1 and _place_of(db, ids[0]).name == "Berlin, Germany"
    assert _place_of(db, ids[0]).radius_m <= towns.geo.max_place_km * 1000
    assert _place_of(db, ids[6]).id != _place_of(db, ids[7]).id
    assert result.places == 3
    assert find_places(db, towns, refresh=True).changed == 0  # stable


def test_places_without_name_data_use_coordinates(db, settings):
    with db.write() as s:
        image_id = _photos(s, [BERLIN])[0]
    find_places(db, settings)
    place = _place_of(db, image_id)
    assert place.source == PlaceSource.COORDS
    assert place.name == coords_name(*BERLIN) == "52.516° N, 13.378° E"
    assert _props(db, image_id) == {"place": place.name}
    assert needs_update(db, settings)  # named again once the data is there


def test_user_names_survive_regrouping(db, towns):
    rng = np.random.default_rng(2)
    with db.write() as s:
        ids = _photos(s, rng.normal(0, 0.002, (8, 2)) + BERLIN)
    find_places(db, towns)
    place = _place_of(db, ids[0])
    rename_place(db, towns, place.id, "  Home  sweet home ")
    renamed = _place_of(db, ids[0])
    assert (renamed.name, renamed.user_named, renamed.source) == (
        "Home sweet home",
        True,
        PlaceSource.USER,
    )
    assert _props(db, ids[-1])["place"] == "Home sweet home"

    # stricter grouping would tear the photos apart - the named place keeps them all
    towns.geo.cluster_km = 0.01
    find_places(db, towns, refresh=True)
    for image_id in ids:
        assert _place_of(db, image_id).id == place.id
    assert _place_of(db, ids[0]).name == "Home sweet home"
    with db.write() as s:  # a later photo nearby joins it
        later = _photos(s, [(BERLIN[0] + 0.0005, BERLIN[1])], start=40)[0]
    find_places(db, towns)
    assert _place_of(db, later).id == place.id
    assert _props(db, later)["place"] == "Home sweet home"

    rename_place(db, towns, place.id, "")  # empty: back to the automatic name
    back = _place_of(db, ids[0])
    assert (back.name, back.user_named, back.source) == (
        "Berlin, Germany",
        False,
        PlaceSource.GEONAMES,
    )
    assert _props(db, later)["place"] == "Berlin, Germany"
    with pytest.raises(LookupError):
        rename_place(db, towns, 9999, "x")


def test_property_page_renames_are_respected(db, towns):
    from vision_tagger.properties import merge_values

    with db.write() as s:
        image_id = _photos(s, [BERLIN])[0]
    find_places(db, towns)
    with db.write() as s:
        merge_values(s, "place", ["Berlin, Germany"], "Hauptstadt")
    find_places(db, towns, refresh=True)
    assert _props(db, image_id)["place"] == "Hauptstadt"


# --------------------------------------------------------------------------- OpenStreetMap
class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


PRENZLAUER_BERG = {
    "licence": "Data © OpenStreetMap contributors, ODbL 1.0.",
    "display_name": "Kollwitzplatz, Prenzlauer Berg, Pankow, Berlin, 10405, Deutschland",
    "address": {
        "suburb": "Prenzlauer Berg",
        "city_district": "Pankow",
        "city": "Berlin",
        "state": "Berlin",
        "country": "Deutschland",
        "country_code": "de",
    },
}


def nominatim_server(answers=None, status=200):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status != 200:
            return httpx.Response(status, text="blocked")
        lat = request.url.params["lat"]
        return httpx.Response(200, json=(answers or {}).get(lat, PRENZLAUER_BERG))

    return httpx.MockTransport(handler), requests


def test_nominatim_rate_limit_user_agent_and_cache(db, settings):
    transport, requests = nominatim_server()
    fake = FakeClock()
    geo = Nominatim(
        "https://nominatim.example",
        language="de",
        limiter=RateLimiter(1.0, clock=fake.clock, sleep=fake.sleep),
        transport=transport,
    )
    times = []
    real_get = geo.client.get

    def timed_get(*args, **kwargs):
        times.append(fake.now)
        fake.now += 0.2  # each answer takes 200 ms
        return real_get(*args, **kwargs)

    geo.client.get = timed_get
    for lat in (52.5401, 52.6, 52.7):
        assert osm_name(geo.lookup(db, lat, 13.41)).name == "Prenzlauer Berg, Berlin"
    assert len(requests) == 3
    assert all(b - a >= 1.0 for a, b in itertools.pairwise(times))  # 1 request/s
    assert fake.slept == pytest.approx([1.0, 1.0])  # a full second after each answer
    first = requests[0]
    assert first.headers["user-agent"] == user_agent()
    assert first.headers["user-agent"].startswith("vision-tagger/")
    assert "python-httpx" not in first.headers["user-agent"]
    assert first.headers["accept-language"] == "de"
    assert first.url.path == "/reverse"
    assert dict(first.url.params) == {
        "format": "jsonv2",
        "lat": "52.540",  # rounded to ~100 m before it leaves the computer
        "lon": "13.410",
        "zoom": "14",
        "addressdetails": "1",
        "accept-language": "de",
    }

    # cached in the database: the same (rounded) position is never asked again
    assert osm_name(geo.lookup(db, 52.54004, 13.40996)).name == "Prenzlauer Berg, Berlin"
    assert len(requests) == 3
    with db.session() as s:
        assert s.get(GeocodeCache, geo.cache_key(52.54, 13.41)).result == PRENZLAUER_BERG
    geo.lookup(db, 52.54, 13.41, refresh=True)
    assert len(requests) == 4
    geo.close()


def test_nominatim_no_address_and_refusals(db, settings):
    transport, requests = nominatim_server({"40.000": {"error": "Unable to geocode"}})
    geo = Nominatim(limiter=RateLimiter(0.0), transport=transport)
    assert geo.lookup(db, 40.0, -40.0) is None
    assert geo.lookup(db, 40.0, -40.0) is None  # "nothing here" is cached too
    assert len(requests) == 1

    transport, requests = nominatim_server(status=403)
    blocked = Nominatim(limiter=RateLimiter(0.0), transport=transport)
    with pytest.raises(GeocoderUnavailable):
        blocked.lookup(db, 1.0, 2.0)
    with db.session() as s:  # failures are not cached
        assert s.get(GeocodeCache, blocked.cache_key(1.0, 2.0)) is None


def test_osm_name_formats():
    assert osm_name(PRENZLAUER_BERG).name == "Prenzlauer Berg, Berlin"
    town = osm_name({"address": {"town": "Füssen", "state": "Bayern", "country": "Germany"}})
    assert (town.name, town.locality, town.admin, town.country, town.source) == (
        "Füssen, Germany",
        "Füssen",
        "Bayern",
        "Germany",
        PlaceSource.OSM,
    )
    assert osm_name({"address": {"city": "Monaco", "country": "Monaco"}}).name == "Monaco"
    assert osm_name({"name": "Mont Blanc", "address": {}}).name == "Mont Blanc"
    assert osm_name({"display_name": "Atlantic Ocean, Somewhere"}).name == (
        "Atlantic Ocean, Somewhere"
    )
    assert osm_name({}) is None and osm_name(None) is None


def test_find_places_with_openstreetmap_names(db, towns):
    with db.write() as s:
        ids = _photos(s, [BERLIN, EIFFEL])
    transport, requests = nominatim_server(
        {"48.858": {"address": {"suburb": "Gros-Caillou", "city": "Paris", "country": "France"}}}
    )
    geo = Nominatim(limiter=RateLimiter(0.0), transport=transport)
    towns.geo.nominatim = True
    result = find_places(db, towns, geocoder=geo)
    assert result.osm_requests == 2
    assert _place_of(db, ids[0]).name == "Prenzlauer Berg, Berlin"
    assert _place_of(db, ids[0]).source == PlaceSource.OSM
    assert _place_of(db, ids[1]).name == "Gros-Caillou, Paris"
    assert _props(db, ids[1]) == {
        "place": "Gros-Caillou, Paris",
        "city": "Paris",
        "country": "France",
    }
    assert not needs_update(db, towns)
    find_places(db, towns, refresh=True, geocoder=geo)  # refresh asks again
    assert len(requests) == 4


def test_blocked_nominatim_falls_back_to_offline_names(db, towns):
    with db.write() as s:
        ids = _photos(s, [BERLIN, EIFFEL, NEUSCHWANSTEIN])
    transport, requests = nominatim_server(status=429)
    towns.geo.nominatim = True
    result = find_places(
        db, towns, geocoder=Nominatim(limiter=RateLimiter(0.0), transport=transport)
    )
    assert len(requests) == 1  # no further requests after the first refusal
    assert [_place_of(db, i).name for i in ids] == [
        "Berlin, Germany",
        "Paris, France",
        "Schwangau, Germany",
    ]
    assert any("OpenStreetMap" in note for note in result.notes)
    assert needs_update(db, towns)  # OpenStreetMap names are still wanted


def test_index_run_finds_places(db, towns, photos):
    from vision_tagger.indexer import Indexer

    from .conftest import make_image

    make_image(photos / "berlin.jpg", seed=1, gps=BERLIN)
    make_image(photos / "paris.jpg", seed=2, gps=EIFFEL)
    make_image(photos / "nowhere.jpg", seed=3)
    stats = Indexer(db, towns).run()
    assert stats.places.startswith("Places: 2 (2 new, 0 removed), 2 photos with GPS")
    with db.session() as s:
        names = {
            i.filename: s.get(Place, i.place_id).name if i.place_id else None
            for i in s.scalars(select(Image))
        }
    assert names == {
        "berlin.jpg": "Berlin, Germany",
        "paris.jpg": "Paris, France",
        "nowhere.jpg": None,
    }
    assert Indexer(db, towns).run().places == "Places: up to date"
    towns.geo.auto = False
    assert Indexer(db, towns).run().places == ""


def test_cli_places(db, towns, monkeypatch):
    from vision_tagger import config as config_mod
    from vision_tagger.cli import app

    with db.write() as s:
        _photos(s, [BERLIN, EIFFEL])
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.setenv("VT_PATHS__DATA_DIR", str(towns.data_dir))
    monkeypatch.setenv("VT_GEO__DOWNLOAD_GAZETTEER", "false")
    monkeypatch.setenv("VT_GEO__CITIES_FILE", str(towns.geo.cities_file))
    runner = CliRunner()
    r = runner.invoke(app, ["places"])
    assert r.exit_code == 0, r.output
    assert "Places: 2 (2 new, 0 removed), 2 photos with GPS" in r.output
    r = runner.invoke(app, ["places"])
    assert "up to date" in r.output
    r = runner.invoke(app, ["places", "--refresh"])
    assert "Places: 2 (0 new, 0 removed)" in r.output
