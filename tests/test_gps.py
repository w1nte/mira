"""GPS positions: EXIF parsing (all hemispheres) and the indexed lat / lon columns."""

from __future__ import annotations

import math
from datetime import datetime

import pytest
from PIL import ExifTags, Image
from sqlalchemy import select

from vision_tagger.db.models import Image as ImageRow
from vision_tagger.db.models import ImageProperty, Place
from vision_tagger.imaging import _gps_to_decimal, gps_position, read_image_info
from vision_tagger.indexer import Indexer

from .conftest import make_image, set_mtime

QUADRANTS = {
    "north_east.jpg": (52.520008, 13.404954),  # Berlin
    "south_east.jpg": (-33.856784, 151.215297),  # Sydney
    "north_west.jpg": (40.689247, -74.044502),  # New York
    "south_west.jpg": (-22.951916, -43.210487),  # Rio de Janeiro
}


@pytest.mark.parametrize(("name", "pos"), list(QUADRANTS.items()))
def test_exif_gps_signs_in_every_hemisphere(tmp_path, name, pos):
    path = make_image(tmp_path / name, gps=pos)
    gps = read_image_info(path, 0.0).meta["gps"]
    assert gps["lat"] == pytest.approx(pos[0], abs=1e-5)
    assert gps["lon"] == pytest.approx(pos[1], abs=1e-5)


def test_gps_to_decimal_tolerates_odd_writers():
    assert _gps_to_decimal((52.0, 30.0, 0.0), "N") == 52.5
    assert _gps_to_decimal((52.0, 30.0, 0.0), b"S\x00") == -52.5  # NUL-padded bytes
    assert _gps_to_decimal((13.0, 24.0, 0.0), " w ") == -13.4
    assert _gps_to_decimal((-33.5, 0.0, 0.0), "") == -33.5  # negative degrees, no reference
    assert _gps_to_decimal((-33.5, 0.0, 0.0), "S") == -33.5  # both: still south
    assert _gps_to_decimal(48.25, "N") == 48.25  # decimal degrees only
    assert _gps_to_decimal((10.0,), "E") == 10.0
    assert _gps_to_decimal((float("nan"), 0.0, 0.0), "N") is None  # zero denominator
    assert _gps_to_decimal((91.0, 0.0, 0.0), "N", 90.0) is None
    assert _gps_to_decimal((181.0, 0.0, 0.0), "E") is None
    assert _gps_to_decimal(("x", 1, 2), "N") is None
    assert _gps_to_decimal((), "N") is None


def test_gps_altitude_below_sea_level(tmp_path):
    im = Image.new("RGB", (16, 16))
    exif = Image.Exif()
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps[ExifTags.GPS.GPSLatitudeRef], gps[ExifTags.GPS.GPSLatitude] = "N", (31.0, 30.0, 0.0)
    gps[ExifTags.GPS.GPSLongitudeRef], gps[ExifTags.GPS.GPSLongitude] = "E", (35.0, 30.0, 0.0)
    gps[ExifTags.GPS.GPSAltitudeRef], gps[ExifTags.GPS.GPSAltitude] = 1, 430.5  # Dead Sea
    im.save(tmp_path / "dead_sea.jpg", exif=exif)
    meta = read_image_info(tmp_path / "dead_sea.jpg", 0.0).meta
    assert meta["gps"] == {"lat": 31.5, "lon": 35.5, "alt": -430.5}


def test_gps_position_validates():
    assert gps_position({"gps": {"lat": 1.5, "lon": 2}}) == (1.5, 2.0)
    assert gps_position({"gps": {"lat": 0, "lon": 0}}) is None  # "no fix"
    assert gps_position({"gps": {"lat": 95.0, "lon": 2}}) is None
    assert gps_position({"gps": {"lat": "1", "lon": 2}}) is None
    assert gps_position({"gps": {"lat": True, "lon": 2}}) is None
    assert gps_position({"gps": {"lat": math.inf, "lon": 2}}) is None
    assert gps_position({}) is None
    assert gps_position(None) is None


def test_indexer_fills_and_updates_coordinates(db, settings, photos):
    settings.geo.auto = False  # only the columns here (places: test_geo.py)
    berlin = make_image(photos / "berlin.jpg", seed=1, gps=QUADRANTS["north_east.jpg"])
    make_image(photos / "plain.jpg", seed=2)
    Indexer(db, settings).run()
    with db.session() as s:
        rows = {i.filename: i for i in s.scalars(select(ImageRow))}
    assert rows["berlin.jpg"].lat == pytest.approx(52.520008, abs=1e-5)
    assert rows["berlin.jpg"].lon == pytest.approx(13.404954, abs=1e-5)
    assert rows["plain.jpg"].lat is None and rows["plain.jpg"].lon is None

    # a place was assigned meanwhile; the photo is geotagged anew in another app
    with db.write() as s:
        place = Place(name="Berlin, Germany", lat=52.52, lon=13.4)
        s.add(place)
        s.flush()
        img = s.get(ImageRow, rows["berlin.jpg"].id)
        img.place_id = place.id
        s.add(ImageProperty(image_id=img.id, key="place", value=place.name, source="place"))
    # new pixels too (an edit): VLM results go, the place properties (from the position) stay
    make_image(berlin, seed=7, gps=QUADRANTS["south_east.jpg"])
    set_mtime(berlin, datetime(2024, 1, 1))
    Indexer(db, settings).run()
    with db.session() as s:
        img = s.get(ImageRow, rows["berlin.jpg"].id)
        assert img.lat == pytest.approx(-33.856784, abs=1e-5)
        assert img.place_id is None  # to be placed again
        assert s.scalar(select(ImageProperty.value).where(ImageProperty.key == "place"))

    # GPS removed: no position
    make_image(berlin, seed=7)
    set_mtime(berlin, datetime(2024, 2, 1))
    Indexer(db, settings).run()
    with db.session() as s:
        assert s.get(ImageRow, rows["berlin.jpg"].id).lat is None
