from __future__ import annotations

import json
import re
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from vision_tagger.indexer import Indexer
from vision_tagger.web.app import create_app
from vision_tagger.web.routes.images import _month, _month_range

from .conftest import make_image


def test_month_helpers():
    assert _month("2019") == "2019-01" and _month("2019", end=True) == "2019-12"
    assert _month("2019-07") == "2019-07" and _month("2019-07-14", end=True) == "2019-07"
    assert _month(None) is None and _month("abc") is None
    assert _month_range("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]
    assert len(_month_range("1900-01", "2024-12")) == 1200  # capped to the last 100 years


@pytest.fixture
def client(settings, photos):
    for i, when in enumerate(
        [datetime(2018, 3, 1), datetime(2018, 3, 9), datetime(2019, 1, 5), datetime(2020, 6, 1)]
    ):
        make_image(photos / ("trip" if i < 2 else "home") / f"p{i}.jpg", seed=i, date=when)
    app = create_app(settings)
    with TestClient(app) as c:
        Indexer(app.state.ctx.db, settings).run()
        yield c


def timeline(page: str) -> dict:
    m = re.search(r'<script type="application/json" id="timeline-data">(.*?)</script>', page)
    assert m, "timeline missing"
    return json.loads(m.group(1))


def test_timeline_data_ignores_dates_but_keeps_other_filters(client):
    data = timeline(client.get("/images").text)
    assert data["months"][0] == "2018-03" and data["months"][-1] == "2020-06"
    assert len(data["months"]) == 28 and sum(data["counts"]) == 4
    assert data["counts"][0] == 2 and data["from"] is None and data["base"] == ""

    data = timeline(client.get("/images?from=2018&to=2019-01&folder=home").text)
    assert data["from"] == "2018-01" and data["to"] == "2019-01"
    assert data["base"] == "folder=home"  # the dates are the slider's part
    assert data["months"] == _month_range("2019-01", "2020-06")  # only "home" photos counted
    page = client.get("/images?from=2018&to=2019-01&folder=home").text
    assert "x-data=\"vtTimeline('timeline-data')\"" in page and "onpointer" not in page


def test_no_timeline_without_dated_photos(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        assert 'id="timeline-data"' not in c.get("/images").text
