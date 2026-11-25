from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vision_tagger.db.models import Cluster, Face, Person, PropertyAlias
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.faces import FacesStage
from vision_tagger.pipeline.stages.vlm import VlmStage
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app

from .conftest import make_image
from .test_faces_stage import FakeAnalyzer


@pytest.fixture
def client(settings, photos, tmp_path):
    for i in range(6):
        make_image(photos / ("work" if i % 2 else "trip") / f"IMG_2021010{i}_120000.jpg", seed=i)
    app = create_app(settings)
    with TestClient(app) as c:
        ctx = app.state.ctx
        Indexer(ctx.db, settings).run()
        stages = {
            "faces": FacesStage(ctx.db, settings, analyzer=FakeAnalyzer()),
            "vlm": VlmStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
            "embed": EmbedStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
        }
        PipelineRunner(ctx.db, settings, stages.__getitem__).run()
        c.ctx = ctx
        yield c


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def test_pages_render(client):
    for url in [
        "/",
        "/partials/status",
        "/partials/health",
        "/faces",
        "/faces?show=unnamed&sort=unnamed",
        "/faces/unclustered",
        "/faces/ignored",
        "/persons",
        "/properties",
        "/properties/location_type",
        "/images",
        "/images?faces=yes&sort=faces",
        "/images/page?page=1",
        "/images/1",
        "/overview",
        "/overview?from=2021&to=2021",
        "/overview?from=2021-01&to=2021-01",
        "/overview/table?order=size&desc=0",
        "/partials/value-options?key=objects",
        "/api/status",
        "/api/clusters/search?q=",
    ]:
        ok(client.get(url))


def test_media_endpoints(client):
    assert ok(client.get("/media/thumb/1")).headers["content-type"] == "image/webp"
    ok(client.get("/media/image/1"))
    with client.ctx.db.session() as s:
        face_id = s.scalar(select(Face.id))
    ok(client.get(f"/media/face/{face_id}"))
    assert client.get("/media/thumb/9999").status_code == 404


def test_cluster_workflow(client):
    db = client.ctx.db
    with db.session() as s:
        clusters = list(s.scalars(select(Cluster.id).order_by(Cluster.id)))
    assert len(clusters) == 2
    a, b = clusters
    # inline rename via HTMX returns the card
    r = ok(
        client.post(
            f"/clusters/{a}/name",
            data={"name": "Anna", "card": "1"},
            headers={"HX-Request": "true"},
        )
    )
    assert "Anna" in r.text
    ok(client.get(f"/clusters/{a}"))
    # merge
    client.post("/clusters/merge", data={"cluster_ids": [str(a), str(b)]})
    with db.session() as s:
        assert s.get(Cluster, b) is None
        faces = list(s.scalars(select(Face.id).where(Face.cluster_id == a)))
    # split two faces off and name them
    client.post(
        "/faces/action",
        data={"action": "split", "face_ids": [str(f) for f in faces[:2]], "name": "Ben"},
    )
    with db.session() as s:
        assert {p.name for p in s.scalars(select(Person))} == {"Anna", "Ben"}
    # ignore + restore
    client.post("/faces/action", data={"action": "ignore", "face_ids": [str(faces[2])]})
    ok(client.get("/faces/ignored"))
    client.post("/faces/action", data={"action": "unignore", "face_ids": [str(faces[2])]})
    # assign to person
    client.post(
        "/faces/action", data={"action": "person", "face_ids": [str(faces[3])], "name": "Ben"}
    )
    ok(client.get("/persons"))
    with db.session() as s:
        ben = s.scalar(select(Person).where(Person.name == "Ben"))
    ok(client.get(f"/persons/{ben.id}"))
    ok(client.get(f"/images?p={ben.id}"))
    client.post(f"/persons/{ben.id}/rename", data={"name": "Benjamin"})
    client.post(f"/clusters/{a}/auto-split")


def test_property_merge_and_filters(client):
    db = client.ctx.db
    page = ok(client.get("/properties/objects")).text
    assert "tree" in page or "building" in page
    client.post("/properties/objects/merge", data={"values": ["tree", "car"], "target": "nature"})
    with db.session() as s:
        aliases = {(a.from_value, a.to_value) for a in s.scalars(select(PropertyAlias))}
    assert ("car", "nature") in aliases or ("tree", "nature") in aliases
    r = ok(client.get("/images?t=objects:nature&t=work_image:no&tm=any&xt=year:1999"))
    assert "match the filter" in r.text
    r = client.get("/images/add?key=location_type&value=beach&mode=exclude", follow_redirects=False)
    assert r.status_code == 303 and "xt=location_type" in r.headers["location"]
    client.post("/properties/objects/delete", data={"values": ["nature"]})


def test_dashboard_actions(client, tmp_path):
    ok(client.post("/actions/pause"))
    ok(client.post("/actions/resume"))
    ok(client.post("/actions/stop"))
    ok(client.post("/actions/retry-errors"))
    ok(client.post("/actions/requeue-outdated?stage=vlm"))
    r = client.post("/actions/recluster?full=1")
    assert r.status_code == 200
    _wait_jobs(client)
    r = client.post("/actions/index", data={"roots": str(client.ctx.settings.paths.image_roots[0])})
    assert r.status_code == 200
    _wait_jobs(client)
    assert client.ctx.jobs.history[0].status == "done", client.ctx.jobs.history[0].message
    # export via the UI (background job)
    client.post(
        "/images/export", data={"target": str(tmp_path / "out"), "mode": "copy", "query": ""}
    )
    _wait_jobs(client)
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 6


def test_run_from_ui_with_fake_backends(client):
    # re-queue VLM and run it through the background runner using the fake backend
    ok(client.post("/images/1/requeue", data={"stage": "vlm"}))
    ok(client.post("/actions/run", data={"stages": ["vlm"], "batch_size": "2"}))
    client.ctx.runner.join(20)
    assert client.ctx.runner.last_summary.status == "completed"


def _wait_jobs(client, timeout=20):
    end = time.time() + timeout
    while client.ctx.jobs.busy and time.time() < end:
        time.sleep(0.05)


def test_csrf_and_host_protection(client):
    # foreign Origin -> blocked
    r = client.post("/actions/stop", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/actions/stop", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = client.post("/actions/stop", headers={"Origin": "null"})
    assert r.status_code == 403
    # DNS rebinding: unknown Host -> blocked, even for GET
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    # same-origin form posts keep working
    r = client.post("/actions/stop", headers={"Origin": "http://testserver"})
    assert r.status_code == 200
    # redirects never leave the UI (no open redirect via Referer)
    r = client.post(
        "/actions/stop",
        headers={"Referer": "http://evil.example/x"},
        follow_redirects=False,
    )
    assert r.status_code == 403  # foreign referer counts as cross-site
    r = client.post(
        "/references/999/delete",
        headers={"Referer": "http://testserver/persons?msg=old"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/persons")
    assert "msg=old" not in r.headers["location"]


def test_no_user_data_in_inline_js(client):
    db = client.ctx.db
    from vision_tagger.faces import clustering as C

    with db.write() as s:
        cid = s.scalar(select(Cluster.id))
        C.name_cluster(s, cid, "O'Brien </script>")
        pid = s.scalar(select(Person.id))
    page = ok(client.get(f"/persons/{pid}")).text
    assert "confirm(" not in page
    assert 'data-confirm="Delete O&#39;Brien &lt;/script&gt;?' in page


def test_search_page_and_results(client):
    page = ok(client.get("/")).text
    assert 'class="navsearch"' in page
    page = ok(client.get("/search")).text
    assert "Try:" in page
    page = ok(client.get("/search?q=trees%20and%20cars&deep=1")).text
    assert 'hx-get="/search/results?q=trees+and+cars&amp;deep=1"' in page
    html = ok(client.get("/search/results?q=trees and cars&deep=1")).text
    assert "basic mode" in html  # fake backend: keyword interpretation
    assert "/media/thumb/" in html
    assert ok(client.get("/search/results?q=")).text == ""
    r = client.post(
        "/search/export",
        data={"ids": "1,2", "target": "/tmp/x", "q": "x"},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403  # CSRF protection also covers the search export


def test_quiz_pages_and_actions(client):
    db = client.ctx.db
    from vision_tagger.faces import clustering as C

    ok(client.get("/quiz"))  # fewer than two named persons -> friendly hint
    page = ok(client.get("/quiz?mode=name")).text
    assert "Who is this?" in page
    with db.session() as s:
        cids = list(s.scalars(select(Cluster.id).order_by(Cluster.id)))
    r = ok(client.post("/quiz/name", data={"mode": "name", "cluster_id": cids[0], "name": "Anna"}))
    assert "Named" in r.text
    with db.write() as s:
        C.name_cluster(s, cids[1], "Ben")
    page = ok(client.get("/quiz?mode=quiz")).text
    assert "Who's that?" in page and 'data-key="1"' in page
    with db.session() as s:
        face = s.scalar(select(Face).where(Face.cluster_id == cids[0]))
        ben = s.scalar(select(Person).where(Person.name == "Ben"))
    r = ok(
        client.post(
            "/quiz/answer",
            data={
                "mode": "quiz",
                "face_id": face.id,
                "choice": ben.id,
                "image_id": face.image_id,
                "streak": "4",
            },
        )
    )
    assert "Nope" in r.text and "That's not Anna!" in r.text
    r = ok(
        client.post("/quiz/fix", data={"mode": "quiz", "face_id": face.id, "action": "unassign"})
    )
    assert "Removed" in r.text
    with db.session() as s:
        assert s.get(Face, face.id).cluster_id is None
