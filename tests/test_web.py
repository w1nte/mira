from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vision_tagger.db.models import Cluster, Face, Person, PropertyAlias
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.ages import AgesStage
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.faces import FacesStage
from vision_tagger.pipeline.stages.vlm import VlmStage
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app

from .conftest import make_image
from .helpers import FakeAgeEstimator
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
            "faces": FacesStage(
                ctx.db, settings, analyzer=FakeAnalyzer(), age_estimator=FakeAgeEstimator()
            ),
            "ages": AgesStage(ctx.db, settings, estimator=FakeAgeEstimator()),
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
        "/database",
        "/database/missing",
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


def test_age_ui(client):
    db = client.ctx.db
    from vision_tagger.faces import clustering as C

    with db.write() as s:
        cid = s.scalar(select(Cluster.id).order_by(Cluster.id))
        C.name_cluster(s, cid, "Anna")
        pid = s.scalar(select(Person.id))
        image_id = s.scalar(select(Face.image_id).where(Face.cluster_id == cid))
    page = ok(client.get(f"/persons/{pid}")).text
    assert "Age" in page and "Set birth year" in page
    ok(client.post(f"/persons/{pid}/birth-year", data={"birth_year": "1990"}))
    with db.session() as s:
        assert s.get(Person, pid).birth_year == 1990
    page = ok(client.get(f"/persons/{pid}")).text
    assert "Born <b>1990</b>" in page and "Through the years" in page
    assert "born 1990" in ok(client.get("/persons")).text
    page = ok(client.get(f"/images/{image_id}")).text
    assert "Anna · " in page  # age next to the name on the face box
    ok(client.post(f"/images/{image_id}/date", data={"year": "2005"}))
    assert "manual" in ok(client.get(f"/images/{image_id}")).text
    r = client.post(f"/persons/{pid}/birth-year", data={"birth_year": "abc"})
    assert "Enter a year" in r.text or r.status_code == 200


def test_cached_pages_follow_database_changes(client):
    """Whole-library aggregates are cached, but never shown after the data changed."""
    from vision_tagger.db.models import Image, ImageProperty

    assert "zzfresh" not in ok(client.get("/partials/value-options?key=tags")).text
    assert "6 images" in ok(client.get("/overview")).text
    ok(client.get("/properties"))
    with client.ctx.db.write() as s:
        s.add(ImageProperty(image_id=1, key="tags", value="zzfresh"))
        s.get(Image, 2).missing = True
    assert "zzfresh (1)" in ok(client.get("/partials/value-options?key=tags")).text
    assert "zzfresh" in ok(client.get("/properties?limit=500")).text
    assert "5 images" in ok(client.get("/overview")).text


# --------------------------------------------------------------------------- layout and theme
def _checked_theme(page: str) -> list[str]:
    return re.findall(r'aria-checked="true"[^>]*data-theme-set="([a-z]+)"', page)


def test_theme_cookie_renders_data_theme(client):
    page = ok(client.get("/")).text
    assert '<html lang="en" data-theme="system">' in page
    assert _checked_theme(page) == ["system"]
    for theme in ("dark", "light"):
        client.cookies.set("vt_theme", theme)
        page = ok(client.get("/faces")).text
        assert f'<html lang="en" data-theme="{theme}">' in page
        assert _checked_theme(page) == [theme]
    for invalid in ("purple", "Dark", "system", ""):
        client.cookies.set("vt_theme", invalid)
        page = ok(client.get("/persons")).text
        assert '<html lang="en" data-theme="system">' in page
        assert _checked_theme(page) == ["system"]
    client.cookies.clear()
    assert 'data-theme="system"' in ok(client.get("/images")).text


def test_sidebar_navigation_and_theme_switch(client):
    page = ok(client.get("/")).text
    assert '<aside class="sidebar" id="sidebar">' in page
    for href in ("/", "/search", "/images", "/overview", "/faces", "/persons", "/generations",
                 "/quiz", "/properties"):  # fmt: skip
        assert f'href="{href}"' in page, href
    assert re.search(r'class="navlink active" href="/" aria-current="page"', page)
    assert page.count('aria-current="page"') == 1
    assert 'role="radiogroup" aria-label="Theme"' in page
    assert all(f'data-theme-set="{t}"' in page for t in ("system", "light", "dark"))
    assert 'hx-get="/partials/activity"' in page
    assert '<meta name="color-scheme" content="light dark">' in page
    assert 'href="/static/favicon.svg"' in page
    # app.js defines helpers Alpine components use, so it must load before Alpine
    scripts = re.findall(r'<script src="/static/([a-z.]+)" defer>', page)
    assert scripts == ["htmx.min.js", "app.js", "alpine.min.js"]
    assert ok(client.get("/static/favicon.svg")).headers["content-type"].startswith("image/svg")
    # cluster pages belong to "Faces"
    with client.ctx.db.session() as s:
        cid = s.scalar(select(Cluster.id))
    page = ok(client.get(f"/clusters/{cid}")).text
    assert re.search(r'class="navlink active" href="/faces" aria-current="page"', page)
    assert '<nav class="crumbs" aria-label="Breadcrumb"><a href="/faces">Faces</a>' in page


def test_activity_partial(client):
    from vision_tagger.db.models import Control, Run, utcnow
    from vision_tagger.web.context import Job

    html = ok(client.get("/partials/activity")).text
    assert "Pipeline idle" in html and 'class="activity"' in html
    db = client.ctx.db
    with db.write() as s:
        ctrl = s.get(Control, 1)
        ctrl.active_run_id = s.scalar(select(Run.id))
        ctrl.heartbeat_at = utcnow()
        ctrl.active_stage = "vlm"
        ctrl.desired_state = "running"
    html = ok(client.get("/partials/activity")).text
    assert "Running · Properties" in html and 'class="dot run"' in html
    assert 'aria-valuenow="100"' in html  # all images of the fixture are done
    with db.write() as s:
        s.get(Control, 1).desired_state = "paused"
    html = ok(client.get("/partials/activity")).text
    assert "Paused" in html and 'class="dot pause"' in html
    with db.write() as s:
        s.get(Control, 1).active_run_id = None
    client.ctx.jobs.current = Job(name="index", message="3/10 files analysed")
    html = ok(client.get("/partials/activity")).text
    assert "Index" in html and "3/10 files analysed" in html
    client.ctx.jobs.current = None
    assert "Pipeline idle" in ok(client.get("/partials/activity")).text


def test_ages_stage_is_labelled(client):
    from vision_tagger.db.models import Control, Run, utcnow

    status = ok(client.get("/partials/status")).text
    assert "Face ages" in status and "Stage 1b · AgeNet + InsightFace" in status
    with client.ctx.db.write() as s:
        ctrl = s.get(Control, 1)
        ctrl.active_run_id = s.scalar(select(Run.id))
        ctrl.heartbeat_at = utcnow()
        ctrl.active_stage = "ages"
        ctrl.desired_state = "running"
    assert "Running · Face ages" in ok(client.get("/partials/activity")).text


def test_icon_helper():
    from vision_tagger.web.common import icon

    assert str(icon("check")) == '<svg class="i" aria-hidden="true"><use href="#i-check"/></svg>'
    assert 'class="i sm"' in str(icon("chevron-left", "sm"))
    for bad in ('x"><script>', "", "Check", "x\n"):
        with pytest.raises(ValueError):
            icon(bad)
    with pytest.raises(ValueError):
        icon("check", 'sm" onclick="x')


def test_no_inline_event_handlers_and_icons_exist(client):
    from vision_tagger.faces import clustering as C
    from vision_tagger.web.common import TEMPLATES_DIR

    db = client.ctx.db
    with db.write() as s:
        cids = list(s.scalars(select(Cluster.id).order_by(Cluster.id)))
        C.name_cluster(s, cids[0], "Anna")
        C.name_cluster(s, cids[1], "Ben")
        pid = s.scalar(select(Person.id).where(Person.name == "Anna"))
        face = s.scalar(select(Face).where(Face.cluster_id == cids[0]))
        face_id, image_id = face.id, face.image_id
    pages = [
        "/",
        "/partials/status",
        "/partials/health",
        "/partials/activity",
        "/faces",
        "/faces/unclustered",
        "/faces/ignored",
        f"/clusters/{cids[0]}",
        "/persons",
        f"/persons/{pid}",
        "/images",
        f"/images/{image_id}",
        "/properties",
        "/properties/objects",
        "/overview",
        "/overview/table",
        "/search",
        "/search?q=trees",
        "/search/results?q=trees",
        "/quiz",
        "/quiz?mode=name",
        "/generations",
        "/database",
        "/database/missing",
        f"/generations/{pid}",
        f"/generations/{pid}/compare/{pid + 1}",
    ]
    sprite = (TEMPLATES_DIR / "partials" / "icons.html").read_text()
    symbols = set(re.findall(r'<symbol id="i-([a-z0-9-]+)"', sprite))
    inline_handler = re.compile(r"<[^>]*\son[a-z]+\s*=", re.IGNORECASE)
    for url in pages:
        page = ok(client.get(url)).text
        assert not inline_handler.search(page), (url, inline_handler.search(page).group())
        assert "javascript:" not in page, url
        for name in re.findall(r'<use href="#i-([a-z0-9-]+)"', page):
            assert name in symbols, (url, name)
    answer = ok(
        client.post(
            "/quiz/answer",
            data={"mode": "quiz", "face_id": face_id, "choice": pid, "image_id": image_id},
        )
    ).text
    assert not inline_handler.search(answer)
    # the "open photo" link inside a selectable face is a plain link (no JS needed)
    page = ok(client.get(f"/clusters/{cids[0]}")).text
    assert 'class="open" href="/images/' in page and 'aria-label="Open photo"' in page
