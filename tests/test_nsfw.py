"""NSFW: the property, the classifier stage, the manual override and the covers in the web UI."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from datetime import date, datetime
from html.parser import HTMLParser

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import select

from vision_tagger import library
from vision_tagger.animals.grouping import move_animals_to_name
from vision_tagger.animals.models import ModelFile, ModelUnavailable, ensure_model
from vision_tagger.config import Settings
from vision_tagger.db.models import (
    Animal,
    Cluster,
    Image,
    ImageProperty,
    Person,
    Place,
    StageStatus,
    Status,
)
from vision_tagger.faces import clustering as C
from vision_tagger.indexer import Indexer
from vision_tagger.nsfw import verdict as V
from vision_tagger.nsfw.classifier import NsfwClassifier, nsfw_probability, preprocess
from vision_tagger.nsfw.models import VIT_NSFW, default_model_dir
from vision_tagger.pipeline import control
from vision_tagger.pipeline.runner import BackgroundRunner, PipelineRunner
from vision_tagger.pipeline.stages import STAGE_NAMES, stage_versions
from vision_tagger.pipeline.stages.ages import AgesStage
from vision_tagger.pipeline.stages.animals import AnimalsStage
from vision_tagger.pipeline.stages.base import StageAbort
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.faces import FacesStage
from vision_tagger.pipeline.stages.nsfw import NsfwStage
from vision_tagger.pipeline.stages.visual import VisualStage
from vision_tagger.pipeline.stages.vlm import VlmStage, build_properties
from vision_tagger.query import ImageFilter, count_images
from vision_tagger.search.document import build_document
from vision_tagger.search.interpret import KeywordInterpreter
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.normalize import Normalizer
from vision_tagger.vlm.prompt import system_prompt
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app

from .conftest import make_image
from .helpers import (
    FakeAgeEstimator,
    FakeAnimalAnalyzer,
    FakeNsfwClassifier,
    FakeVisualEncoder,
    add_image,
)
from .test_faces_stage import FakeAnalyzer

DE = {"Accept-Language": "de"}
SHOW = {"Cookie": "vt_nsfw=show"}


def nsfw_rows(db) -> dict[int, tuple[str, str, float | None]]:
    with db.session() as s:
        return {
            p.image_id: (p.value, p.source, p.confidence)
            for p in s.scalars(select(ImageProperty).where(ImageProperty.key == "nsfw"))
        }


def answer(vocab: Vocabulary, **fields) -> dict:
    """A complete VLM answer (every field present) with some fields set."""
    data = FakeBackend(vocab).analyze(PILImage.new("RGB", (8, 8)), "", "", {})
    data.update(fields)
    return data


# --------------------------------------------------------------------------- the verdict
def test_decide_combines_classifier_and_vision_model():
    assert V.decide(None, None, 0.7) is None  # nobody looked yet
    assert V.decide(0.9, None, 0.7) == ("yes", "classifier")
    assert V.decide(0.7, False, 0.7) == ("yes", "classifier")  # at the threshold
    assert V.decide(0.2, True, 0.7) == ("yes", "vlm")  # either of them is enough
    assert V.decide(0.2, False, 0.7) == ("no", "classifier")
    assert V.decide(None, False, 0.7) == ("no", "vlm")
    assert V.decide(0.5, None, 0.4) == ("yes", "classifier")  # a lower threshold: more covers


def test_vocabulary_asks_the_vision_model():
    vocab = Vocabulary.load()
    assert vocab.fields["nsfw"].base_type == "boolean"
    assert vocab.json_schema()["properties"]["nsfw"]["type"] == "boolean"
    prompt = system_prompt(vocab)
    assert "- nsfw:" in prompt and "nudity" in prompt and "gore" in prompt
    # the fake vision model never finds anything NSFW (trying the UI must not cover photos)
    assert (
        FakeBackend(vocab).analyze(PILImage.new("RGB", (8, 8), "white"), "", "", {})["nsfw"]
        is False
    )
    # the stage combines the answer with the classifier instead of storing it as it is
    rows = build_properties(vocab, vocab.coerce(answer(vocab, nsfw=True)), Normalizer(), 10)
    assert all(key != "nsfw" for key, *_ in rows)


def test_property_from_the_vlm_result(db, settings):
    with db.write() as s:
        ids = [add_image(s, n).id for n in range(3)]
    vocab = Vocabulary.load()
    backend = FakeBackend(
        vocab,
        responses=[answer(vocab, nsfw=True), answer(vocab, nsfw=False), answer(vocab, nsfw="yes")],
    )
    stage = VlmStage(db, settings, backend=backend)
    stage.prepare = lambda image_id: (None, PILImage.new("RGB", (16, 16)))  # no files here
    for image_id in ids:
        with db.session() as s:
            img = s.get(Image, image_id)
            s.expunge(img)
        stage.process_prepared(image_id, (img, PILImage.new("RGB", (16, 16))))
    rows = nsfw_rows(db)
    assert rows[ids[0]] == ("yes", "vlm", None)
    assert rows[ids[1]] == ("no", "vlm", None)
    assert rows[ids[2]][0] == "yes"  # a lenient answer ("yes") counts as well
    # the classifier says yes, the vision model no: the photo stays covered, and analysing it
    # again keeps the classifier's score
    with db.write() as s:
        V.record_score(s, ids[1], 0.93, settings.nsfw.threshold)
    assert nsfw_rows(db)[ids[1]] == ("yes", "classifier", 0.93)
    with db.session() as s:
        img = s.get(Image, ids[1])
        s.expunge(img)
    backend.responses = [answer(vocab, nsfw=False)]
    stage.process_prepared(ids[1], (img, PILImage.new("RGB", (16, 16))))
    assert nsfw_rows(db)[ids[1]] == ("yes", "classifier", 0.93)
    with db.session() as s:  # still one row, and the other properties are back
        props = s.scalars(select(ImageProperty.key).where(ImageProperty.image_id == ids[1])).all()
    assert props.count("nsfw") == 1 and "setting" in props


# --------------------------------------------------------------------------- the stage
def test_classifier_stage_threshold_and_registry(db, settings, photos):
    for i in range(4):  # 300 px wide: NSFW for the fake classifier, 320 px: safe
        make_image(photos / f"{i}.jpg", seed=i, size=(300, 240) if i % 2 else (320, 240))
    Indexer(db, settings, stages=["nsfw"]).run()
    classifier = FakeNsfwClassifier(lambda image: 0.98 if image.width == 300 else 0.75)
    stage = NsfwStage(db, settings, classifier=classifier)
    settings.nsfw.threshold = 0.8
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["nsfw"])
    assert summary.status == "completed", summary.messages
    assert classifier.calls == 4 and not classifier.loaded  # unloaded after the stage
    rows = nsfw_rows(db)
    assert sorted(v for v, _src, _c in rows.values()) == ["no", "no", "yes", "yes"]
    assert {c for _v, _src, c in rows.values()} == {0.98, 0.75}
    with db.session() as s:
        versions = set(s.scalars(select(StageStatus.stage_version)))
    assert versions == {stage.version} and "-t0.8-" in stage.version
    # another threshold makes the results outdated (the stage is fast: re-run them)
    assert stage_versions(Settings(nsfw={"threshold": 0.5}))["nsfw"] != stage.version
    assert STAGE_NAMES[0] == "nsfw" and Settings().pipeline.stages[0] == "nsfw"
    assert stage_versions(Settings())["nsfw"].startswith(f"vit:{VIT_NSFW.sha256[:8]}-t0.7-")
    with pytest.raises(ValueError):
        Settings(nsfw={"threshold": 1.5})


def test_classifier_stage_pause_and_stop(db, settings, photos):
    for i in range(6):
        make_image(photos / f"{i}.jpg", seed=i)
    Indexer(db, settings, stages=["nsfw"]).run()
    seen: list[int] = []
    first = threading.Event()

    def scorer(image):
        seen.append(len(seen) + 1)
        if len(seen) == 1:
            control.request_state(db, control.PAUSED)
            first.set()
        if len(seen) == 3:
            control.request_state(db, control.STOPPED)
        return 0.1

    stage = NsfwStage(db, settings, classifier=FakeNsfwClassifier(scorer))
    runner = PipelineRunner(db, settings, lambda _n: stage)
    runner.poll_interval = 0.05
    bg = BackgroundRunner(runner)
    assert bg.start(["nsfw"], batch_size=4)
    assert first.wait(5)
    time.sleep(0.4)
    assert bg.running and seen == [1]  # paused after the first photo
    control.request_state(db, control.RUNNING)
    bg.join(10)
    assert bg.last_summary.status == "stopped" and seen == [1, 2, 3]
    with db.session() as s:
        statuses = [
            st.status for st in s.scalars(select(StageStatus).order_by(StageStatus.image_id))
        ]
    assert statuses.count(Status.DONE) == 3 and statuses.count(Status.PENDING) == 3
    assert len(nsfw_rows(db)) == 3  # nothing written for the photos it did not get to
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["nsfw"])
    assert summary.status == "completed" and len(nsfw_rows(db)) == 6


def test_preprocessing_and_probability():
    im = PILImage.new("RGB", (640, 480), (255, 0, 128))
    x = preprocess(im)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x[0, 0].max() == pytest.approx(1.0) and x[0, 1].min() == pytest.approx(-1.0)
    assert x[0, 2, 0, 0] == pytest.approx(128 / 255 * 2 - 1, abs=1e-6)
    p = nsfw_probability(np.array([[0.0, 0.0], [-4.0, 4.0], [3.0, -3.0]]))
    assert p == pytest.approx([0.5, 0.99966, 0.00247], abs=1e-4)


# --------------------------------------------------------------------------- downloads
def test_the_model_is_pinned():
    assert VIT_NSFW.license == "Apache-2.0" and VIT_NSFW.size == 171_801_382
    assert re.search(r"/resolve/[0-9a-f]{40}/", VIT_NSFW.url)  # one commit, not "main"
    assert len(VIT_NSFW.sha256) == 64
    s = Settings(faces={"model_root": "/models"})
    assert default_model_dir(s).as_posix() == "/models/models/nsfw"
    assert default_model_dir(Settings(nsfw={"model_dir": "/x"})).as_posix() == "/x"


def test_download_verification_and_offline_error(tmp_path, settings):
    data = b"onnx" * 500
    spec = ModelFile(
        "tiny.onnx",
        "https://example.invalid/m",
        hashlib.sha256(data).hexdigest(),
        len(data),
        "Apache-2.0",
        "Tiny",
    )

    def tampered(request):
        return httpx.Response(200, content=b"x" * len(data))

    models = tmp_path / "models"
    with pytest.raises(ModelUnavailable, match="not the expected file"):
        ensure_model(models, spec, client=httpx.Client(transport=httpx.MockTransport(tampered)))
    assert list(models.iterdir()) == []  # nothing half-written is left behind

    def offline(request):
        raise httpx.ConnectError("no route to host")

    settings.nsfw.model_dir = tmp_path / "nsfw"
    client = httpx.Client(transport=httpx.MockTransport(offline))
    stage = NsfwStage(None, settings, classifier=NsfwClassifier(settings, client=client))
    with pytest.raises(StageAbort) as err:  # the run stops, the photos stay pending
        stage.setup()
    message = str(err.value)
    assert "cannot download the NSFW classifier (ViT) model" in message
    assert VIT_NSFW.url in message and VIT_NSFW.sha256 in message
    assert not any((tmp_path / "nsfw").glob("*.part"))


# --------------------------------------------------------------------------- your verdict wins
def test_manual_override_survives_reruns(db, settings, photos):
    for i in range(3):
        make_image(photos / f"{i}.jpg", seed=i)
    Indexer(db, settings, stages=["nsfw", "vlm"]).run()
    with db.session() as s:
        a, b, c = s.scalars(select(Image.id).order_by(Image.id)).all()
    thr = settings.nsfw.threshold
    with db.write() as s:
        V.record_score(s, a, 0.99, thr)
        V.record_score(s, b, 0.01, thr)
        assert V.set_manual(s, [a], "no", thr) == 1
        assert V.set_manual(s, [b, c, 99999], "yes", thr) == 2  # unknown ids are ignored
        assert V.set_manual(s, [b], "yes", thr) == 0  # already so
    assert nsfw_rows(db) == {
        a: ("no", "manual", 0.99),
        b: ("yes", "manual", 0.01),
        c: ("yes", "manual", None),
    }
    # the classifier runs again: only the score changes
    stage = NsfwStage(db, settings, classifier=FakeNsfwClassifier(lambda image: 0.97))
    control.requeue(db, "nsfw")
    assert PipelineRunner(db, settings, lambda _n: stage).run(["nsfw"]).status == "completed"
    assert nsfw_rows(db)[a] == ("no", "manual", 0.97)
    # the vision model finds it NSFW: still not NSFW
    vocab = Vocabulary.load()
    vlm = VlmStage(db, settings, backend=FakeBackend(vocab, responses=[answer(vocab, nsfw=True)]))
    vlm.process(a)
    assert nsfw_rows(db)[a] == ("no", "manual", 0.97)
    # the file changes: everything else is analysed again, your verdict stays
    make_image(photos / "0.jpg", seed=42)
    Indexer(db, settings, stages=["nsfw", "vlm"]).run()
    with db.session() as s:
        changed = s.scalar(select(Image.id).where(Image.filename == "0.jpg"))
        assert s.get(StageStatus, (changed, "nsfw")).status == Status.PENDING
    assert nsfw_rows(db)[changed][:2] == ("no", "manual")
    # automatic again: the stored score and the vision model's answer decide
    with db.write() as s:
        assert V.set_manual(s, [a, b, c], None, thr) == 3
    rows = nsfw_rows(db)
    assert rows[a] == ("yes", "classifier", 0.97)
    assert rows[b] == ("yes", "classifier", 0.97)
    with db.write() as s:
        V.set_manual(s, [c], "no", thr)
        s.execute(
            ImageProperty.__table__.update()
            .where(ImageProperty.image_id == c)
            .values(confidence=None)
        )
    with db.write() as s:
        V.set_manual(s, [c], None, thr)
    assert c not in nsfw_rows(db) or nsfw_rows(db)[c][1] != "manual"
    with db.write() as s, pytest.raises(ValueError):
        V.set_manual(s, [a], "maybe", thr)


# --------------------------------------------------------------------------- filters, search
def test_filters_hide_only_and_or(db):
    with db.write() as s:
        imgs = [add_image(s, n) for n in range(4)]
        for img, key, value in (
            (imgs[0], "nsfw", "yes"),
            (imgs[1], "nsfw", "no"),
            (imgs[2], "location_type", "beach"),
            (imgs[0], "location_type", "beach"),
        ):
            s.add(ImageProperty(image_id=img.id, key=key, value=value, source="manual"))
    with db.session() as s:
        hide = ImageFilter.from_params({"xt": "nsfw:yes"})
        assert count_images(s, hide) == 3
        assert count_images(s, ImageFilter.from_params({"t": "nsfw:yes"})) == 1
        either = ImageFilter.from_params({"t": "nsfw:yes|location_type:beach"})
        assert either.prop_groups == [[("nsfw", "yes"), ("location_type", "beach")]]
        assert count_images(s, either) == 2
        beach_hidden = ImageFilter.from_params({"t": "location_type:beach", "xt": "nsfw:yes"})
        assert count_images(s, beach_hidden) == 1
        assert ImageFilter().with_prop_alt("nsfw", "yes") == "t=nsfw%3Ayes"


def test_search_knows_nsfw_and_the_document_leaves_it_out(db, settings):
    from vision_tagger.search.engine import SearchEngine

    with db.write() as s:
        img = add_image(s, 1)
        props = [
            ImageProperty(image_id=img.id, key="nsfw", value="yes", source="classifier"),
            ImageProperty(image_id=img.id, key="season", value="summer", source="vlm"),
        ]
        s.add_all(props)
        doc = build_document(img, props, None)
    assert "nsfw" not in doc.lower() and "summer" in doc  # marking never re-embeds a photo
    ctx = SearchEngine(db, settings, backend=FakeBackend(Vocabulary.load())).context()
    it = KeywordInterpreter().interpret("nsfw photos", ctx)
    assert ("nsfw", "yes") in {(t.key, t.value) for t in it.must + it.should}
    it = KeywordInterpreter().interpret("summer without nsfw", ctx)
    assert [(t.key, t.value) for t in it.exclude] == [("nsfw", "yes")]


# --------------------------------------------------------------------------- the web UI
def _library(settings, photos, nsfw_all: bool):
    """Photos with faces, animals, GPS, memories; NSFW: all (``nsfw_all``) or the narrow ones."""
    today = date.today()
    for i in range(6):
        make_image(
            photos / f"IMG_{i}.jpg",
            seed=i,
            size=(300, 240) if i % 2 else (320, 240),
            date=datetime(2016 if i % 2 else 2020, today.month, today.day, 12),  # leap years
            gps=(52.52, 13.40),
        )
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    ctx = app.state.ctx
    Indexer(ctx.db, settings).run()
    scorer = (lambda im: 0.99) if nsfw_all else (lambda im: 0.99 if im.width == 300 else 0.02)
    stages = {
        "nsfw": NsfwStage(ctx.db, settings, classifier=FakeNsfwClassifier(scorer)),
        "visual": VisualStage(ctx.db, settings, encoder=FakeVisualEncoder()),
        "faces": FacesStage(
            ctx.db, settings, analyzer=FakeAnalyzer(), age_estimator=FakeAgeEstimator()
        ),
        "ages": AgesStage(ctx.db, settings, estimator=FakeAgeEstimator()),
        "animals": AnimalsStage(ctx.db, settings, analyzer=FakeAnimalAnalyzer()),
        "vlm": VlmStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
        "embed": EmbedStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
    }
    assert PipelineRunner(ctx.db, settings, stages.__getitem__).run().status == "completed"
    ctx.search.text_encoder = FakeVisualEncoder()  # search pages without the real model
    with ctx.db.write() as s:
        clusters = s.scalars(select(Cluster.id).order_by(Cluster.id)).all()
        C.name_cluster(s, clusters[0], "Anna")
        C.name_cluster(s, clusters[1], "Ben")
        animals = s.scalars(select(Animal.id)).all()
        move_animals_to_name(s, animals, "Bello")
        library.add(s, "All", "")
    client.ctx = ctx
    return client


@pytest.fixture
def all_nsfw(settings, photos):
    client = _library(settings, photos, nsfw_all=True)
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def mixed(settings, photos):
    client = _library(settings, photos, nsfw_all=False)
    yield client
    client.__exit__(None, None, None)


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def _urls(client) -> list[str]:
    with client.ctx.db.session() as s:
        anna = s.scalar(select(Person.id).where(Person.name == "Anna"))
        ben = s.scalar(select(Person.id).where(Person.name == "Ben"))
        cluster = s.scalar(select(Cluster.id).where(Cluster.person_id == anna))
        pet = s.scalar(select(Animal.pet_id))
        image = s.scalar(select(Image.id))
    return [
        "/",
        "/collections",
        "/images",
        "/images?sort=name",
        "/images/page?page=1",
        f"/images/{image}",
        "/search/results?q=trees",
        "/faces",
        "/faces/unclustered",
        f"/clusters/{cluster}",
        "/persons",
        f"/persons/{anna}",
        "/quiz?mode=name",
        "/quiz?mode=quiz",
        "/animals",
        f"/pets/{pet}",
        "/animals/ungrouped",
        "/properties/location_type",
        "/properties/nsfw",
        "/overview/table",
        "/generations",
        f"/generations/{anna}",
        f"/generations/{anna}/compare/{ben}",
    ]


MEDIA_SRC = re.compile(r'(?<![\w-])src="/media/(thumb|image|face|animal|aligned|reference)/')


class Covers(HTMLParser):
    """Each cover on a page (``<img data-nsfw>``) with its src and the link or label around it."""

    def __init__(self, page: str) -> None:
        super().__init__()
        self.stack: list[tuple[str, str | None]] = []  # open <a> / <label>
        self.covers: list[dict] = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("a", "label"):
            self.stack.append((tag, a.get("href")))
        elif tag == "img" and "data-nsfw" in a:
            around = self.stack[-1] if self.stack else (None, None)
            self.covers.append({"id": a["data-nsfw"], "src": a.get("src"), "around": around})

    def handle_endtag(self, tag):
        if tag in ("a", "label") and self.stack:
            self.stack.pop()


def covers_of(page: str) -> list[dict]:
    covers = Covers(page).covers
    for c in covers:  # never the real picture; always in a link (or a selectable grid tile)
        assert c["src"] == "/static/nsfw.svg", c
        assert c["around"][0] == "label" or c["around"][1], c
    return covers


def test_every_picture_of_an_nsfw_photo_is_covered_in_lists(all_nsfw):
    client = all_nsfw
    photo_page = _urls(client)[5]
    total = 0
    for url in _urls(client):
        if url == photo_page:
            continue
        for headers in ({}, DE):
            page = ok(client.get(url, headers=headers)).text
            assert not MEDIA_SRC.search(page), (url, MEDIA_SRC.search(page).group(0))
            assert "/media/thumb/" not in page and "/media/face/" not in page, url  # not at all
            total += len(covers_of(page))
    assert total > 100
    # a click on a cover in a list follows its link: to the photo page
    for url in ("/images?sort=name", "/search/results?q=trees", "/"):
        covers = covers_of(ok(client.get(url)).text)
        to_photo = [c for c in covers if (c["around"][1] or "").startswith("/images/")]
        assert to_photo, url
        assert all(c["around"][1].split("?")[0] == f"/images/{c['id']}" for c in to_photo), url
    home = ok(client.get("/")).text
    assert "memory" in home and "coll-card" in home  # On this day, collections, recent
    # covers that are in no link of their own link to the photo page themselves
    with client.ctx.db.session() as s:
        anna = s.scalar(select(Person.id).where(Person.name == "Anna"))
    person = ok(client.get(f"/persons/{anna}")).text
    assert re.search(r'<a class="nsfw-link" href="/images/\d+"><img class="avatar lg"', person)
    quiz = ok(client.get("/quiz?mode=quiz")).text
    assert '<a class="nsfw-link" href="/images/' in quiz
    page = ok(client.get("/images", headers=DE)).text
    assert 'title="Nicht jugendfrei (NSFW) \u2013 öffne das Foto, um es zu sehen"' in page
    assert "NSFW-Fotos ausblenden" in page and "NSFW-Fotos ohne Nachfrage zeigen" in page
    # the map popups: the photos come with the ids to cover
    places = ok(client.get("/api/geo/places")).json()["places"]
    detail = ok(client.get(f"/api/geo/places/{places[0]['id']}")).json()
    assert detail["images"] and detail["nsfw"] == detail["images"]
    assert ok(client.get(f"/api/geo/places/{places[0]['id']}", headers=SHOW)).json()["nsfw"] == []
    # the grow-up viewer leaves faces of NSFW photos out
    assert "/media/aligned/" not in ok(client.get(f"/generations/{anna}")).text


def test_the_photo_page_shows_the_photo(all_nsfw):
    client = all_nsfw
    for headers, note in (({}, "Not safe for work (NSFW)"), (DE, "Nicht jugendfrei (NSFW)")):
        page = ok(client.get("/images/1?sort=name", headers=headers)).text
        assert '<img src="/media/image/1" alt=' in page and "data-nsfw=" not in page
        assert 'class="nsfw-note"' in page and note in page
        assert 'class="facebox' in page  # face boxes as usual
        assert 'data-hotkey="ArrowRight"' in page  # the next photo page shows its photo too
    page = ok(client.get("/images/1")).text
    assert "found by the NSFW check (score 99 %)" in page and 'id="nsfw"' in page


def test_show_without_asking_is_a_cookie(all_nsfw):
    client = all_nsfw
    page = ok(client.get("/images", headers=SHOW)).text
    assert ' src="/media/thumb/' in page and "data-nsfw=" not in page
    assert re.search(r"data-nsfw-show\s+checked", page)
    assert "data-nsfw=" not in ok(client.get("/persons/1", headers=SHOW)).text
    client.cookies.clear()
    assert "data-nsfw=" in ok(client.get("/images")).text


def test_only_nsfw_photos_are_covered(mixed):
    client = mixed
    with client.ctx.db.session() as s:
        nsfw = sorted(V.nsfw_images(s))
        safe = sorted(set(s.scalars(select(Image.id))) - set(nsfw))
    assert len(nsfw) == 3 and len(safe) == 3
    page = ok(client.get("/images")).text
    assert sorted(int(c["id"]) for c in covers_of(page)) == nsfw
    for i in nsfw:
        assert f"/media/thumb/{i}" not in page and f'data-nsfw="{i}"' in page
    for i in safe:
        assert f' src="/media/thumb/{i}"' in page and f'data-nsfw="{i}"' not in page
    # the gallery's NSFW section: hide them, only them (with the count)
    assert "/images?xt=nsfw%3Ayes" in page and "/images?t=nsfw%3Ayes" in page
    assert re.search(r"Only NSFW photos <span class=\"n\">3</span>", page)
    hidden = ok(client.get("/images?xt=nsfw%3Ayes")).text
    assert all(f"/images/{i}?" not in hidden for i in nsfw) and "data-nsfw=" not in hidden
    assert 'class="chip active" href="/images"' in hidden  # a click shows them again
    only = ok(client.get("/images?t=nsfw%3Ayes")).text
    assert all(f'/media/thumb/{i}"' not in only for i in safe)
    # collections and memories prefer covers without a cover
    home = ok(client.get("/")).text
    coll = re.search(r'<div class="coll-cover n\d">(.*?)</div>', home, re.S).group(1)
    assert coll.count(' src="/media/thumb/') == 3 and "data-nsfw=" in coll
    # face crops of NSFW photos are covered, the others not; a click on a tile still selects it
    with client.ctx.db.session() as s:
        cluster = s.scalar(select(Cluster.id))
    faces = ok(client.get(f"/clusters/{cluster}")).text
    grid = [c for c in covers_of(faces) if c["around"][0] == "label"]
    assert len(grid) == 3 and ' src="/media/face/' in faces
    # the photo page of a safe photo has no NSFW note
    assert 'class="nsfw-note"' not in ok(client.get(f"/images/{safe[0]}")).text


def test_marking_from_the_ui(mixed):
    client = mixed
    with client.ctx.db.session() as s:
        nsfw = sorted(V.nsfw_images(s))
        safe = sorted(set(s.scalars(select(Image.id))) - set(nsfw))
    page = ok(client.get(f"/images/{safe[0]}")).text
    assert "Safe for work \u2013 checked by the NSFW check (score 2 %)." in page
    assert f'action="/images/{safe[0]}/nsfw"' in page
    r = client.post(f"/images/{safe[0]}/nsfw", data={"value": "yes"}, follow_redirects=False)
    assert r.status_code == 303 and "marked+as+NSFW" in r.headers["location"]
    assert nsfw_rows(client.ctx.db)[safe[0]][:2] == ("yes", "manual")
    page = ok(client.get(f"/images/{safe[0]}", headers=DE)).text
    assert "Nicht jugendfrei \u2013 von dir markiert." in page and "Automatisch" in page
    # the gallery selection: several at once
    page = ok(client.get("/images")).text
    assert 'id="gallery-actions"' in page and 'form="gallery-actions"' in page
    r = client.post("/images/nsfw", data={"value": "no", "image_ids": [str(i) for i in nsfw]})
    assert r.status_code == 200 and "3 photos marked as not NSFW." in r.text
    rows = nsfw_rows(client.ctx.db)
    assert all(rows[i][:2] == ("no", "manual") for i in nsfw)
    assert f'data-nsfw="{safe[0]}"' in ok(client.get("/images")).text  # covered now
    r = client.post("/images/nsfw", data={"value": "auto", "image_ids": [str(i) for i in nsfw]})
    assert "3 photos are checked automatically again." in r.text
    assert all(nsfw_rows(client.ctx.db)[i][:2] == ("yes", "classifier") for i in nsfw)
    assert client.post("/images/nsfw", data={"value": "maybe"}).status_code == 400
    # the NSFW values are set per photo, not merged or deleted on the property page
    r = client.post("/properties/nsfw/delete", data={"values": ["yes"]})
    assert r.status_code == 200 and "can&#39;t be merged or deleted" in r.text
    assert len(nsfw_rows(client.ctx.db)) == 6


def test_marking_needs_post_from_this_ui(mixed):
    client = mixed
    for url in ("/images/nsfw", "/images/1/nsfw"):
        r = client.post(
            url,
            data={"value": "no", "image_ids": ["1"]},
            headers={"Origin": "https://evil.example"},
        )
        assert r.status_code == 403
        r = client.post(url, data={"value": "no"}, headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403
    assert client.get("/images/1/nsfw").status_code in (404, 405)
    assert client.get("/images/nsfw").status_code in (404, 405, 422)
    with client.ctx.db.session() as s:
        assert s.scalar(select(Place.id)) is not None
    assert all(src != "manual" for _v, src, _c in nsfw_rows(client.ctx.db).values())


def test_static_cover_and_service_worker(mixed):
    client = mixed
    svg = ok(client.get("/static/nsfw.svg"))
    assert "NSFW" in svg.text and 'fill="#000"' in svg.text
    sw = ok(client.get("/sw.js")).text
    assert '"/static/nsfw.svg"' in sw
    js = ok(client.get("/static/app.js")).text
    assert "vt_nsfw=show" in js  # the setting
    # no reveal in place any more: no revealing script, no list of revealed photos
    for gone in ("data-nsfw-src", "dataset.nsfw", "img[data-nsfw", "vtNsfw", "sessionStorage"):
        assert gone not in js, gone
    css = ok(client.get("/static/app.css")).text
    assert "data-nsfw-src" not in css and "nsfw-hint" not in css
