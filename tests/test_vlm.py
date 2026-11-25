from __future__ import annotations

import json
import os

import httpx
import pytest
from PIL import Image as PILImage
from sqlalchemy import select

from vision_tagger.db.models import ImageProperty, PropertyAlias, StageStatus, Status, VlmResult
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages import stage_versions
from vision_tagger.pipeline.stages.vlm import VlmStage, build_properties
from vision_tagger.vlm.backend import FakeBackend, OllamaBackend, parse_json_answer
from vision_tagger.vlm.normalize import Normalizer, singularize
from vision_tagger.vlm.prompt import prompt_version, system_prompt
from vision_tagger.vlm.schema import Vocabulary, default_synonyms_path

from .conftest import make_image


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return Vocabulary.load()


def good_answer(vocab: Vocabulary, **over):
    data = {
        "image_type": "Photo",
        "setting": "outdoor",
        "location_type": "Beach",
        "time_of_day": "day",
        "season": "summer",
        "weather": "sunny",
        "lighting": "natural",
        "people_count": 4,
        "composition": "group",
        "main_activity": "relaxing",
        "event_type": "vacation",
        "objects": ["Umbrellas", "a towel", "sunglasses", "beach chairs", "umbrella"],
        "animals": ["dogs"],
        "vehicles": [],
        "food": ["ice cream"],
        "dominant_colors": ["blue", "yellow", "turquoise"],
        "text_present": False,
        "text_language": "none",
        "text_excerpt": "",
        "landmark": "",
        "caption": "Friends relaxing on a sunny beach.",
        "tags": ["Summer", "holiday", "sea", "ocean"],
        "sharpness": "sharp",
        "exposure": "ok",
        "aesthetic": "high",
        "is_work_related": False,
        "is_work_related_confidence": 0.95,
        "work_category": "not_work",
        "work_category_confidence": 0.9,
        "is_personal_memory": True,
        "is_personal_memory_confidence": 0.9,
        "likely_purpose": "memory",
        "likely_purpose_confidence": 0.8,
        "contains_sensitive_info": False,
        "contains_sensitive_info_confidence": 0.9,
    }
    data.update(over)
    return data


def test_schema_has_all_fields_and_confidences(vocab):
    schema = vocab.json_schema()
    props = schema["properties"]
    assert props["location_type"]["enum"][0] == "home"
    assert props["is_work_related"]["type"] == "boolean"
    assert "is_work_related_confidence" in props
    assert set(schema["required"]) == set(props)
    assert "(assume)" in system_prompt(vocab)


def test_prompt_version_changes_with_vocabulary(vocab, tmp_path):
    v1 = prompt_version(vocab)
    raw = dict(vocab.raw)
    raw["fields"] = dict(raw["fields"])
    raw["fields"]["mood"] = {"type": "enum", "values": ["happy", "sad"]}
    assert prompt_version(Vocabulary(raw)) != v1
    assert prompt_version(Vocabulary.load()) == v1


def test_coerce_is_lenient(vocab):
    out = vocab.coerce(good_answer(vocab, setting="Out-door?!", weather="hail", people_count="7.0"))
    assert out["image_type"] == "photo"
    assert out["location_type"] == "beach"
    assert out["setting"] == "unknown"  # unknown enum -> fallback value
    assert out["weather"] == "unknown"
    assert out["people_count"] == 7
    assert out["dominant_colors"] == ["blue", "yellow"]  # turquoise is not in the palette
    with pytest.raises(ValueError):
        vocab.coerce({"image_type": "photo"})


def test_normalizer():
    assert singularize("beach chairs") == "beach chair"
    assert singularize("glasses") == "glasses"
    assert singularize("boxes") == "box"
    n = Normalizer.from_file(default_synonyms_path(), {("objects", "towel"): "beach towel"})
    assert n.canonical("objects", "Smartphones") == "phone"
    assert n.canonical("objects", "a Towel") == "beach towel"
    assert n.canonical("tags", "ocean") == "sea"
    assert n.canonical("objects", "none") is None
    n.aliases[("tags", "summer")] = None
    assert n.canonical("tags", "Summer") is None


def test_build_properties(vocab):
    data = vocab.coerce(good_answer(vocab))
    n = Normalizer.from_file(default_synonyms_path())
    rows = build_properties(vocab, data, n, max_items=15)
    keys = {(k, v) for k, v, _c, _s in rows}
    assert ("location_type", "beach") in keys
    assert ("objects", "umbrella") in keys
    assert ("objects", "beach chair") in keys
    assert sum(1 for k, v in keys if k == "objects" and v == "umbrella") == 1
    assert ("tags", "sea") in keys and ("tags", "ocean") not in keys
    assert ("is_work_related", "no") in keys
    assert ("animals", "dog") in keys
    assert not any(k in ("caption", "text_excerpt", "people_count") for k, _ in keys)
    inferred = next(r for r in rows if r[0] == "is_work_related")
    assert inferred[2] == 0.95 and inferred[3] == "inferred"


def test_parse_json_answer():
    assert parse_json_answer('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_answer('{"a": [1, 2]}') == {"a": [1, 2]}


@pytest.fixture
def indexed(db, settings, photos):
    make_image(photos / "beach.jpg", seed=1)
    make_image(photos / "Screenshots" / "Screenshot_2023-01-01.png", seed=2, size=(1920, 1080))
    Indexer(db, settings, stages=["vlm"]).run()
    return db


def props(db, image_id):
    with db.session() as s:
        return {
            (p.key, p.value): p
            for p in s.scalars(select(ImageProperty).where(ImageProperty.image_id == image_id))
        }


def test_vlm_stage_stores_properties(indexed, settings, vocab):
    work = good_answer(
        vocab,
        image_type="screenshot",
        is_work_related=True,
        is_work_related_confidence=0.9,
        work_category="screenshot",
        is_personal_memory=False,
        people_count=0,
    )
    backend = FakeBackend(vocab, responses=[good_answer(vocab), work])
    stage = VlmStage(indexed, settings, backend=backend)
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["vlm"])
    assert summary.status == "completed" and summary.failed["vlm"] == 0
    p1, p2 = props(indexed, 1), props(indexed, 2)
    assert ("work_image", "no") in p1
    assert ("work_image", "yes") in p2
    assert p2[("work_image", "yes")].source == "derived"
    assert ("people", "3-5") in p1
    assert ("orientation", "landscape") in p1 and p1[("orientation", "landscape")].source == "exif"
    with indexed.session() as s:
        res = s.get(VlmResult, 1)
        assert res.caption == "Friends relaxing on a sunny beach."
        assert res.raw_json["location_type"] == "beach"
        assert s.get(StageStatus, (1, "vlm")).stage_version == stage.version
    assert stage_versions(settings)["vlm"] == stage.version

    # re-running replaces VLM properties but keeps metadata ones
    backend.responses = [good_answer(vocab, location_type="forest")]
    stage.process(1)
    p1 = props(indexed, 1)
    assert ("location_type", "forest") in p1 and ("location_type", "beach") not in p1
    assert ("orientation", "landscape") in p1


def test_vlm_retries_invalid_answer_then_errors(indexed, settings, vocab):
    settings.vlm.retries = 1
    backend = FakeBackend(
        vocab, responses=["not json at all", good_answer(vocab), "{}", '{"image_type": "x"}']
    )
    stage = VlmStage(indexed, settings, backend=backend)
    PipelineRunner(indexed, settings, lambda _n: stage).run(["vlm"])
    with indexed.session() as s:
        assert s.get(StageStatus, (1, "vlm")).status == Status.DONE
        row = s.get(StageStatus, (2, "vlm"))
        assert row.status == Status.ERROR and "misses most fields" in row.last_error
    assert backend.calls == 4


def test_vlm_applies_user_aliases(indexed, settings, vocab):
    with indexed.write() as s:
        s.add(PropertyAlias(key="location_type", from_value="beach", to_value="seaside"))
        s.add(PropertyAlias(key="objects", from_value="towel", to_value=None))
    backend = FakeBackend(vocab, responses=[good_answer(vocab)])
    stage = VlmStage(indexed, settings, backend=backend)
    stage.setup()
    stage.process(1)
    p1 = props(indexed, 1)
    assert ("location_type", "seaside") in p1
    assert ("objects", "towel") not in p1


def test_setup_fails_when_backend_unhealthy(indexed, settings, vocab):
    class Down(FakeBackend):
        def health(self):
            return {"ok": False, "error": "connection refused"}

    stage = VlmStage(indexed, settings, backend=Down(vocab))
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["vlm"])
    assert summary.status == "failed"
    assert "connection refused" in summary.messages[0]
    with indexed.session() as s:  # claimed rows are released again
        assert {r.status for r in s.scalars(select(StageStatus))} == {Status.PENDING}


def test_fake_backend_default_answers_are_valid(vocab):
    backend = FakeBackend(vocab)
    im = PILImage.new("RGB", (64, 64), (200, 30, 30))
    out = vocab.coerce(backend.analyze(im, "", "", vocab.json_schema()))
    assert out["dominant_colors"] == ["red"]


def test_ollama_backend_payload(settings, vocab):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            seen["payload"] = json.loads(request.content)
            return httpx.Response(
                200, json={"message": {"content": json.dumps(good_answer(vocab))}}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:7b"}]})
        return httpx.Response(404)

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    backend = OllamaBackend(settings.vlm, client=client)
    im = PILImage.new("RGB", (32, 32))
    out = backend.analyze(im, "sys", "user", vocab.json_schema())
    assert out["location_type"] == "Beach"
    p = seen["payload"]
    assert p["model"] == "qwen2.5vl:7b"
    assert p["options"]["temperature"] == 0 and p["options"]["seed"] == 42
    assert p["format"]["type"] == "object"
    assert p["messages"][1]["images"][0]
    assert backend.health()["ok"] is True


@pytest.mark.ollama
@pytest.mark.skipif(not os.environ.get("VT_OLLAMA_URL"), reason="set VT_OLLAMA_URL to run")
def test_real_ollama(settings, vocab):
    settings.vlm.base_url = os.environ["VT_OLLAMA_URL"]
    settings.vlm.model = os.environ.get("VT_OLLAMA_MODEL", settings.vlm.model)
    backend = OllamaBackend(settings.vlm)
    assert backend.health()["ok"], backend.health()
    im = PILImage.new("RGB", (256, 256), (30, 120, 220))
    raw = backend.analyze(im, system_prompt(vocab), "Catalogue this image.", vocab.json_schema())
    out = vocab.coerce(raw)
    assert out["image_type"] in vocab.fields["image_type"].values
