from __future__ import annotations

import json
from datetime import datetime

import httpx
import numpy as np
import pytest
from sqlalchemy import select

from vision_tagger.db.models import (
    Cluster,
    ImageEmbedding,
    ImageProperty,
    Person,
    StageStatus,
    Status,
    VlmResult,
)
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.vlm import VlmStage
from vision_tagger.properties import merge_values
from vision_tagger.search.document import build_document
from vision_tagger.search.engine import SearchEngine
from vision_tagger.search.index import EmbeddingMatrix
from vision_tagger.search.interpret import KeywordInterpreter
from vision_tagger.vlm.backend import FakeBackend, OllamaBackend
from vision_tagger.vlm.schema import Vocabulary

from .helpers import add_face, add_image, person_centers

LIBRARY = {
    # n: (caption, date, folder, props, persons)
    1: (
        "Friends relaxing on a sunny beach",
        datetime(2021, 7, 10),
        "Holidays/Italy",
        [
            ("location_type", "beach"),
            ("season", "summer"),
            ("objects", "umbrella"),
            ("work_image", "no"),
        ],
        ["Anna", "Ben"],
    ),
    2: (
        "A whiteboard with project notes in an office",
        datetime(2021, 3, 2),
        "Work",
        [("location_type", "office"), ("image_type", "whiteboard"), ("work_image", "yes")],
        ["Ben"],
    ),
    3: (
        "Snowy mountain hike with a view",
        datetime(2020, 1, 5),
        "Holidays/Alps",
        [
            ("location_type", "mountain"),
            ("weather", "snowy"),
            ("season", "winter"),
            ("work_image", "no"),
        ],
        ["Anna"],
    ),
    4: (
        "Family christmas dinner at home",
        datetime(2020, 12, 24),
        "Family",
        [
            ("location_type", "home"),
            ("event_type", "christmas"),
            ("season", "winter"),
            ("work_image", "no"),
        ],
        ["Anna", "Ben"],
    ),
    5: (
        "Sunset over the sea seen from the beach",
        datetime(2019, 8, 1),
        "Holidays/Greece",
        [
            ("location_type", "beach"),
            ("time_of_day", "golden_hour"),
            ("season", "summer"),
            ("work_image", "no"),
        ],
        [],
    ),
}


@pytest.fixture
def vocab():
    return Vocabulary.load()


@pytest.fixture
def library(db, settings, vocab):
    centers = person_centers(2)
    with db.write() as s:
        anna = Cluster(locked=True, person=Person(name="Anna"))
        ben = Cluster(locked=True, person=Person(name="Ben"))
        s.add_all([anna, ben])
        s.flush()
        clusters = {"Anna": anna, "Ben": ben}
        for n, (caption, when, folder, props, persons) in LIBRARY.items():
            img = add_image(s, n, recorded_at=when, folder=folder)
            img.stages.append(StageStatus(stage="embed", status=Status.PENDING))
            for k, v in props:
                s.add(ImageProperty(image_id=img.id, key=k, value=v, source="vlm"))
            s.add(
                VlmResult(
                    image_id=img.id,
                    model="fake",
                    prompt_version="p",
                    caption=caption,
                    raw_json={"caption": caption, "text_excerpt": ""},
                )
            )
            for i, name in enumerate(persons):
                add_face(s, img.id, centers[i], cluster_id=clusters[name].id)
    backend = FakeBackend(vocab)
    stage = EmbedStage(db, settings, backend=backend)
    for n in LIBRARY:
        stage.process(n)
    return db, backend


def engine(db, settings, vocab, chat=None):
    backend = FakeBackend(vocab, chat_responses=chat or [])
    return SearchEngine(db, settings, backend=backend), backend


def ids(result):
    return [h.image_id for h in result.hits]


# --------------------------------------------------------------------------- index
def test_document_contains_everything_but_person_names(library):
    db, _ = library
    with db.session() as s:
        from vision_tagger.db.models import Image

        img = s.get(Image, 1)
        props = list(s.scalars(select(ImageProperty).where(ImageProperty.image_id == 1)))
        doc = build_document(img, props, s.get(VlmResult, 1))
    assert "Friends relaxing on a sunny beach" in doc
    assert "Location type: beach." in doc and "Objects: umbrella." in doc
    assert "Folder: Holidays / Italy" in doc
    assert "July 2021" in doc and "summer 2021" in doc
    assert "Anna" not in doc and "Ben" not in doc


def test_embed_stage_skips_unchanged_and_requeues_on_changes(library, settings, vocab):
    db, backend = library
    with db.session() as s:
        first = s.get(ImageEmbedding, 1).created_at
        assert s.get(ImageEmbedding, 1).dim == FakeBackend.EMBED_DIM
    EmbedStage(db, settings, backend=backend).process(1)
    with db.session() as s:
        assert s.get(ImageEmbedding, 1).created_at == first  # unchanged document -> no work

    # property merge re-queues embed for the affected images
    with db.write() as s:
        for st in s.scalars(select(StageStatus).where(StageStatus.stage == "embed")):
            st.status = Status.DONE
        merge_values(s, "location_type", ["beach"], "seaside")
    with db.session() as s:
        pending = {
            r.image_id
            for r in s.scalars(
                select(StageStatus).where(
                    StageStatus.stage == "embed", StageStatus.status == Status.PENDING
                )
            )
        }
        assert pending == {1, 5}

    # re-running the VLM stage re-queues embed as well
    from .test_vlm import good_answer

    stage = VlmStage(db, settings, backend=FakeBackend(vocab))
    with db.session() as s:
        from vision_tagger.db.models import Image

        img3 = s.get(Image, 3)
        s.expunge(img3)
    stage.analyze = lambda image_id: (img3, vocab.coerce(good_answer(vocab)), 0.1)
    stage.process(3)
    with db.session() as s:
        assert s.get(StageStatus, (3, "embed")).status == Status.PENDING


def test_embed_stage_aborts_without_model(db, settings, vocab):
    from vision_tagger.pipeline.stages.base import StageAbort

    class NoModel(FakeBackend):
        def has_model(self, name):
            return False

    with pytest.raises(StageAbort, match="ollama pull nomic-embed-text"):
        EmbedStage(db, settings, backend=NoModel(vocab)).setup()


def test_embedding_matrix_incremental(library, settings):
    db, backend = library
    m = EmbeddingMatrix(db)
    m.ensure("nomic-embed-text")
    assert m.count == 5
    with db.write() as s:
        from vision_tagger.db.models import utcnow

        row = s.get(ImageEmbedding, 2)
        row.vector = np.ones(FakeBackend.EMBED_DIM, dtype=np.float16).tobytes()
        row.created_at = utcnow()
    m.ensure("nomic-embed-text")
    q = np.ones(FakeBackend.EMBED_DIM, dtype=np.float32)
    sims = m.similarities(q, [1, 2])
    assert sims[2] > sims[1]


# --------------------------------------------------------------------------- interpretation
def test_keyword_interpreter_maps_people_values_dates_and_negation(library, settings, vocab):
    db, _ = library
    eng, _ = engine(db, settings, vocab)
    ctx = eng.context()
    it = KeywordInterpreter().interpret("Anna am Strand im Sommer 2021 ohne Arbeit", ctx)
    assert [n for _, n in it.persons] == ["Anna"]
    assert {(t.key, t.value) for t in it.should} >= {
        ("location_type", "beach"),
        ("season", "summer"),
    }
    assert [(t.key, t.value) for t in it.exclude] == [("work_image", "yes")]
    assert (it.date_from, it.date_to) == ("2021", "2021")
    assert it.mode == "basic"
    it = KeywordInterpreter().interpret("photos without Ben", ctx)
    assert [n for _, n in it.exclude_persons] == ["Ben"] and not it.persons


def test_llm_answer_is_validated(library, settings, vocab):
    db, _ = library
    answer = {
        "persons": ["anna", "Zoe"],
        "persons_mode": "all",
        "exclude_persons": [],
        "must": [{"key": "location_type", "value": "Beach"}, {"key": "colour", "value": "red"}],
        "should": [
            {"key": "objects", "value": "umbrellas", "weight": 2},
            {"key": "objects", "value": "kite", "weight": 0.5},
        ],
        "exclude": [{"key": "work_image", "value": "yes"}],
        "date_from": "2021",
        "date_to": "last year",
        "semantic_query": "friends on a sunny beach",
        "keywords": [],
    }
    eng, backend = engine(db, settings, vocab, chat=[answer])
    it = eng.interpret("anna and zoe at the beach")
    assert it.mode == "llm"
    assert [n for _, n in it.persons] == ["Anna"]  # case-insensitive, unknown Zoe -> keyword
    assert [(t.key, t.value) for t in it.must] == [("location_type", "beach")]
    assert [(t.key, t.value, t.weight) for t in it.should] == [("objects", "umbrella", 1.0)]
    assert {"Zoe", "kite", "red"} <= set(it.keywords)
    assert (it.date_from, it.date_to) == ("2021", None)  # invalid date dropped
    assert "Anna" in backend.chat_calls[0][0]  # person list is part of the prompt
    assert "location_type: home" in backend.chat_calls[0][0]


# --------------------------------------------------------------------------- engine
def test_keyword_mode_search_ranks_and_filters(library, settings, vocab):
    db, _ = library
    eng, _ = engine(db, settings, vocab)
    r = eng.search("Anna in the snow")
    assert ids(r)[0] == 3
    assert set(ids(r)) <= {1, 3, 4}  # Anna is a hard filter
    assert r.semantic_used and r.semantic_coverage == 1.0
    r = eng.search("beach sunset without work")
    assert ids(r)[0] == 5 and 2 not in ids(r)
    r = eng.search("whiteboard notes")  # only in the caption -> semantic + keywords
    assert ids(r)[0] == 2


def test_llm_must_relaxes_when_nothing_matches(library, settings, vocab):
    db, _ = library
    answer = {
        "persons": ["Anna"],
        "persons_mode": "all",
        "exclude_persons": [],
        "must": [{"key": "location_type", "value": "office"}],
        "should": [],
        "exclude": [],
        "date_from": "",
        "date_to": "",
        "semantic_query": "Anna in the office",
        "keywords": [],
    }
    eng, _ = engine(db, settings, vocab, chat=[answer])
    r = eng.search("Anna at the office")
    assert r.relaxed and set(ids(r)) == {1, 3, 4}


def test_persons_any_prefers_images_with_more_of_them(library, settings, vocab):
    db, _ = library
    answer = {
        "persons": ["Anna", "Ben"],
        "persons_mode": "any",
        "exclude_persons": [],
        "must": [],
        "should": [],
        "exclude": [],
        "date_from": "",
        "date_to": "",
        "semantic_query": "people together",
        "keywords": [],
    }
    eng, _ = engine(db, settings, vocab, chat=[answer])
    r = eng.search("Anna or Ben")
    assert set(ids(r)) == {1, 2, 3, 4}
    assert set(ids(r)[:2]) == {1, 4}  # both persons


def test_deep_search_reranks_with_reasons(library, settings, vocab):
    db, _ = library
    rerank_answer = {
        "results": [
            {"id": 4, "score": 9, "reason": "Weihnachten mit der Familie"},
            {"id": 1, "score": 2, "reason": "Strand"},
            {"id": 999, "score": 10, "reason": "invented"},
        ]
    }
    from vision_tagger.vlm.backend import VlmError

    # first chat call (interpretation) fails -> keyword mode; second is the re-ranking
    eng, backend = engine(db, settings, vocab, chat=[VlmError("no LLM"), rerank_answer])
    r = eng.search("Familie Anna", deep=True)
    assert r.deep and ids(r)[0] == 4
    assert r.hits[0].reason == "Weihnachten mit der Familie"
    assert 999 not in ids(r)
    assert "people: Anna, Ben" in backend.chat_calls[-1][1]


def test_search_without_embeddings_still_works(db, settings, vocab):
    with db.write() as s:
        img = add_image(s, 1)
        s.add(ImageProperty(image_id=img.id, key="location_type", value="beach", source="vlm"))
        add_image(s, 2)
    eng, _ = engine(db, settings, vocab)
    r = eng.search("beach")
    assert not r.semantic_used and ids(r)[0] == 1


# --------------------------------------------------------------------------- Ollama payloads
def test_ollama_chat_json_and_embed(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.content:
            seen[request.url.path] = json.loads(request.content)
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[3.0, 4.0], [0.0, 2.0]]})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest"}]})
        return httpx.Response(404)

    backend = OllamaBackend(
        settings.vlm,
        client=httpx.Client(base_url="http://o", transport=httpx.MockTransport(handler)),
    )
    assert backend.chat_json("sys", "user", {"type": "object"}, model="qwen2.5:7b") == {"ok": True}
    chat = seen["/api/chat"]
    assert chat["model"] == "qwen2.5:7b" and "images" not in chat["messages"][1]
    vecs = backend.embed(["a", "b"], "nomic-embed-text")
    assert seen["/api/embed"]["input"] == ["a", "b"]
    assert np.allclose(vecs[0], [0.6, 0.8]) and np.allclose(vecs[1], [0.0, 1.0])
    assert backend.has_model("nomic-embed-text") and not backend.has_model("bge-m3")
