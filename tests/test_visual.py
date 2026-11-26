"""Visual search: model files, encoders, the ``visual`` stage, the search engine, UI and CLI."""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from PIL import ImageDraw
from sqlalchemy import func, select
from typer.testing import CliRunner

from vision_tagger import config as config_mod
from vision_tagger.animals.models import ModelFile, ModelUnavailable
from vision_tagger.cli import app
from vision_tagger.config import Settings
from vision_tagger.db.models import (
    Cluster,
    Image,
    ImageProperty,
    Person,
    StageStatus,
    Status,
    VisualEmbedding,
    VlmResult,
)
from vision_tagger.imaging import thumbnail_path
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline import control
from vision_tagger.pipeline.runner import BackgroundRunner, PipelineRunner
from vision_tagger.pipeline.stages.base import Stage, StageInterrupted
from vision_tagger.pipeline.stages.visual import VisualStage
from vision_tagger.search.engine import SearchEngine, visual_score
from vision_tagger.search.index import EmbeddingMatrix
from vision_tagger.visual import encoder as encoder_mod
from vision_tagger.visual.encoder import ImageEncoder, TextEncoder, preprocess
from vision_tagger.visual.models import (
    SIGLIP2_BASE,
    DownloadInterrupted,
    VisualModel,
    current_model,
    default_model_dir,
    ensure_files,
    visual_version,
)
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app

from .helpers import VISUAL_DIM, FakeVisualEncoder, add_face, add_image, person_centers

DE = {"Accept-Language": "de-DE,de;q=0.9"}


# --------------------------------------------------------------------------- model files
def _file(name: str, data: bytes) -> ModelFile:
    return ModelFile(
        name=name,
        url=f"https://example.invalid/{name}",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        license="Apache-2.0",
        title=f"Tiny {name}",
    )


FILES = {"vision": b"vision" * 500, "text": b"text" * 700, "tokenizer": b"{}" * 10}
TINY = VisualModel(
    key="tiny",
    title="Tiny",
    size=8,
    dim=4,
    files={name: _file(f"{name}.bin", data) for name, data in FILES.items()},
)


def _serve(calls: list[str], content=None):
    def handler(request):
        calls.append(str(request.url))
        name = request.url.path.rsplit("/", 1)[-1].removesuffix(".bin")
        return httpx.Response(200, content=content if content is not None else FILES[name])

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_model_files_are_downloaded_verified_and_report_progress(tmp_path):
    calls: list[str] = []
    reports: list[dict | None] = []
    paths = ensure_files(tmp_path, TINY, client=_serve(calls), report=reports.append)
    assert {n: p.read_bytes() for n, p in paths.items()} == FILES and len(calls) == 3
    total = sum(len(d) for d in FILES.values())
    downloads = [r for r in reports if r and r["phase"] == "download"]
    assert downloads[-1]["completed"] == total and downloads[-1]["total"] == total
    assert {r["stage"] for r in downloads} == {"visual"} and downloads[-1]["percent"] == 100
    assert any(r and r["phase"] == "verify" for r in reports) and reports[-1] is None
    ensure_files(tmp_path, TINY, client=_serve(calls))
    assert len(calls) == 3  # present and correct: not downloaded again
    paths["text"].write_bytes(b"x" * len(FILES["text"]))  # damaged: downloaded again
    ensure_files(tmp_path, TINY, client=_serve(calls))
    assert len(calls) == 4 and paths["text"].read_bytes() == FILES["text"]
    # only some files (the search needs no image encoder)
    only = ensure_files(tmp_path / "t", TINY, names=("tokenizer",), client=_serve(calls))
    assert list(only) == ["tokenizer"]


def test_model_download_errors_name_the_url_and_the_target(tmp_path):
    with pytest.raises(ModelUnavailable, match="not the expected file"):
        ensure_files(tmp_path, TINY, client=_serve([], content=b"evil"))
    assert list(tmp_path.iterdir()) == []

    def offline(request):
        raise httpx.ConnectError("no route to host")

    client = httpx.Client(transport=httpx.MockTransport(offline))
    with pytest.raises(ModelUnavailable, match=r"cannot download the Tiny vision\.bin") as err:
        ensure_files(tmp_path, TINY, client=client)
    spec = TINY.files["vision"]
    assert spec.url in str(err.value) and str(tmp_path / spec.name) in str(err.value)
    assert spec.sha256 in str(err.value)
    assert list(tmp_path.iterdir()) == []  # nothing half-written is left behind


def test_model_download_gives_way_to_pause_and_stop(tmp_path):
    with pytest.raises(DownloadInterrupted):
        ensure_files(tmp_path, TINY, client=_serve([]), interrupted=lambda: True)
    assert list(tmp_path.iterdir()) == []


def test_model_specs_and_settings(settings):
    assert current_model(settings) is SIGLIP2_BASE
    assert SIGLIP2_BASE.download_size == 371_807_752 + 564_862_230 + 34_363_039
    assert all(
        f.url.startswith("https://huggingface.co/onnx-community/")
        for f in SIGLIP2_BASE.files.values()
    )
    assert all(len(f.sha256) == 64 and "/resolve/" in f.url for f in SIGLIP2_BASE.files.values())
    assert visual_version(settings).startswith("siglip2-base-p16-224:c0573e3f")
    assert default_model_dir(settings).parts[-2:] == ("models", "visual")
    fast = Settings(
        visual={"model": "siglip2-base-fast", "model_dir": "/m"}, faces={"model_root": "/x"}
    )
    assert current_model(fast).key == "siglip2-base-p32-256" and current_model(fast).size == 256
    assert visual_version(fast) != visual_version(settings)
    assert str(default_model_dir(fast)).replace("\\", "/").endswith("/m")
    assert str(default_model_dir(Settings(faces={"model_root": "/x"}))).replace("\\", "/") == (
        "/x/models/visual"
    )


# --------------------------------------------------------------------------- encoders
def test_preprocess_squeezes_to_the_input_size_and_scales_to_minus_one_one():
    im = PILImage.new("RGB", (640, 480), (255, 0, 128))
    arr = preprocess(im, 224)
    assert arr.shape == (3, 224, 224) and arr.dtype == np.float32
    assert arr[0].min() == pytest.approx(1.0) and arr[1].max() == pytest.approx(-1.0)
    assert arr[2, 100, 100] == pytest.approx(128 * 2 / 255 - 1, abs=1e-5)
    assert preprocess(PILImage.new("L", (50, 300), 0), 32).shape == (3, 32, 32)  # grey, tall


class FakeSession:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.inputs: list[dict] = []

    def run(self, outputs, feeds):
        self.inputs.append(feeds)
        x = next(iter(feeds.values()))
        out = np.zeros((len(x), self.dim), dtype=np.float32)
        if x.dtype == np.int64:  # text: the first token id, scaled
            out[:, 0] = x[:, 0]
            out[:, 1] = 1.0
        else:
            out[:, 0] = x.reshape(len(x), -1).mean(axis=1) + 2.0
            out[:, 1] = 1.0
        return [out]

    def get_providers(self):
        return ["CPUExecutionProvider"]


def test_image_encoder_loads_once_and_returns_unit_vectors(settings, monkeypatch, tmp_path):
    session = FakeSession()
    loads: list[dict] = []

    def fake_files(folder, model, **kw):
        loads.append(kw)
        return {"vision": tmp_path / "v.onnx", "text": tmp_path / "t", "tokenizer": tmp_path / "k"}

    monkeypatch.setattr(encoder_mod, "ensure_files", fake_files)
    monkeypatch.setattr(encoder_mod, "_session", lambda path, providers, threads=0: session)
    enc = ImageEncoder(settings)
    assert enc.model == "siglip2-base-p16-224" and enc.size == 224
    progress = []
    enc.load(interrupted=lambda: False, report=progress.append)
    enc.load()
    assert len(loads) == 1 and loads[0]["report"] is not None  # downloads report to the stage
    pixels = np.stack([preprocess(PILImage.new("RGB", (64, 48), (200, 10, 10)), 224)] * 3)
    out = enc.embed(pixels)
    assert out.shape == (3, 4) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert enc.providers == ["CPUExecutionProvider"] and not enc.using_gpu
    enc.unload()
    assert enc._session is None


def _tokenizer_file(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"<pad>": 0, "<eos>": 1, "[UNK]": 2, "a": 3, "red": 4, "car": 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="$A <eos>", special_tokens=[("<eos>", 1)]
    )
    tok.save(str(path))


def test_text_encoder_lowercases_pads_caches_and_never_downloads(settings, monkeypatch, tmp_path):
    settings.visual.model_dir = tmp_path
    enc = TextEncoder(settings)
    with pytest.raises(ModelUnavailable, match="run the 'visual' stage"):
        enc.embed(["a red car"])
    _tokenizer_file(tmp_path / SIGLIP2_BASE.files["tokenizer"].name)
    (tmp_path / SIGLIP2_BASE.files["text"].name).write_bytes(b"onnx")
    session = FakeSession()
    opened = []
    monkeypatch.setattr(
        encoder_mod,
        "_session",
        lambda path, providers, threads=0: opened.append((providers, threads)) or session,
    )
    out = enc.embed(["Red  Car", "a red car"])
    ids = session.inputs[0]["input_ids"]
    assert ids.shape == (2, 64) and ids.dtype == np.int64
    assert ids[0, :4].tolist() == [4, 5, 1, 0]  # lower-cased, ended by <eos>, padded
    assert ids[1, :4].tolist() == [3, 4, 5, 1]
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert opened == [(["CPUExecutionProvider"], 4)]  # the text model runs on the CPU
    again = enc.embed(["red car"])  # cached (same text after normalising)
    assert len(session.inputs) == 1 and np.allclose(again[0], out[0])
    assert enc.low < enc.high and enc.model == SIGLIP2_BASE.key

    # unloaded when it is not used for a while (the cache stays), loaded again when needed
    enc.unload()
    enc.idle_seconds = 0.2
    enc.embed(["a car"])
    assert enc.loaded
    deadline = time.monotonic() + 5
    while enc.loaded and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not enc.loaded and len(opened) == 2
    enc.embed(["red car"])  # cached: no model needed
    assert not enc.loaded
    enc.embed(["a red"])
    assert enc.loaded and len(opened) == 3
    enc.unload()
    assert not enc.loaded


# --------------------------------------------------------------------------- the stage
def colour_photo(path, rgb, seed: int, size=(400, 300)):
    """A photo of mostly one colour with its own pattern (distinct perceptual hashes)."""
    rng = np.random.default_rng(seed)
    im = PILImage.new("RGB", size, rgb)
    draw = ImageDraw.Draw(im)
    for _ in range(6):
        x, y = int(rng.integers(0, size[0])), int(rng.integers(0, size[1]))
        r = int(rng.integers(15, 60))
        shade = tuple(max(0, min(255, c + int(rng.integers(-60, 60)))) for c in rgb)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=shade)
    for _ in range(3):
        x = int(rng.integers(0, size[0]))
        draw.line([x, 0, size[0] - x, size[1]], fill=(20, 20, 20), width=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return path


COLOURS = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220)}


@pytest.fixture
def coloured(db, settings, photos):
    """Six photos: two red, two green, two blue (indexed, with thumbnails)."""
    for i, name in enumerate(["red", "green", "blue", "red", "green", "blue"]):
        colour_photo(photos / f"{i}_{name}.jpg", COLOURS[name], seed=i)
    Indexer(db, settings, stages=["visual"]).run()
    return db


def vectors(db) -> dict[str, np.ndarray]:
    with db.session() as s:
        rows = s.execute(select(Image.filename, VisualEmbedding.vector).join(VisualEmbedding))
        return {name: np.frombuffer(v, dtype=np.float16) for name, v in rows}


def test_stage_embeds_every_photo_in_batches_with_prefetch(coloured, settings):
    settings.visual.batch_size = 4
    encoder = FakeVisualEncoder()
    stage = VisualStage(coloured, settings, encoder=encoder)
    assert stage.prefetch > 0 and stage.batch == 4 and stage.redo_outdated
    summary = PipelineRunner(coloured, settings, lambda _n: stage).run(["visual"], batch_size=6)
    assert summary.status == "completed" and summary.processed == {"visual": 6}
    assert encoder.embedded == 6 and not encoder.loaded  # unloaded after the stage
    with coloured.session() as s:
        rows = s.scalars(select(VisualEmbedding)).all()
        assert len(rows) == 6
        assert {(r.model, r.stage_version, r.dim) for r in rows} == {
            ("fake-visual", "fake-visual-v1", VISUAL_DIM)
        }
        done = s.scalars(select(StageStatus).where(StageStatus.stage == "visual")).all()
        assert {(st.status, st.stage_version) for st in done} == {(Status.DONE, "fake-visual-v1")}
    for name, vec in vectors(coloured).items():
        assert np.linalg.norm(vec.astype(np.float32)) == pytest.approx(1.0, abs=1e-2)
        channel = int(np.argmax(vec[:3]))
        assert ["red", "green", "blue"][channel] in name  # the photo's colour

    # one photo per model call: prefetched one by one
    settings.visual.batch_size = 1
    one = FakeVisualEncoder()
    one.version = "fake-visual-v2"
    stage = VisualStage(coloured, settings, encoder=one)
    assert stage.batch == 1
    summary = PipelineRunner(coloured, settings, lambda _n: stage).run(["visual"])
    assert summary.processed == {"visual": 6} and one.embedded == 6


def test_stage_reads_the_thumbnail_and_falls_back_to_the_original(coloured, settings, monkeypatch):
    from vision_tagger.pipeline.stages import visual as visual_stage

    opened: list[str] = []
    real = visual_stage.load_image
    monkeypatch.setattr(
        visual_stage, "load_image", lambda path, **kw: opened.append(str(path)) or real(path, **kw)
    )
    stage = VisualStage(coloured, settings, encoder=FakeVisualEncoder())
    stage.prepare(1)
    assert opened == []  # the gallery thumbnail is large enough
    thumbnail_path(settings.thumbs_dir, 1, settings.web.thumb_size).unlink()
    arr = stage.prepare(1)
    assert len(opened) == 1 and arr.shape == (3, 16, 16)
    stage.encoder.size = 1000  # thumbnail smaller than the model input: the original
    stage.prepare(2)
    assert len(opened) == 2


def test_missing_files_fail_alone(coloured, settings):
    with coloured.session() as s:
        path = s.get(Image, 3).path
    from pathlib import Path

    Path(path).unlink()
    thumbnail_path(settings.thumbs_dir, 3, settings.web.thumb_size).unlink()
    stage = VisualStage(coloured, settings, encoder=FakeVisualEncoder())
    summary = PipelineRunner(coloured, settings, lambda _n: stage).run(["visual"])
    assert summary.failed == {"visual": 1} and summary.processed == {"visual": 6}
    with coloured.session() as s:
        assert s.scalar(select(func.count()).select_from(VisualEmbedding)) == 5
        assert s.get(StageStatus, (3, "visual")).status == Status.ERROR


def test_a_new_model_version_embeds_again(coloured, settings):
    run = PipelineRunner(coloured, settings, lambda _n: stage)
    stage = VisualStage(coloured, settings, encoder=FakeVisualEncoder())
    run.run(["visual"])
    again = FakeVisualEncoder()
    stage = VisualStage(coloured, settings, encoder=again)
    assert run.run(["visual"]).processed == {"visual": 0} and again.embedded == 0
    newer = FakeVisualEncoder()
    newer.version = "fake-visual-v2"
    stage = VisualStage(coloured, settings, encoder=newer)
    assert run.run(["visual"]).processed == {"visual": 6} and newer.embedded == 6
    with coloured.session() as s:
        assert set(s.scalars(select(VisualEmbedding.stage_version))) == {"fake-visual-v2"}
        assert s.scalar(select(func.count()).select_from(VisualEmbedding)) == 6  # replaced


def test_stop_and_pause_during_the_stage(coloured, settings):
    settings.visual.batch_size = 2

    def stop(_n):
        control.request_state(coloured, control.STOPPED)

    stage = VisualStage(coloured, settings, encoder=FakeVisualEncoder(on_embed=stop))
    summary = PipelineRunner(coloured, settings, lambda _n: stage).run(["visual"], batch_size=6)
    assert summary.status == "stopped" and summary.processed == {"visual": 2}
    with coloured.session() as s:
        statuses = [st.status for st in s.scalars(select(StageStatus))]
        assert statuses.count(Status.DONE) == 2 and statuses.count(Status.PENDING) == 4

    paused = threading.Event()

    def pause(_n):
        if not paused.is_set():
            control.request_state(coloured, control.PAUSED)
            paused.set()

    encoder = FakeVisualEncoder(on_embed=pause)
    stage = VisualStage(coloured, settings, encoder=encoder)
    runner = PipelineRunner(coloured, settings, lambda _n: stage)
    runner.poll_interval = 0.05
    bg = BackgroundRunner(runner)
    control.request_state(coloured, control.RUNNING)
    assert bg.start(["visual"], batch_size=6)
    assert paused.wait(5)
    time.sleep(0.3)
    assert bg.running and encoder.embedded == 2  # waits after the first batch
    control.request_state(coloured, control.RUNNING)
    bg.join(10)
    assert bg.last_summary.status == "completed" and encoder.embedded == 4


class OtherStage(Stage):
    name = "faces"

    def __init__(self, db, settings):
        super().__init__(db, settings)
        self.done: list[int] = []

    def process(self, image_id):
        self.done.append(image_id)


def test_without_the_model_the_stage_is_skipped_and_the_others_run(coloured, settings):
    class Offline(FakeVisualEncoder):
        def load(self, interrupted=None, report=None):
            raise ModelUnavailable("cannot download the SigLIP 2 image encoder (offline)")

    other = OtherStage(coloured, settings)
    stages = {"visual": VisualStage(coloured, settings, encoder=Offline()), "faces": other}
    summary = PipelineRunner(coloured, settings, stages.__getitem__).run(["visual", "faces"])
    assert summary.status == "completed" and len(other.done) == 6
    assert any("visual skipped" in m and "offline" in m for m in summary.messages)
    with coloured.session() as s:
        rows = s.scalars(select(StageStatus).where(StageStatus.stage == "visual")).all()
        assert {(r.status, r.attempts) for r in rows} == {(Status.PENDING, 0)}  # next run

    class Interrupted(FakeVisualEncoder):
        def load(self, interrupted=None, report=None):
            raise DownloadInterrupted("download of the SigLIP 2 text encoder interrupted")

    with pytest.raises(StageInterrupted):
        VisualStage(coloured, settings, encoder=Interrupted()).setup()


# --------------------------------------------------------------------------- the engine
def unit(*values: float) -> np.ndarray:
    v = np.zeros(VISUAL_DIM, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


#: n: (vector or None, date, persons, described by the vision model (caption, props), phash)
VISUAL_LIBRARY = {
    1: (unit(1, 0.05, 0.05), datetime(2021, 7, 1), ["Anna"], ("a red car", ["street"]), 0x00FF),
    # (photo 1 also has dominant_colors = red, see the fixture)
    2: (unit(0.05, 1, 0.05), datetime(2020, 5, 1), ["Ben"], None, 0xF0F0),
    3: (unit(0.05, 0.05, 1), datetime(2019, 3, 1), ["Anna"], None, 0x0F0F),
    4: (unit(0.8, 0.35, 0.1), datetime(2018, 2, 1), [], None, 0xFF00),
    5: (unit(0.05, 1, 0.1), datetime(2021, 8, 1), [], ("green trees", ["beach"]), 0x3333),
    6: (None, datetime(2017, 1, 1), [], ("a red apple", ["kitchen"]), 0x5555),
}


@pytest.fixture
def visual_db(db):
    centers = person_centers(2)
    with db.write() as s:
        clusters = {}
        for name in ("Anna", "Ben"):
            clusters[name] = Cluster(locked=True, person=Person(name=name))
            s.add(clusters[name])
        s.flush()
        for n, (vec, when, persons, described, phash) in VISUAL_LIBRARY.items():
            img = add_image(s, n, recorded_at=when)
            img.phash = f"{phash:016x}"
            s.add(ImageProperty(image_id=img.id, key="year", value=str(when.year), source="exif"))
            if n == 1:
                s.add(ImageProperty(image_id=1, key="dominant_colors", value="red", source="vlm"))
            for i, name in enumerate(persons):
                add_face(s, img.id, centers[i], cluster_id=clusters[name].id)
            if described:
                caption, locations = described
                s.add(
                    VlmResult(
                        image_id=img.id,
                        model="fake",
                        prompt_version="p",
                        caption=caption,
                        raw_json={"caption": caption},
                    )
                )
                for value in locations:
                    s.add(
                        ImageProperty(
                            image_id=img.id, key="location_type", value=value, source="vlm"
                        )
                    )
            if vec is not None:
                s.add(
                    VisualEmbedding(
                        image_id=img.id,
                        model="fake-visual",
                        stage_version="fake-visual-v1",
                        dim=VISUAL_DIM,
                        vector=vec.astype(np.float16).tobytes(),
                    )
                )
    return db


def visual_engine(db, settings, chat=None):
    backend = FakeBackend(Vocabulary.load(), chat_responses=chat or [])
    return SearchEngine(db, settings, backend=backend, text_encoder=FakeVisualEncoder())


def ids(result) -> list[int]:
    return [h.image_id for h in result.hits]


def llm(**fields):
    base = {
        "persons": [],
        "persons_mode": "all",
        "exclude_persons": [],
        "pets": [],
        "exclude_pets": [],
        "must": [],
        "should": [],
        "exclude": [],
        "any_of": [],
        "date_from": "",
        "date_to": "",
        "semantic_query": "",
        "keywords": [],
    }
    return {**base, **fields}


def test_visual_score_is_calibrated_and_keeps_the_order():
    assert visual_score(0.03, 0.04, 0.12) == 0.0 and visual_score(0.04, 0.04, 0.12) == 0.0
    assert visual_score(0.12, 0.04, 0.12) == pytest.approx(0.75)
    scores = [visual_score(c, 0.04, 0.12) for c in (0.06, 0.1, 0.14, 0.2, 0.3)]
    assert scores == sorted(scores) and scores[-1] < 1.0


def test_basic_mode_finds_photos_by_what_they_show_in_english_and_german(visual_db, settings):
    eng = visual_engine(visual_db, settings)
    r = eng.search("red")
    assert r.interpretation.mode == "basic" and r.visual_used
    assert ids(r) == [1, 4]  # the red photos by what they show
    assert [(t.key, t.value) for t in r.interpretation.should] == [("dominant_colors", "red")]
    assert r.hits[0].visual > r.hits[1].visual > 0
    assert ids(eng.search("rot"))[:2] == [1, 4]  # German
    assert ids(eng.search("zebra")) == []  # nothing matches: no random photos
    r = eng.search("apple")
    assert ids(r) == [6] and r.hits[0].visual is not None  # no vector: found by its caption


def test_the_visual_query_leaves_out_names_places_dates_and_exclusions(visual_db, settings):
    eng = visual_engine(visual_db, settings)
    assert eng.interpret("photos of Anna in red from 2021").visual_query == "red"
    assert eng.interpret("Anna oder Ben, blau").visual_query == "blau"
    assert eng.interpret("blue without green").visual_query == "blue"
    assert eng.interpret("Rote Autos ohne Anna").visual_query == "Rote Autos"  # as typed
    assert eng.interpret("Straße ohne Anna").visual_query == "strasse"  # (case-folded)
    assert eng.interpret("Anna").visual_query == ""  # nothing to look at
    it = eng.interpret("Anna am Strand")
    assert it.persons and it.visual_query == "Strand"


def test_hybrid_with_persons_alternatives_dates_and_exclusions(visual_db, settings):
    eng = visual_engine(visual_db, settings)
    r = eng.search("Anna red")
    assert ids(r) == [1, 3]  # the strict candidates stay, the red one first
    assert eng.text_encoder.texts[-1] == "red"
    r = eng.search("Anna or Ben blue")
    assert set(ids(r)) == {1, 2, 3} and ids(r)[0] == 3
    r = eng.search("red without Anna")
    assert ids(r)[0] == 4 and 1 not in ids(r) and 3 not in ids(r)
    r = eng.search("red photos from 2018 or 2021")
    assert set(ids(r)) == {1, 4, 5} and set(ids(r)[:2]) == {1, 4}


def test_llm_mode_uses_the_english_rewrite(visual_db, settings):
    eng = visual_engine(visual_db, settings, chat=[llm(semantic_query="a green photo")])
    r = eng.search("ein grünes Foto")
    assert r.interpretation.mode == "llm" and eng.text_encoder.texts[-1] == "a green photo"
    assert set(ids(r)[:2]) == {2, 5}


def test_photos_without_a_description_stay_candidates_for_required_properties(visual_db, settings):
    answer = llm(must=[{"key": "location_type", "value": "street"}], semantic_query="red")
    r = visual_engine(visual_db, settings, chat=[answer]).search("a red street")
    assert ids(r)[:2] == [1, 4]  # 4 has no description yet, but shows red
    assert r.hits[0].properties == 1.0 and r.hits[1].properties == 0.0
    assert not {2, 3, 5, 6} & set(ids(r))  # described elsewhere, or showing nothing red
    assert any("without a description yet" in str(n) for n in r.notes)
    # a property every photo has from the start (year) stays strict
    answer = llm(must=[{"key": "year", "value": "2021"}], semantic_query="red")
    r = visual_engine(visual_db, settings, chat=[answer]).search("red in 2021")
    assert set(ids(r)) == {1, 5}


def test_coverage_note_and_no_visual_index(visual_db, settings, db):
    r = visual_engine(visual_db, settings).search("red")
    assert r.visual_coverage == pytest.approx(5 / 6)
    assert any("83%" in str(n) and "Visual search" in str(n) for n in r.notes)
    with visual_db.write() as s:
        s.execute(VisualEmbedding.__table__.delete())
    eng = visual_engine(visual_db, settings)
    r = eng.search("red")
    assert not r.visual_used and r.visual_coverage == 0 and eng.text_encoder.texts == []
    assert ids(r) == [1]  # the colour property still counts
    assert ids(eng.search("apple")) == [6]  # and the caption


def test_photos_are_scored_by_the_parts_they_have(visual_db, settings):
    settings.search.weights.keywords = 0.0
    r = visual_engine(visual_db, settings).search("red")
    by_id = {h.image_id: h for h in r.hits}
    # photo 4 has no description: its score is its visual score, not diluted by the colour
    # property it can't have yet
    assert by_id[4].properties == 0 and by_id[4].score == pytest.approx(by_id[4].visual, abs=1e-3)
    # photo 1 is described: the property counts
    assert by_id[1].properties == 1.0
    assert by_id[1].score == pytest.approx((0.45 * by_id[1].visual + 0.2) / 0.65, abs=1e-3)
    with visual_db.write() as s:
        s.execute(ImageProperty.__table__.delete().where(ImageProperty.key == "dominant_colors"))
    r = visual_engine(visual_db, settings).search("red")
    assert ids(r) == [1, 4]  # a property no photo has tells nothing
    with visual_db.write() as s:
        s.add(ImageProperty(image_id=6, key="dominant_colors", value="red", source="vlm"))
    r = visual_engine(visual_db, settings).search("red")
    assert ids(r) == [6, 4, 1]  # described without the colour: after the unknown one


def test_deep_search_judges_described_photos_and_keeps_the_others_in_place(visual_db, settings):
    rerank = {"results": [{"id": 1, "score": 2, "reason": "a car"}]}
    eng = visual_engine(visual_db, settings, chat=[ValueError("no LLM"), rerank])
    r = eng.search("red", deep=True)
    assert ids(r)[:2] == [1, 4] and r.deep
    assert r.hits[0].reason == "a car" and r.hits[1].rerank is None
    assert "[4]" not in eng.backend.chat_calls[-1][1]  # nothing to read about photo 4


def test_similar_photos_leave_out_the_photo_and_copies(visual_db, settings):
    with visual_db.write() as s:
        s.get(Image, 4).phash = f"{0x00FF ^ 0b101:016x}"  # 2 bits from photo 1: a copy
        img = add_image(s, 7)
        img.phash = f"{0x7E81:016x}"
        s.add(
            VisualEmbedding(
                image_id=img.id,
                model="fake-visual",
                stage_version="fake-visual-v1",
                dim=VISUAL_DIM,
                vector=unit(0.9, 0.3, 0.2).astype(np.float16).tobytes(),
            )
        )
    eng = visual_engine(visual_db, settings)
    r = eng.similar(1)
    assert r.similar_to == 1 and r.visual_used
    assert 1 not in ids(r) and 4 not in ids(r)
    assert ids(r)[0] == 7 and len(ids(r)) == 4  # 7, then the others; 6 has no vector
    assert r.hits[0].score == pytest.approx(r.hits[0].visual, abs=1e-3)
    assert ids(eng.similar(1, limit=2)) == [7, 5] or ids(eng.similar(1, limit=2))[0] == 7
    none = eng.similar(6)
    assert none.hits == [] and any("no visual vector" in str(n) for n in none.notes)


def test_the_matrix_loads_one_model_and_follows_changes(visual_db):
    m = EmbeddingMatrix(visual_db, VisualEmbedding, dtype=np.float32)
    m.ensure("fake-visual")
    assert m.count == 5 and m.has(1) and not m.has(6) and m.nbytes == 5 * VISUAL_DIM * 4
    half = EmbeddingMatrix(visual_db, VisualEmbedding)  # float16: the same similarities
    half.ensure("fake-visual")
    assert half.nbytes == 5 * VISUAL_DIM * 2
    for few in ([1], [1, 2, 3, 4, 5]):  # gathered rows / one pass over all
        a, b = m.similarities(unit(1, 0, 0), few), half.similarities(unit(1, 0, 0), few)
        assert a.keys() == b.keys() and all(abs(a[i] - b[i]) < 1e-3 for i in a)
    assert [i for i, _ in m.nearest(unit(1, 0, 0), 2)] == [1, 4]
    assert m.similarities(unit(1, 0, 0), [1, 6, 99]).keys() == {1}
    with visual_db.write() as s:
        s.add(
            VisualEmbedding(
                image_id=6,
                model="fake-visual",
                stage_version="fake-visual-v1",
                dim=VISUAL_DIM,
                vector=unit(1, 0, 0).astype(np.float16).tobytes(),
            )
        )
    m.ensure("fake-visual")
    assert m.count == 6 and m.nearest(unit(1, 0, 0), 1)[0][0] == 6
    m.ensure("other-model")
    assert m.count == 0 and m.nearest(unit(1, 0, 0), 3) == []


# --------------------------------------------------------------------------- UI and CLI
@pytest.fixture
def web(settings, photos):
    for i, name in enumerate(["red", "green", "blue", "red"]):
        colour_photo(photos / f"{i}_{name}.jpg", COLOURS[name], seed=i)
    app_ = create_app(settings)
    with TestClient(app_) as c:
        ctx = app_.state.ctx
        Indexer(ctx.db, settings).run()
        stage = VisualStage(ctx.db, settings, encoder=FakeVisualEncoder())
        PipelineRunner(ctx.db, settings, lambda _n: stage).run(["visual"])
        ctx.search.text_encoder = FakeVisualEncoder()
        c.ctx = ctx
        yield c


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def test_search_results_show_the_visual_part_in_english_and_german(web):
    page = ok(web.get("/search/results?q=red")).text
    assert "visual · 100% indexed" in page and "basic mode" in page
    assert page.index("/media/thumb/1") < page.index("/media/thumb/4")
    assert "/media/thumb/2" not in page  # green: nothing about it matches
    assert "visual 0." in page  # the score breakdown in the tooltip
    de = ok(web.get("/search/results?q=rot", headers=DE)).text
    assert "visuell · 100 % indexiert" in de and "/media/thumb/1" in de
    page = ok(web.get("/search", headers=DE)).text
    assert "was die Fotos zeigen" in page


def test_similar_photos_button_and_page(web):
    page = ok(web.get("/images/1")).text
    assert 'href="/search?similar=1"' in page and "Similar photos" in page
    assert "Ähnliche Fotos" in ok(web.get("/images/1", headers=DE)).text
    page = ok(web.get("/search?similar=1")).text
    assert "<h1>Similar photos</h1>" in page and "/search/results?similar=1" in page
    assert 'class="searchbox"' not in page and 'class="similar-source"' in page  # the photo
    results = ok(web.get("/search/results?similar=1")).text
    assert "/media/thumb/4" in results and 'href="/images/1"' not in results
    assert "Open in filter builder" not in results and "visual · 100% indexed" in results
    assert results.index("/media/thumb/4") < results.index("/media/thumb/2")
    de = ok(web.get("/search?similar=1", headers=DE)).text
    assert "Ähnliche Fotos" in de and "Zurück zum Foto" in de
    assert web.get("/search?similar=999").status_code == 404


def test_pipeline_page_lists_the_visual_stage(web):
    page = ok(web.get("/pipeline")).text
    assert 'value="visual"' in page and "Visual search (siglip2-base)" in page
    status = ok(web.get("/partials/status")).text
    assert "Stage 0 · SigLIP 2" in status


def test_cli_search_and_similar(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    for i, name in enumerate(["red", "green", "red"]):
        colour_photo(photos / f"{i}_{name}.jpg", COLOURS[name], seed=i)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[paths]\n"
        f"image_roots = ['{photos.as_posix()}']\n"
        f"data_dir = '{(tmp_path / 'data').as_posix()}'\n"
        "[vlm]\nbackend = 'fake'\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    from vision_tagger.pipeline.stages import visual as visual_stage

    monkeypatch.setattr(visual_stage, "ImageEncoder", lambda settings: FakeVisualEncoder())
    monkeypatch.setattr(encoder_mod, "TextEncoder", lambda settings: FakeVisualEncoder())

    def invoke(*args):
        result = CliRunner().invoke(app, ["-c", str(cfg), *args])
        assert result.exit_code == 0, result.output + repr(result.exception)
        return result.output

    invoke("index")
    assert "completed" in invoke("run", "--stage", "visual")
    out = invoke("search", "red")
    assert "visual=on" in out and "looking for='red'" in out
    lines = [line for line in out.splitlines() if ".jpg" in line]
    assert len(lines) == 2 and all("_red.jpg" in line for line in lines)
    out = invoke("similar", "1")
    assert "2_red.jpg" in out.splitlines()[0] and "0_red.jpg" not in out
    assert "2_red.jpg" in invoke("similar", str(photos / "0_red.jpg")).splitlines()[0]
    result = CliRunner().invoke(app, ["-c", str(cfg), "similar", "999"])
    assert result.exit_code == 1


# --------------------------------------------------------------------------- the real model
def _real_model_dir():
    settings = Settings()
    folder = default_model_dir(settings)
    spec = current_model(settings)
    if not all((folder / f.name).is_file() for f in spec.files.values()):
        pytest.skip(f"the SigLIP 2 files are not in {folder} (run the visual stage once)")
    return settings


@pytest.mark.integration
def test_real_siglip2_matches_text_and_pictures_in_english_and_german(tmp_path):
    settings = _real_model_dir()
    image_encoder, text_encoder = ImageEncoder(settings), TextEncoder(settings)
    pictures = {}
    for name, rgb in COLOURS.items():
        pictures[name] = colour_photo(tmp_path / f"{name}.jpg", rgb, seed=1)
    pixels = np.stack([image_encoder.preprocess(PILImage.open(p)) for p in pictures.values()])
    photo_vectors = image_encoder.embed(pixels)
    queries = ["a red picture", "a green picture", "a blue picture"]
    queries += ["ein rotes Bild", "ein grünes Bild", "ein blaues Bild"]
    sims = text_encoder.embed(queries) @ photo_vectors.T
    assert sims.argmax(axis=1).tolist() == [0, 1, 2, 0, 1, 2]
