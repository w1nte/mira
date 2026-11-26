"""Ollama models: which ones are needed, downloading them (web UI, CLI, pipeline runs)."""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from typer.testing import CliRunner

from vision_tagger import config as config_mod
from vision_tagger import i18n
from vision_tagger.cli import app as cli_app
from vision_tagger.db.models import Control, StageStatus, Status
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline import control
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.base import StageAbort
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.vlm import VlmStage
from vision_tagger.vlm import models
from vision_tagger.vlm.backend import FakeBackend, OllamaBackend
from vision_tagger.vlm.models import (
    OllamaUnreachable,
    PullCancelled,
    PullError,
    pull,
    pull_message,
    required_models,
)
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app
from vision_tagger.web.context import Job

from .conftest import make_image
from .fake_ollama import FakeOllama

DE = {"Accept-Language": "de-DE,de;q=0.9"}
VLM = "qwen2.5vl:7b"
EMBED = "nomic-embed-text:latest"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ollama_settings(settings, url: str):
    settings.vlm.backend = "ollama"
    settings.vlm.base_url = url
    return settings


def wait_for(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def ndjson(lines: list[dict]) -> bytes:
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))


@pytest.fixture
def lang():
    tokens = []
    yield lambda code: tokens.append(i18n.activate(code))
    for token in reversed(tokens):
        i18n.reset(token)


# --------------------------------------------------------------------------- required models
def test_required_models_are_normalised_and_deduplicated(settings):
    settings.vlm.backend = "ollama"
    req = required_models(settings)
    assert [(m.name, m.key) for m in req] == [(VLM, "vlm"), (EMBED, "embed")]
    assert req[0].purposes == ("Properties", "Search queries")  # search.model defaults to it
    assert req[1].purposes == ("Search index",)

    settings.search.model = "qwen2.5:7b"
    req = required_models(settings)
    assert [(m.name, m.key) for m in req] == [
        (VLM, "vlm"),
        (EMBED, "embed"),
        ("qwen2.5:7b", "search"),
    ]
    assert req[0].purposes == ("Properties",)

    settings.search.model = "qwen2.5vl:7b"  # the same model as vlm.model: listed once
    assert [m.name for m in required_models(settings)] == [VLM, EMBED]
    settings.search.embed_model = "qwen2.5vl:7b"
    assert [m.purposes for m in required_models(settings)] == [
        ("Properties", "Search index", "Search queries")
    ]

    assert models.normalize("bge-m3") == "bge-m3:latest"
    assert models.normalize("host:5000/team/model") == "host:5000/team/model:latest"
    assert models.normalize("host:5000/team/model:q4") == "host:5000/team/model:q4"
    assert models.missing_models(required_models(settings), ["qwen2.5vl:7b"]) == []

    settings.vlm.backend = "fake"
    assert required_models(settings) == []


# --------------------------------------------------------------------------- pull
def test_pull_adds_up_the_layers(settings):
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 4000},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 4000, "completed": 1000},
        {"status": "pulling bbb", "digest": "sha256:bbb", "total": 1000, "completed": 500},
        {"status": "pulling aaa", "digest": "sha256:aaa", "total": 4000, "completed": 4000},
        {"status": "pulling bbb", "digest": "sha256:bbb", "total": 1000, "completed": 1000},
        {"status": "verifying sha256 digest"},
        {"status": "writing manifest"},
        {"status": "removing any unused layers"},
        {"status": "success"},
    ]
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, content=ndjson(lines))

    steps = []
    result = pull(
        EMBED, client=mock_client(handler), on_progress=lambda p: steps.append(p.as_dict())
    )
    assert seen == {"path": "/api/pull", "body": {"model": EMBED, "stream": True}}
    assert [(s["completed"], s["total"], s["percent"]) for s in steps[:6]] == [
        (0, 0, None),
        (0, 4000, 0),
        (1000, 4000, 25),
        (1500, 5000, 30),  # the second layer adds to the total
        (4500, 5000, 90),
        (5000, 5000, 100),
    ]
    assert [s["phase"] for s in steps] == [
        "manifest",
        *["download"] * 5,
        "verify",
        "write",
        "write",
        "done",
    ]
    assert result.phase == "done" and result.percent == 100 and result.total == 5000


@pytest.mark.parametrize(
    ("error", "expected", "german"),
    [
        ("pull model manifest: file does not exist", "does not know the model", "kennt das Modell"),
        (
            "write /root/.ollama/x: no space left on device",
            "Not enough disk space",
            "Speicherplatz",
        ),
        ("There is not enough space on the disk.", "Not enough disk space", "Speicherplatz"),
        (
            'Get "https://registry.ollama.ai/v2/": dial tcp: lookup registry.ollama.ai: '
            "no such host",
            "Check the internet connection",
            "Internetverbindung",
        ),
        ("something else broke", "could not download", "konnte"),
    ],
)
def test_pull_error_lines_are_explained(error, expected, german, lang):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=ndjson([{"status": "pulling manifest"}, {"error": error}])
        )

    with pytest.raises(PullError) as info:
        pull("some-model", client=mock_client(handler))
    assert expected in str(info.value) and "some-model" in str(info.value)
    lang("de")
    assert german in str(info.value.text)  # translated when shown


def test_pull_reports_http_errors_and_early_ends():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "pull model manifest: file does not exist"})

    with pytest.raises(PullError, match="does not know the model x:1"):
        pull("x:1", client=mock_client(failing))

    def short(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson([{"status": "pulling manifest"}]))

    with pytest.raises(PullError, match="ended early"):
        pull("x:1", client=mock_client(short))


def test_pull_says_how_to_get_ollama_when_it_is_not_reachable():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OllamaUnreachable, match=r"https://ollama\.com/download"):
        pull("x:1", client=mock_client(refused))
    url = f"http://127.0.0.1:{free_port()}"
    with pytest.raises(OllamaUnreachable, match="not reachable at " + url):
        pull("x:1", base_url=url, connect_timeout=2)


def test_pull_can_be_cancelled_between_lines():
    lines = [{"status": "pulling manifest"}] + [
        {"status": "pulling a", "digest": "sha256:a", "total": 100, "completed": i}
        for i in range(50)
    ]
    steps = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson([*lines, {"status": "success"}]))

    with pytest.raises(PullCancelled):
        pull(
            "x:1",
            client=mock_client(handler),
            on_progress=steps.append,
            cancel=lambda: len(steps) >= 3,
        )
    assert len(steps) == 3
    with pytest.raises(PullCancelled):  # cancelled before it started: no request at all
        pull("x:1", client=mock_client(lambda r: pytest.fail("no request")), cancel=lambda: True)


def test_pull_is_cancelled_while_ollama_is_silent():
    """Ollama sends nothing while it verifies a big layer; cancelling must not wait for it."""
    with FakeOllama(silence=20) as ollama:
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(PullCancelled):
            pull(EMBED, base_url=ollama.url, cancel=cancel.is_set)
        assert time.monotonic() - started < 5
        assert EMBED not in ollama.installed


def test_pull_missing_downloads_only_what_is_missing_one_after_another(settings):
    settings.search.model = "qwen2.5:7b"
    with FakeOllama(installed=[VLM], delay=0.01) as ollama:
        ollama_settings(settings, ollama.url)
        started = []
        pulled = models.pull_missing(settings, on_start=lambda i, todo: started.append((i, todo)))
        assert pulled == [EMBED, "qwen2.5:7b"]
        assert ollama.pulls == [EMBED, "qwen2.5:7b"] and ollama.max_active == 1
        assert started == [(0, pulled), (1, pulled)]
        assert models.pull_missing(settings) == []  # nothing left to do
        settings.search.embed_model = "bge-m3"
        assert models.pull_missing(settings, keys=["search"]) == []
        assert models.pull_missing(settings, keys=["embed"]) == ["bge-m3:latest"]
    with pytest.raises(OllamaUnreachable):
        models.pull_missing(settings)  # the server is gone


def test_pull_message_is_translated_when_shown(lang):
    progress = {"model": VLM, "phase": "download", "completed": 1_234_000_000, "total": 6e9}
    message = pull_message({**progress, "percent": 20, "index": 0, "count": 2})
    assert str(message) == "1/2 · qwen2.5vl:7b · 1.2 GB of 6.0 GB (20 %)"
    assert (
        str(pull_message({"model": EMBED, "phase": "verify"})) == f"{EMBED} · checking the download"
    )
    lang("de")
    assert str(message) == "1/2 · qwen2.5vl:7b · 1.2 GB von 6.0 GB (20 %)"


# --------------------------------------------------------------------------- pipeline: auto pull
class PullingBackend(FakeBackend):
    """Has no models until ``pull``; ``block`` makes a download run until it is cancelled."""

    def __init__(self, vocab, *, block=False):
        super().__init__(vocab)
        self.installed: set[str] = set()
        self.block = block
        self.pulls: list[str] = []
        self.pulling = threading.Event()

    def has_model(self, name):
        return models.normalize(name) in self.installed

    def health(self):
        ok = models.normalize(self.model) in self.installed
        return {"backend": "fake", "ok": ok, "error": None if ok else "not pulled"}

    def pull(self, name, on_progress=None, cancel=None):
        self.pulls.append(name)
        progress = models.PullProgress(model=name)
        progress.update(
            {"status": "pulling a", "digest": "sha256:a", "total": 100, "completed": 40}
        )
        if on_progress:
            on_progress(progress)
        self.pulling.set()
        while self.block and len(self.pulls) == 1:
            if cancel and cancel():
                self.pulling.clear()
                raise PullCancelled(name)
            time.sleep(0.02)
        self.installed.add(name)
        return progress


@pytest.fixture
def vocab():
    return Vocabulary.load()


@pytest.fixture
def indexed(db, settings, photos):
    for i in range(3):
        make_image(photos / f"img{i}.jpg", seed=i)
    Indexer(db, settings, stages=["vlm", "embed"]).run()
    return db


def test_embed_stage_downloads_a_missing_model_first(indexed, settings, vocab):
    backend = PullingBackend(vocab)
    stage = EmbedStage(indexed, settings, backend=backend)
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["embed"])
    assert summary.status == "completed", summary.messages
    assert backend.pulls == [EMBED]
    assert control.pull_progress(indexed) is None  # cleared after the download


def test_vlm_stage_downloads_a_missing_model_first(indexed, settings, vocab):
    backend = PullingBackend(vocab)
    backend.model = VLM
    stage = VlmStage(indexed, settings, backend=backend)
    stage.setup()
    assert backend.pulls == [VLM]


def test_without_auto_pull_the_stage_asks_for_ollama_pull(indexed, settings, vocab):
    settings.vlm.auto_pull = False
    backend = PullingBackend(vocab)
    with pytest.raises(StageAbort, match="ollama pull nomic-embed-text"):
        EmbedStage(indexed, settings, backend=backend).setup()
    backend.model = VLM
    with pytest.raises(RuntimeError, match="not pulled"):
        VlmStage(indexed, settings, backend=backend).setup()
    assert backend.pulls == []


def test_a_failed_download_stops_the_run_without_burning_attempts(indexed, settings, vocab):
    class Failing(PullingBackend):
        def pull(self, name, on_progress=None, cancel=None):
            raise PullError(models.error_text(name, "no space left on device"))

    stage = EmbedStage(indexed, settings, backend=Failing(vocab))
    summary = PipelineRunner(indexed, settings, lambda _n: stage).run(["embed"])
    assert summary.status == "failed"
    assert "Not enough disk space for nomic-embed-text:latest" in summary.messages[0]
    with indexed.session() as s:
        rows = list(s.scalars(select(StageStatus).where(StageStatus.stage == "embed")))
    assert {r.status for r in rows} == {Status.PENDING} and {r.attempts for r in rows} == {0}


def test_embed_stage_downloads_through_ollama(indexed, settings):
    with FakeOllama(installed=[VLM]) as ollama:
        ollama_settings(settings, ollama.url)
        stage = EmbedStage(indexed, settings, backend=OllamaBackend(settings.vlm))
        stage.setup()
        assert ollama.finished == [EMBED]


def run_in_thread(runner, stages):
    out = {}
    thread = threading.Thread(target=lambda: out.update(summary=runner.run(stages)))
    thread.start()
    return thread, out


def test_pause_during_the_download_cancels_it_and_resume_continues(indexed, settings, vocab):
    backend = PullingBackend(vocab, block=True)
    stage = EmbedStage(indexed, settings, backend=backend)
    runner = PipelineRunner(indexed, settings, lambda _n: stage)
    runner.poll_interval = 0.05
    thread, out = run_in_thread(runner, ["embed"])
    assert backend.pulling.wait(10)
    progress = control.pull_progress(indexed)  # for the UI, also of other processes
    assert progress["model"] == EMBED and progress["stage"] == "embed" and progress["percent"] == 40
    control.request_state(indexed, control.PAUSED)  # a write while the download runs
    wait_for(lambda: not backend.pulling.is_set())
    time.sleep(0.2)
    assert backend.pulls == [EMBED] and thread.is_alive()  # waits while paused
    control.request_state(indexed, control.RUNNING)
    thread.join(10)
    assert out["summary"].status == "completed"
    assert backend.pulls == [EMBED, EMBED]  # set up again: Ollama continues the download
    assert control.pull_progress(indexed) is None


def test_stop_during_the_download_ends_the_run_cleanly(indexed, settings, vocab):
    backend = PullingBackend(vocab, block=True)

    class Stage(EmbedStage):
        torn_down = False

        def teardown(self):
            self.torn_down = True

    stage = Stage(indexed, settings, backend=backend)
    runner = PipelineRunner(indexed, settings, lambda _n: stage)
    thread, out = run_in_thread(runner, ["embed"])
    assert backend.pulling.wait(10)
    control.request_state(indexed, control.STOPPED)
    thread.join(10)
    assert out["summary"].status == "stopped" and not stage.torn_down
    with indexed.session() as s:
        rows = list(s.scalars(select(StageStatus).where(StageStatus.stage == "embed")))
    assert {r.status for r in rows} == {Status.PENDING} and {r.attempts for r in rows} == {0}
    assert control.pull_progress(indexed) is None


# --------------------------------------------------------------------------- web UI
@pytest.fixture
def web(settings):
    """``web(ollama_url)`` -> a TestClient of an app that uses Ollama at that address."""
    clients = []

    def make(url: str, **overrides):
        ollama_settings(settings, url)
        for key, value in overrides.items():
            setattr(settings, key, value)
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def models_job(client):
    return client.app.state.ctx.downloads


def test_download_button_pulls_the_missing_models_in_a_background_job(web):
    with FakeOllama(installed=[VLM], delay=0.05) as ollama:
        client = web(ollama.url)
        page = client.get("/pipeline").text
        assert "Download missing models" not in page  # only one is missing
        assert 'name="key" value="embed"' in page and "ollama pull nomic-embed-text:latest" in page
        r = client.post("/models/download", data={"key": "embed"}, follow_redirects=False)
        assert r.status_code == 303 and "msg=" in r.headers["location"]
        wait_for(lambda: models_job(client).current.progress.get("current"))
        # one download job at a time
        r = client.post("/models/download", data={}, headers={"Referer": "http://testserver/"})
        assert "The AI models are being downloaded already." in r.text
        page = client.get("/partials/models").text
        assert (
            "downloading" in page and 'hx-trigger="every 2s' in page and "Cancel download" in page
        )
        activity = client.get("/partials/activity").text
        assert "Downloading AI models" in activity and EMBED in activity
        wait_for(lambda: not models_job(client).busy)
        job = models_job(client).current
        assert job.status == "done" and str(job.message) == f"AI models ready: {EMBED}"
        assert ollama.pulls == [EMBED]
        page = client.get("/partials/models").text
        assert "every 2s" not in page and page.count("badge ok") == 2
        assert "AI models ready" in client.get("/partials/status").text  # in the job list


def test_download_job_can_be_cancelled(web):
    with FakeOllama(delay=0.1) as ollama:
        client = web(ollama.url)
        assert "No download is running." in client.post("/models/cancel").text
        client.post("/models/download")
        wait_for(lambda: models_job(client).current.progress.get("current"))
        client.post("/models/cancel")
        wait_for(lambda: not models_job(client).busy)
        assert "Download cancelled." in str(models_job(client).current.message)
        assert ollama.finished == [] and ollama.pulls == [VLM]


def test_a_failed_download_is_shown(web, lang):
    with FakeOllama(installed=[VLM], errors={EMBED: "no space left on device"}) as ollama:
        client = web(ollama.url)
        client.post("/models/download")
        wait_for(lambda: not models_job(client).busy)
        assert models_job(client).current.status == "failed"
        assert "Not enough disk space" in client.get("/partials/models").text
        nudge = client.get("/partials/models-nudge", headers=DE).text
        assert "Nicht genug Speicherplatz" in nudge and "Jetzt herunterladen" in nudge


def test_model_endpoints_are_post_only_same_origin_and_take_no_model_names(web):
    with FakeOllama(installed=[VLM]) as ollama:
        client = web(ollama.url)
        assert client.get("/models/download").status_code == 405
        assert client.get("/models/cancel").status_code == 405
        evil = {"Origin": "http://evil.example"}
        assert client.post("/models/download", headers=evil).status_code == 403
        assert client.post("/models/cancel", headers=evil).status_code == 403
        r = client.post("/models/download", data={"key": "evil:latest"})
        assert "Unknown model." in r.text and not models_job(client).busy
        # a model name in the form is ignored: only the configured models are downloaded
        client.post("/models/download", data={"model": "evil:latest", "name": "evil:latest"})
        wait_for(lambda: not models_job(client).busy)
        assert ollama.pulls == [EMBED]


def test_fake_backend_needs_no_models(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/partials/models-nudge").text == ""
        assert "no models needed" in client.get("/partials/health").text
        r = client.post("/models/download")
        assert "This configuration needs no models from Ollama." in r.text


@pytest.mark.parametrize("headers", [{}, DE], ids=["en", "de"])
def test_environment_card_and_home_nudge_in_every_state(web, headers):
    de = bool(headers)
    # 1. everything is there
    with FakeOllama(installed=[VLM, EMBED]) as ollama:
        client = web(ollama.url)
        health = client.get("/partials/health", headers=headers).text
        assert health.count("badge ok") >= 3
        assert "/models/download" not in health and "ollama pull" not in health
        assert client.get("/partials/models-nudge", headers=headers).text == ""
        assert 'hx-get="/partials/models-nudge"' in client.get("/", headers=headers).text
        done = client.get("/partials/models-nudge?polling=1", headers=headers).text
        assert ("Die KI-Modelle sind bereit" if de else "The AI models are ready") in done
    # 2. models missing
    with FakeOllama() as ollama:
        client = web(ollama.url)
        health = client.get("/partials/health", headers=headers).text
        assert health.count('action="/models/download"') == 3  # two models + "all missing"
        assert ("Fehlende Modelle herunterladen" if de else "Download missing models") in health
        assert (
            "ollama pull qwen2.5vl:7b" in health and "ollama pull nomic-embed-text:latest" in health
        )
        assert ("fehlt" if de else "missing") in health
        nudge = client.get("/partials/models-nudge", headers=headers).text
        assert ("Jetzt herunterladen" if de else "Download now") in nudge
        assert ("noch 2 KI-Modelle" if de else "still needs 2 AI models") in nudge
        assert "every 2s" not in nudge
    # 3. Ollama is not reachable
    client = web(f"http://127.0.0.1:{free_port()}")
    health = client.get("/partials/health", headers=headers).text
    assert ("nicht erreichbar" if de else "not reachable") in health
    assert 'href="https://ollama.com/download"' in health and "/models/download" not in health
    nudge = client.get("/partials/models-nudge", headers=headers).text
    assert ("Ollama für Eigenschaften und Suche installieren" if de else "Install Ollama") in nudge
    assert 'href="https://ollama.com/download"' in nudge and "/models/download" not in nudge


def test_remote_mode_asks_to_start_the_servers_ollama(web, settings, tmp_path):
    settings.remote.upload_dir = tmp_path / "uploads"
    client = web(f"http://127.0.0.1:{free_port()}", mode="remote")
    for url in ("/partials/models-nudge", "/partials/health"):
        page = client.get(url).text
        assert "Start the Ollama service on the server" in page and "ollama.com" not in page


def test_nudge_follows_a_download_and_stays_away_when_search_is_off(web, settings):
    with FakeOllama(installed=[VLM]) as ollama:
        client = web(ollama.url)
        ctx = client.app.state.ctx
        job = Job(name="models", message="x")
        job.progress = {
            "current": {
                "model": EMBED,
                "phase": "download",
                "completed": 5,
                "total": 10,
                "percent": 50,
            }
        }
        ctx.downloads.current = job
        nudge = client.get("/partials/models-nudge").text
        assert "Downloading the AI models…" in nudge and "every 2s" in nudge and "(50 %)" in nudge
        activity = client.get("/partials/activity", headers=DE).text
        assert "KI-Modelle werden heruntergeladen" in activity and 'style="width: 50%"' in activity
        health = client.get("/partials/health").text
        assert "downloading 50 %" in health and "Cancel download" in health
        job.status = "done"
        settings.pipeline.stages = ["faces", "ages"]
        assert client.get("/partials/models-nudge").text == ""


def test_pipeline_download_shows_in_status_and_activity(web, settings):
    with FakeOllama(installed=[VLM]) as ollama:
        client = web(ollama.url)
        db = client.app.state.ctx.db
        assert "every 2s" not in client.get("/partials/models").text
        control.try_acquire(db, ["embed"], 4, None, 30)
        with db.write() as s:
            s.get(Control, 1).active_stage = "embed"
        # a run with a model missing may download it: the Environment card follows
        page = client.get("/partials/models").text
        assert "every 2s" in page and "missing" in page
        control.set_progress(
            db,
            control.PULL_PROGRESS,
            {
                "stage": "embed",
                "model": EMBED,
                "phase": "download",
                "completed": 1,
                "total": 4,
                "percent": 25,
            },
        )
        status = client.get("/partials/status").text
        assert "Downloading the model for Search index first" in status and "(25 %)" in status
        activity = client.get("/partials/activity").text
        assert "Search index" in activity and EMBED in activity and 'style="width: 25%"' in activity
        assert "downloading 25 %" in client.get("/partials/models").text
        german = client.get("/partials/status", headers=DE).text
        assert "Zuerst wird das Modell für Suchindex heruntergeladen" in german


# --------------------------------------------------------------------------- CLI
@pytest.fixture
def cli_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)

    def write(url: str, backend: str = "ollama"):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f"[paths]\ndata_dir = '{(tmp_path / 'data').as_posix()}'\n"
            f"[vlm]\nbackend = '{backend}'\nbase_url = '{url}'\n"
        )
        return str(cfg)

    return write


def test_cli_models_lists_and_pulls(cli_config):
    runner = CliRunner()
    with FakeOllama(installed=[VLM]) as ollama:
        cfg = cli_config(ollama.url)
        result = runner.invoke(cli_app, ["-c", cfg, "models"])
        assert result.exit_code == 1, result.output
        assert "Ollama 0.12.0" in result.output
        assert "qwen2.5vl:7b             installed  properties, search queries" in result.output
        assert "nomic-embed-text:latest  missing    search index" in result.output
        assert "vision-tagger models pull" in result.output
        result = runner.invoke(cli_app, ["-c", cfg, "models", "pull"])
        assert result.exit_code == 0, result.output
        assert "downloading nomic-embed-text:latest (1/1)" in result.output
        assert "nomic-embed-text:latest: 1.5 GB of 3.0 GB (50 %)" in result.output
        assert f"downloaded: {EMBED}" in result.output and ollama.pulls == [EMBED]
        result = runner.invoke(cli_app, ["-c", cfg, "models"])
        assert result.exit_code == 0 and "all models are there" in result.output
        result = runner.invoke(cli_app, ["-c", cfg, "models", "pull"])
        assert result.exit_code == 0 and "all models are there" in result.output


def test_cli_models_errors(cli_config):
    runner = CliRunner()
    cfg = cli_config(f"http://127.0.0.1:{free_port()}")
    for args in (["models"], ["models", "pull"]):
        result = runner.invoke(cli_app, ["-c", cfg, *args])
        assert result.exit_code == 1 and "https://ollama.com/download" in result.output
    with FakeOllama(errors={VLM: "pull model manifest: file does not exist"}) as ollama:
        result = runner.invoke(cli_app, ["-c", cli_config(ollama.url), "models", "pull"])
        assert result.exit_code == 1 and "does not know the model qwen2.5vl:7b" in result.output
    result = runner.invoke(cli_app, ["-c", cli_config("http://x", backend="fake"), "models"])
    assert result.exit_code == 0 and "no Ollama models needed" in result.output
