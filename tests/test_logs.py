"""The log file (``vision_tagger.logs``), the Log page and the "Report a bug" button."""

from __future__ import annotations

import logging
import re
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from vision_tagger import __version__, bugreport, desktop, logs
from vision_tagger import config as config_mod
from vision_tagger.cli import app as cli_app
from vision_tagger.config import Settings
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.web.app import create_app

from .conftest import make_image
from .test_runner import FakeStage

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
LOCAL = {"base_url": "http://127.0.0.1:8765", "client": ("127.0.0.1", 50123)}


def record(message: str, level: int = logging.INFO, name: str = "vision_tagger.test", exc=None):
    return logging.LogRecord(name, level, __file__, 1, message, (), exc)


def handler(path: Path, max_bytes: int = 4096, backups: int = 2) -> logs.LogFile:
    h = logs.LogFile(path, max_bytes=max_bytes, backups=backups)
    h.setFormatter(logging.Formatter(logs.FORMAT, logs.DATE_FORMAT))
    return h


def all_lines(path: Path) -> list[str]:
    return [line for p in logs.files(path) for line in p.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- the file
def test_log_file_stays_below_its_size_and_drops_the_oldest(tmp_path):
    path = tmp_path / "logs" / logs.FILE_NAME
    h = handler(path, max_bytes=4096, backups=2)
    assert not path.parent.exists()  # nothing is created before something is logged
    for i in range(400):  # paths with umlauts and CJK take more bytes than characters
        h.handle(record(f"line {i:04d} " + "x" * 30 + "ü漢" * 10))
    h.close()
    assert logs.files(path) == [h.backup(2), h.backup(1), path]
    assert not h.backup(3).exists()
    for part in logs.files(path):
        assert part.stat().st_size <= 4096
    lines = all_lines(path)
    numbers = [int(re.search(r"line (\d{4})", line)[1]) for line in lines]
    assert numbers == sorted(numbers) and numbers[-1] == 399  # in order, the newest last
    assert numbers[0] > 0  # the oldest are gone
    assert sum(p.stat().st_size for p in logs.files(path)) <= 3 * 4096


def test_without_backups_the_file_is_just_cut(tmp_path):
    path = tmp_path / logs.FILE_NAME
    h = handler(path, max_bytes=4096, backups=0)
    for i in range(200):
        h.handle(record(f"line {i} " + "y" * 60))
    h.close()
    assert logs.files(path) == [path] and path.stat().st_size <= 4096
    assert "line 199 " in path.read_text(encoding="utf-8")


def test_two_processes_share_the_file(tmp_path):
    """Two writers of one file (the app and a command line run): everything is kept in order
    and the size limit holds, although each one rotates the file under the other."""
    path = tmp_path / logs.FILE_NAME
    first, second = handler(path, 4096, backups=20), handler(path, 4096, backups=20)
    for i in range(300):
        (first if i % 2 else second).handle(record(f"entry {i:04d} " + "z" * 40))
    first.close()
    second.close()
    lines = all_lines(path)
    numbers = [int(re.search(r"entry (\d{4})", line)[1]) for line in lines]
    assert numbers == list(range(300))
    for part in logs.files(path):
        assert part.stat().st_size <= 4096


def test_a_file_that_cannot_be_rotated_is_written_anyway(tmp_path, monkeypatch):
    """Windows: a viewer or another process holds the file. The records still go in; rotating
    is tried again later, and at twice the limit the old text is dropped without a copy."""
    path = tmp_path / logs.FILE_NAME
    calls = []

    def locked(src, dst):
        calls.append(dst)
        raise PermissionError(13, "used by another process")

    real_copyfile = shutil.copyfile  # logs.shutil is this very module
    monkeypatch.setattr(logs.shutil, "copyfile", locked)
    now = [0.0]
    h = handler(path, max_bytes=4096, backups=2)
    h.clock = lambda: now[0]
    sizes = []
    for i in range(150):
        h.handle(record(f"line {i:04d} " + "x" * 60))
        sizes.append(path.stat().st_size)
    assert 0 < len(calls) < 10  # once per size limit reached, not with each of 150 records
    assert max(sizes) > 4096  # grew beyond the limit meanwhile...
    assert max(sizes) <= 2 * 4096 + 200  # ...but never beyond twice the limit
    assert "line 0149 " in path.read_text(encoding="utf-8")
    monkeypatch.setattr(logs.shutil, "copyfile", real_copyfile)  # the viewer is closed
    now[0] += h.retry_seconds + 1
    for i in range(150, 200):
        h.handle(record(f"line {i:04d} " + "x" * 60))
    h.close()
    assert h.backup(1).is_file() and path.stat().st_size <= 4096


def test_a_log_that_cannot_be_written_never_breaks_the_app(tmp_path, capsys):
    blocker = tmp_path / "data"
    blocker.write_text("a file where the folder should be")
    path = blocker / "logs" / logs.FILE_NAME
    now = [0.0]
    h = handler(path)
    h.clock = lambda: now[0]
    h.handle(record("lost"))
    h.handle(record("lost too"))
    assert "Logging error" not in capsys.readouterr().err  # dropped quietly
    blocker.unlink()
    h.handle(record("still waiting"))  # tried again only after a while
    assert not path.exists()
    now[0] += h.retry_seconds + 1
    h.handle(record("back"))
    h.close()
    assert path.read_text(encoding="utf-8").count("\n") == 1 and "back" in path.read_text()


# --------------------------------------------------------------------------- setup
def file_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, logs.LogFile)]


def test_setup_writes_who_started_and_keeps_noise_out(settings):
    path = logs.setup(settings, "test run")
    assert path == settings.data_dir / "logs" / "vision-tagger.log"
    logging.getLogger("vision_tagger.x").info("something happened")
    logging.getLogger("vision_tagger.x").debug("details nobody asked for")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /media/thumb/1 HTTP/1.1" 200')
    logging.getLogger("uvicorn.error").error("Exception in ASGI application")
    logging.getLogger("alembic.runtime.migration").info("Context impl SQLiteImpl.")
    logging.getLogger("alembic.runtime.migration").info("Running upgrade 0007 -> 0008")
    logging.getLogger("httpx").info("HTTP Request: POST http://localhost:11434/api/chat")
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert f"Mira {__version__} started (test run)" in first and "local mode" in first
    assert "something happened" in text and "Exception in ASGI application" in text
    assert "Running upgrade 0007 -> 0008" in text
    for left_out in ("details nobody", "GET /media", "Context impl", "HTTP Request"):
        assert left_out not in text
    # a second library (migrate --desktop): one log file at a time
    other = Settings(paths={"data_dir": settings.data_dir.parent / "other"})
    logs.setup(other, "again")
    assert len(file_handlers()) == 1 and logs.file_handler().path == logs.log_path(other)
    logs.reset()
    assert not file_handlers()


def test_level_and_verbose(settings):
    settings.logging.level = "warning"
    path = logs.setup(settings, "quiet")
    logging.getLogger("vision_tagger.x").info("not this")
    logging.getLogger("vision_tagger.x").warning("but this")
    text = path.read_text(encoding="utf-8")
    assert "not this" not in text and "but this" in text
    assert "started (quiet)" not in text  # an INFO line as well
    logs.setup(settings, "verbose", verbose=True)
    logging.getLogger("vision_tagger.x").debug("every detail")
    assert "every detail" in path.read_text(encoding="utf-8")


def test_uncaught_exceptions_are_logged(settings, monkeypatch):
    printed = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: printed.append(a))
    monkeypatch.setattr(threading, "excepthook", lambda args: printed.append(args))
    path = logs.setup(settings, "test")
    try:
        raise ValueError("in the main thread")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    try:
        raise OSError("in a thread")
    except OSError as exc:
        args = SimpleNamespace(
            exc_type=OSError,
            exc_value=exc,
            exc_traceback=exc.__traceback__,
            thread=SimpleNamespace(name="job-index"),
        )
        threading.excepthook(args)
    assert len(printed) == 2  # the console still shows them as before
    text = path.read_text(encoding="utf-8")
    assert "CRITICAL" in text and "uncaught exception" in text and "in the main thread" in text
    assert "uncaught exception in thread job-index" in text and "OSError: in a thread" in text
    assert text.count("Traceback (most recent call last)") == 2
    logs.setup(settings, "twice")  # hooks are installed once
    assert sys.excepthook.previous is not sys.excepthook  # type: ignore[attr-defined]
    assert not getattr(sys.excepthook.previous, "_vision_tagger_log", False)  # type: ignore[attr-defined]


def test_warnings_go_into_the_log(settings):
    import warnings

    path = logs.setup(settings, "test")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("a library is unhappy", UserWarning, stacklevel=1)
    assert "a library is unhappy" in path.read_text(encoding="utf-8")


def test_system_names_no_paths(settings):
    text = logs.system(settings)
    assert "Python" in text and text.endswith("local mode")
    assert str(settings.data_dir) not in text and str(Path.home()) not in text


# --------------------------------------------------------------------------- reading
OLD_FORMAT = """\
2026-11-20 09:00:00,001 INFO    vision_tagger.desktop: serving on http://127.0.0.1:50000/
2026-11-20 09:00:01,002 ERROR   vision_tagger.web.context: job index failed
Traceback (most recent call last):
  File "x.py", line 1, in <module>
OSError: disk full
"""


def test_reading_entries(tmp_path):
    path = tmp_path / logs.FILE_NAME
    h = handler(path, max_bytes=100_000)
    h.handle(record("first"))
    try:
        raise KeyError("faces")
    except KeyError:
        h.handle(record("job recluster failed", logging.ERROR, exc=sys.exc_info()))
    h.handle(record("careful", logging.WARNING))
    h.handle(record("debugging", logging.DEBUG))
    h.close()
    view = logs.read(path, min_level=logging.DEBUG)
    assert [e.message for e in view.entries] == [
        "debugging",
        "careful",
        "job recluster failed",
        "first",
    ]
    error = view.entries[2]
    assert (
        error.level == "ERROR"
        and error.severity == "error"
        and error.logger == "vision_tagger.test"
    )
    assert "KeyError: 'faces'" in error.detail_text and "MainThread" in error.where
    assert view.counts == {"info": 1, "error": 1, "warn": 1, "debug": 1} and view.read == 4
    assert [e.message for e in logs.read(path, min_level=logging.WARNING).entries] == [
        "careful",
        "job recluster failed",
    ]
    limited = logs.read(path, min_level=logging.DEBUG, limit=2)
    assert [e.message for e in limited.entries] == ["debugging", "careful"] and limited.more


def test_reading_the_old_format_and_a_cut_tail(tmp_path):
    entries = logs.parse(OLD_FORMAT)
    assert [(e.level, e.logger) for e in entries] == [
        ("INFO", "vision_tagger.desktop"),
        ("ERROR", "vision_tagger.web.context"),
    ]
    assert entries[1].details[-1] == "OSError: disk full"
    path = tmp_path / logs.FILE_NAME
    path.write_text("x" * 50 + "\n" + OLD_FORMAT, encoding="utf-8")
    view = logs.read(path, max_bytes=len(OLD_FORMAT.encode()) + 10)  # starts within the x line
    assert [e.message for e in view.entries][-1].startswith("serving on")


def test_files_and_download_order(tmp_path):
    path = tmp_path / logs.FILE_NAME
    # bytes: write_text would write "\r\n" on Windows, and the download is the files as they are
    for name, data in (("vision-tagger.log.2", b"a\n"), ("vision-tagger.log.1", b"b\n")):
        (tmp_path / name).write_bytes(data)
    (tmp_path / "vision-tagger.log.bak").write_bytes(b"not a part")
    path.write_bytes(b"c\n")
    assert [p.name for p in logs.files(path)] == [
        "vision-tagger.log.2",
        "vision-tagger.log.1",
        "vision-tagger.log",
    ]
    assert b"".join(logs.iter_bytes(path)) == b"a\nb\nc\n"


# --------------------------------------------------------------------------- bug report
def test_bug_report_url_fills_in_the_form(settings):
    url = bugreport.issue_url(settings)
    parts = urlsplit(url)
    assert parts.netloc == "github.com" and parts.path == "/w1nte/mira/issues/new"
    query = parse_qs(parts.query)
    assert query["template"] == ["bug_report.yml"]
    assert query["version"] == [__version__]
    assert query["environment"] == [logs.system(settings)]
    assert str(settings.data_dir) not in url


def test_the_issue_form_has_the_fields_mira_fills_in():
    form = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert TEMPLATE.name == bugreport.TEMPLATE
    assert form["name"] and form["description"] and "bug" in form["labels"]
    ids = [item["id"] for item in form["body"] if "id" in item]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids)
    for field in bugreport.FIELDS:
        assert field in ids, field
    types = {item["type"] for item in form["body"]}
    assert types <= {"markdown", "textarea", "input", "dropdown", "checkboxes"}
    assert any(item.get("id") == "log" for item in form["body"])


# --------------------------------------------------------------------------- the web UI
@pytest.fixture
def launched(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")  # a graphical session: the file buttons may appear
    return []


@pytest.fixture
def local(settings, launched):
    with TestClient(create_app(settings), **LOCAL) as client:
        client.app.state.ctx.file_manager.launcher = launched.append
        client.app.state.ctx.file_manager.platform = "linux"
        yield client


def test_log_page_without_a_log(local):
    page = local.get("/logs")
    assert page.status_code == 200
    assert "There is no log yet." in page.text
    assert 'href="/logs/download"' not in page.text
    assert local.get("/logs/download").status_code == 404


def test_log_page_shows_the_newest_entries(local, settings, launched):
    path = logs.setup(settings, "web test")
    log = logging.getLogger("vision_tagger.pipeline.runner")
    log.warning("vlm failed for image 7 (<b>beach</b>.jpg): ConnectionError")
    try:
        raise RuntimeError("model crashed")
    except RuntimeError:
        logging.getLogger("vision_tagger.web.context").exception("job recluster failed")
    page = local.get("/logs").text
    assert str(path) in page
    assert "vlm failed for image 7 (&lt;b&gt;beach&lt;/b&gt;.jpg)" in page  # escaped
    assert "RuntimeError: model crashed" in page and "<details open>" in page
    assert page.index("job recluster failed") < page.index("vlm failed")  # newest first
    assert "started (web test)" in page
    errors = local.get("/logs?show=errors").text
    assert "job recluster failed" in errors and "vlm failed" not in errors
    assert "started (web test)" not in local.get("/logs?show=warnings").text
    assert local.get("/logs?show=nonsense").status_code == 200
    # opening the file and showing it in its folder (this computer, local mode)
    assert re.search(r'name="kind" value="log"', page) and re.search(r'value="logs"', page)
    for kind in ("log", "logs"):
        r = local.post("/reveal", data={"kind": kind}, headers={"HX-Request": "true"})
        assert r.status_code == 204, r.text
    assert [a.command for a in launched] == [
        ["xdg-open", str(path)],
        ["xdg-open", str(path.parent)],
    ]
    download = local.get("/logs/download")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/plain")
    disposition = download.headers["content-disposition"]
    assert re.fullmatch(r'attachment; filename="vision-tagger-\d{4}-\d\d-\d\d\.log"', disposition)
    assert "job recluster failed" in download.text and "RuntimeError" in download.text


def test_opening_a_missing_log_file(local):
    r = local.post(
        "/reveal",
        data={"kind": "log"},
        headers={"Referer": "http://127.0.0.1:8765/logs"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and "err=The+file+is+not+there" in r.headers["location"]


def test_report_a_bug_and_the_log_on_every_page(local):
    page = local.get("/").text
    assert 'href="/logs"' in page
    links = re.findall(r'href="(https://github.com/w1nte/mira/issues/new\?[^"]+)"', page)
    assert links and "template=bug_report.yml" in links[0]
    assert 'target="_blank" rel="noopener"' in page


def test_remote_mode_shows_no_server_paths(settings, tmp_path, launched):
    settings.mode = "remote"
    settings.remote.upload_dir = tmp_path / "uploads"
    path = logs.setup(settings, "server")
    with TestClient(create_app(settings), **LOCAL) as client:
        page = client.get("/logs").text
        assert "started (server)" in page
        assert str(path) not in page and 'value="log"' not in page
        assert client.post("/reveal", data={"kind": "log"}).status_code == 403
        assert client.get("/logs/download").status_code == 200


def test_errors_shown_to_the_user_are_logged(local, settings):
    path = logs.setup(settings, "web test")
    r = local.post("/import/roots", data={"path": str(settings.data_dir / "nope")})
    assert r.status_code in (200, 303)
    assert "POST /import/roots: Not a folder" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the processes
def test_command_line_logs_to_the_library(tmp_path, monkeypatch, photos):
    make_image(photos / "a.jpg", seed=1)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"[paths]\nimage_roots = ['{photos.as_posix()}']\n"
        f"data_dir = '{(tmp_path / 'cli-data').as_posix()}'\n[vlm]\nbackend = 'fake'\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    path = tmp_path / "cli-data" / "logs" / logs.FILE_NAME
    result = CliRunner().invoke(cli_app, ["-c", str(cfg), "index", "--help"])
    assert result.exit_code == 0 and not path.exists()  # help writes no log
    result = CliRunner().invoke(cli_app, ["-c", str(cfg), "index"])
    assert result.exit_code == 0, result.output
    text = path.read_text(encoding="utf-8")
    assert "started (command line: index)" in text
    assert "scan done: 1 new" in text
    result = CliRunner().invoke(cli_app, ["-c", str(cfg), "migrate"])
    assert result.exit_code == 0, result.output
    assert "vision_tagger.migrate: schema:" in path.read_text(encoding="utf-8")


def test_desktop_app_logs_to_the_library(settings):
    path = desktop.setup_logging(settings)
    assert path == settings.data_dir / "logs" / logs.FILE_NAME
    assert "started (desktop app)" in path.read_text(encoding="utf-8")


def test_pipeline_failures_are_explained_once_per_kind(db, settings, photos):
    for i in range(4):
        make_image(photos / f"img{i}.jpg", seed=i)
    Indexer(db, settings, stages=["faces"]).run()
    path = logs.setup(settings, "test")
    stage = FakeStage(db, settings, fail_ids={1, 2, 3})
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["faces"])
    assert summary.failed == {"faces": 3}
    text = path.read_text(encoding="utf-8")
    assert re.search(r"run \d+ started: faces, batch size 4", text)
    failures = [line for line in text.splitlines() if "faces failed for image" in line]
    assert len(failures) == 3
    # the path of the photo, not only its id
    assert sum(str(photos / f"img{i}.jpg") in text for i in range(4)) == 3
    assert text.count("Traceback (most recent call last)") == 1  # the first one only
    assert re.search(r"run \d+ completed: faces 4 \(3 failed\)", text)
