from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import types

import httpx
import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from vision_tagger import config as config_mod
from vision_tagger import desktop as D
from vision_tagger.cli import app as cli_app
from vision_tagger.db.migrate import current_revision, head_revision, migration_engine
from vision_tagger.db.models import Control, Run, StageStatus, Status
from vision_tagger.db.session import Database
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline import control

from .conftest import make_image
from .test_migrations import build_at, url_of


def test_user_config_is_written_once_with_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no ./config.toml here
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.delenv("VT_CONFIG", raising=False)
    app_dir = tmp_path / "App Data" / "vision-tagger"
    monkeypatch.setattr(D, "user_data_dir", lambda: app_dir)
    settings = D.load_desktop_settings()
    assert settings.data_dir == (app_dir / "data").resolve()
    assert settings.paths.references_dir == app_dir / "references"
    assert settings.paths.image_roots == [] and settings.mode == "local"
    cfg = app_dir / "config.toml"
    text = cfg.read_text(encoding="utf-8")
    assert "# database, thumbnails and face crops" in text  # comments are kept
    cfg.write_text(text.replace('model = "qwen2.5vl:7b"', 'model = "qwen2.5vl:3b"'))
    D.ensure_user_config(app_dir)  # never overwrites the user's changes
    assert config_mod.load_settings(cfg).vlm.model == "qwen2.5vl:3b"


def test_user_data_dir_per_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert D.user_data_dir() == tmp_path / "Local" / "vision-tagger"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    assert D.user_data_dir() == tmp_path / "share" / "vision-tagger"


class FakeWindow:
    """Stands in for the native window: looks at the app, then 'closes' the window."""

    def __init__(self, check):
        self.check = check
        self.extra: list[str] = []

    def show(self, url, idle_seconds):
        self.check(url, idle_seconds)

    def another(self, url):
        self.extra.append(url)


def test_window_serves_the_app_and_closing_stops_everything(settings):
    seen = {}

    def check(url, idle_seconds):
        assert httpx.get(url + "healthz").json() == {"ok": True}
        assert "· Mira</title>" in httpx.get(url).text  # the display name
        assert idle_seconds() < 5
        # starting the app again opens another window instead of a second server
        assert D.InstanceLock(settings.data_dir / "desktop.json").notify_running()
        deadline = time.time() + 5
        while not fake.extra:
            assert time.time() < deadline
            time.sleep(0.02)
        seen["url"] = url

    fake = FakeWindow(check)
    assert D.run(settings, windows=D.Windows([lambda: fake]), port=0) == 0
    assert fake.extra == [seen["url"]]
    with pytest.raises(httpx.TransportError):
        httpx.get(seen["url"] + "healthz", timeout=2)  # the server is gone
    assert not (settings.data_dir / "desktop.json").exists()
    assert not D.InstanceLock(settings.data_dir / "desktop.json").notify_running()


def test_outdated_database_is_upgraded_on_start(settings):
    settings.ensure_dirs()
    build_at(settings.db_path, "0003")
    D.run(settings, windows=D.Windows([lambda: FakeWindow(lambda url, idle: None)]), port=0)
    engine = migration_engine(url_of(settings.db_path))
    assert current_revision(engine) == head_revision()
    engine.dispose()
    assert list(settings.data_dir.glob("*.bak-0003-*"))


def test_falls_back_to_the_next_kind_of_window():
    def unavailable():
        raise D.WindowUnavailable("no GUI toolkit")

    fake = FakeWindow(lambda url, idle: None)
    windows = D.Windows([unavailable, lambda: fake])
    windows.show("http://127.0.0.1:1/", lambda: 0.0)
    assert windows.active is fake
    with pytest.raises(RuntimeError):
        D.Windows([unavailable]).show("http://127.0.0.1:1/", lambda: 0.0)


def test_browser_app_window_waits_for_its_process(tmp_path, monkeypatch):
    script = tmp_path / "browser.py"
    script.write_text("import sys, time\ntime.sleep(float(sys.argv[1]))\n")
    monkeypatch.setattr(D, "QUICK_EXIT_SECONDS", 0.5)
    idle_calls = []

    def idle():
        idle_calls.append(1)
        return 1e9

    # a window that stays open: wait for the process, nothing else
    w = D.BrowserAppWindow([sys.executable, str(script), "1.0"], tmp_path / "profile")
    started = time.monotonic()
    w.show("http://127.0.0.1:1/", idle)
    assert time.monotonic() - started >= 1.0 and not idle_calls
    # handed to an already running browser: wait until nobody uses the app any more
    w = D.BrowserAppWindow([sys.executable, str(script), "0"], tmp_path / "profile")
    w.show("http://127.0.0.1:1/", idle)
    assert idle_calls


def test_find_browser(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    edge = tmp_path / "x86" / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    edge.parent.mkdir(parents=True)
    edge.write_bytes(b"")
    monkeypatch.setenv("PROGRAMFILES(X86)", str(tmp_path / "x86"))
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "none"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "none"))
    assert D.find_browser() == str(edge)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(D.shutil, "which", lambda name: "/usr/bin/chromium" * (name == "chromium"))
    assert D.find_browser() == "/usr/bin/chromium"


def test_instance_lock_ignores_strangers(tmp_path):
    lock = D.InstanceLock(tmp_path / "desktop.json")
    opened = threading.Event()
    lock.listen("http://127.0.0.1:1/", opened.set)
    other = D.InstanceLock(tmp_path / "other.json")
    (tmp_path / "other.json").write_text(
        (tmp_path / "desktop.json").read_text().replace(lock.token, "0" * 32)
    )
    assert not other.notify_running()  # wrong token
    assert not opened.is_set()
    assert D.InstanceLock(tmp_path / "desktop.json").notify_running()
    assert opened.wait(5)
    lock.release()
    assert not (tmp_path / "desktop.json").exists()


def test_self_test_passes(capsys):
    assert D.self_test() == 0
    assert "self-test passed" in capsys.readouterr().out


def test_frozen_linux_app_skips_pywebview(settings, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert len(D.default_windows(settings).backends) == 3  # from source: pywebview first
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert len(D.default_windows(settings).backends) == 2  # no GTK/Qt in the bundle
    monkeypatch.setattr(sys, "platform", "darwin")
    assert len(D.default_windows(settings).backends) == 3


def test_frozen_build_gives_child_programs_the_original_library_path(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert D.child_environment() is None  # not frozen: inherit everything
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/app/_internal:/opt/app/_internal/cv2/../lib64")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/local/lib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/opt/app/_internal")
    for _ in range(2):  # the same every time
        env = D.child_environment()
        assert env["LD_LIBRARY_PATH"] == "/usr/local/lib"
        assert "LD_LIBRARY_PATH_ORIG" not in env and "DYLD_LIBRARY_PATH" not in env
    assert os.environ["LD_LIBRARY_PATH"].startswith("/opt/app")  # the app's own is untouched
    with D._child_environment_applied():
        assert os.environ["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert os.environ["LD_LIBRARY_PATH"].startswith("/opt/app")


# --------------------------------------------------------------------------- quit (installers)
class WindowUntilQuit:
    """Stands in for a window that stays open until the app is asked to quit."""

    def __init__(self):
        self.shown = threading.Event()
        self.closed = threading.Event()

    def show(self, url, idle_seconds):
        self.shown.set()
        assert self.closed.wait(60), "nobody asked the app to quit"

    def another(self, url):
        pass

    def quit(self):
        self.closed.set()


def start_app(settings, window):
    thread = threading.Thread(
        target=D.run, args=(settings,), kwargs={"windows": D.Windows([lambda: window]), "port": 0}
    )
    thread.start()
    assert window.shown.wait(60)
    return thread


def test_quit_closes_the_running_app_and_waits_until_it_is_gone(settings):
    thread = start_app(settings, window := WindowUntilQuit())
    url = json.loads((settings.data_dir / "desktop.json").read_text())["url"]
    assert D.quit_running(settings.data_dir, timeout=60) == D.QUIT
    thread.join(10)  # quit_running returned after the shutdown, only `return 0` is left
    assert not thread.is_alive() and window.closed.is_set()
    with pytest.raises(httpx.TransportError):
        httpx.get(url + "healthz", timeout=2)
    assert not (settings.data_dir / "desktop.json").exists()
    assert D.quit_running(settings.data_dir) == D.NOT_RUNNING


def _vlm_done(db: Database) -> int:
    with db.session() as s:
        return s.scalar(
            select(func.count())
            .select_from(StageStatus)
            .where(StageStatus.stage == "vlm", StageStatus.status == Status.DONE)
        )


def test_quit_pauses_a_running_pipeline(settings, photos, monkeypatch):
    from vision_tagger.vlm.backend import FakeBackend

    analyze = FakeBackend.analyze

    def slow(self, *args, **kwargs):
        time.sleep(0.2)
        return analyze(self, *args, **kwargs)

    monkeypatch.setattr(FakeBackend, "analyze", slow)
    for i in range(12):
        make_image(photos / f"p{i}.jpg", seed=i)
    db = Database(settings.db_path)
    db.create_all()
    Indexer(db, settings, stages=["vlm"]).run()
    with db.write() as s:  # paused when the app last closed: it continues on start
        s.add(Run(stages="vlm", batch_size=2, limit=None, status=control.PAUSED))
        s.get(Control, 1).desired_state = control.PAUSED
    thread = start_app(settings, WindowUntilQuit())
    try:
        deadline = time.time() + 60
        while _vlm_done(db) < 1:
            assert time.time() < deadline, "the pipeline did not run"
            time.sleep(0.05)
        assert D.quit_running(settings.data_dir, timeout=60) == D.QUIT
    finally:
        thread.join(30)
    assert 1 <= _vlm_done(db) < 12  # interrupted, nothing lost
    assert control.desired_state(db) == control.PAUSED  # continues on the next start
    assert control.latest_runs(db, limit=1)[0].status == control.PAUSED
    db.dispose()


def _old_app(path):
    """What an app from before `quit` does: answers `<token> open`, drops anything else."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    token = "0123456789abcdef" * 2
    path.write_text(json.dumps({"url": "x", "control": server.getsockname()[1], "token": token}))

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(3)
                if conn.recv(256).strip() == f"{token} open".encode():
                    conn.sendall(b"ok\n")

    threading.Thread(target=serve, daemon=True).start()
    return server


def test_an_older_app_that_cannot_quit_is_reported_quickly(tmp_path):
    server = _old_app(tmp_path / "desktop.json")
    try:
        started = time.monotonic()
        assert D.quit_running(tmp_path, timeout=60) == D.UNSUPPORTED
        assert time.monotonic() - started < 10
        assert D.InstanceLock(tmp_path / "desktop.json").notify_running()  # still works
    finally:
        server.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_quit_when_no_app_runs(tmp_path):
    assert D.quit_running(tmp_path) == D.NOT_RUNNING  # no desktop.json
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    info = {"url": "x", "control": _free_port(), "token": "t"}
    for leftover in ({**info, "pid": gone.pid}, info):  # from an app that crashed
        (tmp_path / "desktop.json").write_text(json.dumps(leftover))
        assert D.quit_running(tmp_path, timeout=5) == D.NOT_RUNNING
    (tmp_path / "desktop.json").write_text("{broken")
    assert D.quit_running(tmp_path) == D.NOT_RUNNING


def test_quit_gives_up_when_the_app_does_not_exit(tmp_path):
    lock = D.InstanceLock(tmp_path / "desktop.json")
    asked = threading.Event()
    lock.listen("http://127.0.0.1:1/", lambda: None, asked.set)  # never shuts down
    try:
        assert D.quit_running(tmp_path, timeout=1.5) == D.STILL_RUNNING
        assert asked.is_set()
    finally:
        lock.release()
        lock.finished()


def test_the_instance_lock_without_quit_support_ignores_quit(tmp_path):
    lock = D.InstanceLock(tmp_path / "desktop.json")
    lock.listen("http://127.0.0.1:1/", lambda: None)  # no on_quit: like an older app
    try:
        assert D.quit_running(tmp_path, timeout=5) == D.UNSUPPORTED
    finally:
        lock.release()


def test_quit_before_the_window_opens():
    windows = D.Windows([lambda: FakeWindow(lambda url, idle: pytest.fail("window opened"))])
    windows.quit()
    windows.show("http://127.0.0.1:1/", lambda: 0.0)  # returns at once
    windows.another("http://127.0.0.1:1/")  # ignored


def test_browser_app_window_quits(tmp_path):
    script = tmp_path / "browser.py"
    script.write_text("import time\ntime.sleep(60)\n")
    window = D.BrowserAppWindow([sys.executable, str(script)], tmp_path / "profile")
    thread = threading.Thread(target=window.show, args=("http://127.0.0.1:1/", lambda: 0.0))
    thread.start()
    deadline = time.time() + 10
    while not window._procs:
        assert time.time() < deadline
        time.sleep(0.02)
    window.quit()
    thread.join(10)
    assert not thread.is_alive() and window._procs[0].poll() is not None
    window.another("http://127.0.0.1:1/")  # no new browser after quitting
    assert len(window._procs) == 1


def test_default_browser_stops_waiting_on_quit(monkeypatch):
    monkeypatch.setattr(D.webbrowser, "open", lambda url: True)
    browser = D.DefaultBrowser(idle_limit=1e9)
    thread = threading.Thread(target=browser.show, args=("http://127.0.0.1:1/", lambda: 0.0))
    thread.start()
    time.sleep(0.2)
    browser.quit()
    thread.join(10)
    assert not thread.is_alive()


def test_webview_windows_are_destroyed_on_quit(tmp_path, monkeypatch):
    class FakeNativeWindow:
        destroyed = False

        def destroy(self):
            self.destroyed = True

    started = []
    native = [FakeNativeWindow(), FakeNativeWindow()]
    fake = types.SimpleNamespace(
        windows=native,
        settings={},
        create_window=lambda *args, **kwargs: None,
        start=lambda **kwargs: started.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "webview", fake)
    window = D.WebviewWindow(tmp_path)
    window.quit()
    assert all(w.destroyed for w in native)
    window.show("http://127.0.0.1:1/", lambda: 0.0)
    assert not started  # quit before it opened: no window at all


# --------------------------------------------------------------------------- installers' commands
@pytest.fixture
def desktop_library(tmp_path, monkeypatch):
    """The desktop app's (not yet existing) library folder, and no other config."""
    monkeypatch.chdir(tmp_path)  # no ./config.toml here
    monkeypatch.setattr(config_mod, "_config_file", None)
    for name in [n for n in os.environ if n.startswith("VT_")]:
        monkeypatch.delenv(name)
    app_dir = tmp_path / "AppData" / "vision-tagger"
    monkeypatch.setattr(D, "user_data_dir", lambda: app_dir)
    return app_dir


def migrate_desktop():
    return CliRunner().invoke(cli_app, ["migrate", "--desktop"])


def test_migrate_desktop_without_a_library_creates_nothing(desktop_library, tmp_path):
    r = migrate_desktop()
    assert r.exit_code == 0, r.output
    assert "nothing to migrate" in r.output
    assert not desktop_library.exists() and not (tmp_path / "config.toml").exists()
    D.ensure_user_config(desktop_library)  # the app ran once, but made no database yet
    r = migrate_desktop()
    assert r.exit_code == 0 and "no database yet" in r.output
    assert not (desktop_library / "data").exists()


def test_migrate_desktop_upgrades_the_library_with_a_backup(desktop_library):
    D.ensure_user_config(desktop_library)
    data = desktop_library / "data"
    data.mkdir()
    build_at(data / "vision_tagger.sqlite3", "0003")
    r = migrate_desktop()
    assert r.exit_code == 0, r.output
    assert f"migrated: 0003 -> {head_revision()}" in r.output
    backups = list(data.glob("vision_tagger.sqlite3.bak-0003-*"))
    assert len(backups) == 1 and f"backup:   {backups[0]}" in r.output
    engine = migration_engine(url_of(data / "vision_tagger.sqlite3"))
    assert current_revision(engine) == head_revision()
    engine.dispose()
    assert "migrated: 0003" in (data / "logs" / "migrate.log").read_text(encoding="utf-8")
    r = migrate_desktop()
    assert r.exit_code == 0 and "already up to date" in r.output


def test_migrate_desktop_updates_an_old_desktop_config(desktop_library, monkeypatch):
    """Installers run `migrate --desktop`: it brings the app's config.toml up to date too."""
    from vision_tagger.config_migrate import CONFIG_VERSION

    monkeypatch.setattr(config_mod, "last_config_migration", None)
    config = D.ensure_user_config(desktop_library)
    old = config.read_text(encoding="utf-8").replace(
        f"config_version = {CONFIG_VERSION}", "# (no version: written by an early release)"
    )
    old = old.replace("[pipeline]\n", '[pipeline]\nstages = ["faces", "vlm"]\n', 1)
    config.write_text(old, encoding="utf-8")
    data = desktop_library / "data"
    data.mkdir()
    build_at(data / "vision_tagger.sqlite3", "0003")
    r = migrate_desktop()
    assert r.exit_code == 0, r.output
    assert f"version 1 -> {CONFIG_VERSION}" in r.output and "pipeline.stages" in r.output
    assert f"config_version = {CONFIG_VERSION}" in config.read_text(encoding="utf-8")
    assert list(desktop_library.glob("config.toml.bak-v1-*"))
    log = (data / "logs" / "migrate.log").read_text(encoding="utf-8")
    assert "version 1 -> " in log and "migrated: 0003" in log


def test_migrate_desktop_uses_the_config_the_app_would_use(desktop_library, tmp_path, monkeypatch):
    data = tmp_path / "elsewhere"
    data.mkdir()
    build_at(data / "vision_tagger.sqlite3", "0003")
    cfg = tmp_path / "custom.toml"
    cfg.write_text(f"[paths]\ndata_dir = {json.dumps(data.as_posix())}\n")
    monkeypatch.setenv("VT_CONFIG", str(cfg))
    r = migrate_desktop()
    assert r.exit_code == 0, r.output
    assert list(data.glob("*.bak-0003-*")) and not desktop_library.exists()


def test_migrate_desktop_refuses_a_newer_database(desktop_library):
    D.ensure_user_config(desktop_library)
    data = desktop_library / "data"
    data.mkdir()
    build_at(data / "vision_tagger.sqlite3", head_revision())
    con = sqlite3.connect(data / "vision_tagger.sqlite3")
    con.execute("UPDATE alembic_version SET version_num = '9999'")
    con.commit()
    con.close()
    r = migrate_desktop()
    assert r.exit_code == 3, r.output
    assert "newer version" in r.output
    assert not list(data.glob("*.bak-*"))  # nothing was touched
    engine = migration_engine(url_of(data / "vision_tagger.sqlite3"))
    assert current_revision(engine) == "9999"
    engine.dispose()


def test_migrate_desktop_reports_a_failure(desktop_library):
    D.ensure_user_config(desktop_library)
    (desktop_library / "data").mkdir()
    (desktop_library / "data" / "vision_tagger.sqlite3").write_bytes(b"not a database" * 100)
    r = migrate_desktop()
    assert r.exit_code == 4, r.output
    assert "migration failed" in r.output
    assert "Traceback" in (desktop_library / "data/logs/migrate.log").read_text(encoding="utf-8")


def test_quit_commands_without_a_library_create_nothing(desktop_library, tmp_path):
    r = CliRunner().invoke(cli_app, ["quit-app"])
    assert r.exit_code == 0 and "not running" in r.output
    assert D.main(["--quit"]) == 0
    assert not desktop_library.exists() and not (tmp_path / "config.toml").exists()


def test_quit_app_command_quits_the_running_app(desktop_library, monkeypatch):
    D.ensure_user_config(desktop_library)
    settings = config_mod.load_settings(desktop_library / "config.toml")
    thread = start_app(settings, WindowUntilQuit())
    monkeypatch.setattr(config_mod, "_config_file", None)  # as in a fresh process
    r = CliRunner().invoke(cli_app, ["quit-app", "--timeout", "60"])
    thread.join(10)
    assert r.exit_code == 0, r.output
    assert "has quit" in r.output and not thread.is_alive()
