from __future__ import annotations

import os
import sys
import threading
import time

import httpx
import pytest

from vision_tagger import config as config_mod
from vision_tagger import desktop as D
from vision_tagger.db.migrate import current_revision, head_revision, migration_engine

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
