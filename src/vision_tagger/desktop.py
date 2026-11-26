"""The desktop app (local mode): Mira's web UI in a window of its own.

``vision-tagger desktop``, ``vision-tagger-app`` and the Windows installer start the server on a
free local port and show the UI in a native window:

1. pywebview (WebView2 on Windows, WebKit on macOS, GTK/Qt on Linux), from the ``desktop`` extra;
2. otherwise Edge, Chrome or Chromium in app mode (a window without tabs and address bar);
3. otherwise the default browser.

Closing the window stops the server; a running pipeline is paused and continues on the next
start. Starting the app again while it runs opens another window of the running app instead of
a second server.

Without a config file the app keeps its config and data in the user's application-data folder
(``%LOCALAPPDATA%\\vision-tagger`` on Windows), so it also works from a read-only install folder.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import APP_NAME
from .config import Settings

log = logging.getLogger(__name__)

#: folder name of the per-user data (kept from before the app was called Mira)
DATA_FOLDER = "vision-tagger"
WINDOW_SIZE = (1280, 860)
MIN_SIZE = (860, 560)
#: a browser that exits this fast handed the window to a browser that was already running
QUICK_EXIT_SECONDS = 5.0
#: then (and with the default browser) the app quits after this long without any request
IDLE_LIMIT_SECONDS = 30 * 60

IdleClock = Callable[[], float]  # seconds since the last HTTP request


# --------------------------------------------------------------------------- settings
def user_data_dir() -> Path:
    """Where the desktop app keeps its config and data when no config file is given."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / DATA_FOLDER


def _toml_string(path: Path) -> str:
    return json.dumps(path.as_posix(), ensure_ascii=False)  # JSON strings are valid TOML


def ensure_user_config(app_dir: Path) -> Path:
    """The desktop app's ``config.toml``, written on the first start with absolute paths."""
    target = app_dir / "config.toml"
    if target.exists():
        return target
    app_dir.mkdir(parents=True, exist_ok=True)
    example = Path(__file__).parent / "resources" / "config.example.toml"
    text = example.read_text(encoding="utf-8")
    values = {
        "image_roots": "[]",  # chosen on the Import page
        "data_dir": _toml_string(app_dir / "data"),
        "references_dir": _toml_string(app_dir / "references"),
    }
    for key, value in values.items():
        text, n = re.subn(
            rf"(?m)^{key}\s*=\s*[^#\n]*?(\s*#.*)?$",
            lambda m, key=key, value=value: f"{key} = {value}{m.group(1) or ''}",
            text,
            count=1,
        )
        if n != 1:  # pragma: no cover - the example file lost a setting
            raise RuntimeError(f"config.example.toml has no {key} setting")
    target.write_text(text, encoding="utf-8")
    return target


def load_desktop_settings(config_file: Path | None = None) -> Settings:
    """An explicit config, ``VT_CONFIG`` or ``./config.toml``; else the per-user config."""
    from . import config

    if config_file is None and config._resolve_config_file() is None:
        config_file = ensure_user_config(user_data_dir())
    return config.load_settings(config_file)


# --------------------------------------------------------------------------- server
class _ActivityClock:
    """ASGI wrapper that remembers when the last HTTP request arrived."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.last = time.monotonic()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            self.last = time.monotonic()
        await self.app(scope, receive, send)

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last


def _bind(port: int) -> socket.socket:
    """A socket on 127.0.0.1, at ``port`` if it is free, else at any free port."""
    for candidate in dict.fromkeys((port, 0)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", candidate))
        except OSError:
            sock.close()
            continue
        return sock
    raise OSError("no free local port")


class AppServer:
    """The web app on 127.0.0.1, served from a background thread."""

    def __init__(self, settings: Settings, port: int | None = None) -> None:
        import uvicorn

        from .web.app import create_app

        self.sock = _bind(settings.web.port if port is None else port)
        self.port: int = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/"
        self.clock = _ActivityClock(create_app(settings))
        config = uvicorn.Config(
            self.clock, host="127.0.0.1", port=self.port, log_config=None, access_log=False
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.sock]}, name="vt-server", daemon=True
        )

    def start(self, timeout: float = 300.0) -> None:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("the server did not start (details in the log)")
            if time.monotonic() > deadline:
                raise RuntimeError("the server did not start in time")
            time.sleep(0.05)
        log.info("serving on %s", self.url)

    def stop(self, timeout: float = 90.0) -> None:
        """Stops the server; the app's shutdown pauses a running pipeline first."""
        self.server.should_exit = True
        self.thread.join(timeout)
        with contextlib.suppress(OSError):
            self.sock.close()


# --------------------------------------------------------------------------- windows
class WindowUnavailable(RuntimeError):
    """This way of showing a window does not work here; the next one is tried."""


class WebviewWindow:
    """A native window via pywebview (WebView2 on Windows)."""

    name = "pywebview"

    def __init__(self, storage: Path) -> None:
        try:
            import webview
        except ImportError as exc:
            raise WindowUnavailable("pywebview is not installed") from exc
        self.webview: Any = webview
        self.storage = storage

    def _create(self, url: str) -> None:
        self.webview.create_window(
            APP_NAME,
            url,
            width=WINDOW_SIZE[0],
            height=WINDOW_SIZE[1],
            min_size=MIN_SIZE,
            text_select=True,
            zoomable=True,  # Ctrl +/-
        )

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        with contextlib.suppress(Exception):  # exports and animations are downloads
            self.webview.settings["ALLOW_DOWNLOADS"] = True
        self.storage.mkdir(parents=True, exist_ok=True)
        self._create(url)
        try:
            # not private: theme, language and the upload queue survive a restart
            self.webview.start(private_mode=False, storage_path=str(self.storage))
        except Exception as exc:  # e.g. no GUI toolkit on Linux
            raise WindowUnavailable(f"pywebview could not open a window: {exc}") from exc

    def another(self, url: str) -> None:
        self._create(url)


def find_browser() -> str | None:
    """Edge, Chrome or Chromium, for a window in app mode."""
    candidates: list[Path] = []
    if sys.platform == "win32":
        for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                candidates += [
                    Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                    Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe",
                ]
    elif sys.platform == "darwin":
        for app in ("Microsoft Edge", "Google Chrome", "Chromium"):
            candidates.append(Path("/Applications") / f"{app}.app" / "Contents" / "MacOS" / app)
    for path in candidates:
        if path.is_file():
            return str(path)
    for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def child_environment() -> dict[str, str] | None:
    """Environment for programs a frozen build starts (the browser); None: inherit as is.

    PyInstaller's bootloader (and later OpenCV) point ``LD_LIBRARY_PATH`` at the bundle; a
    browser started with it would load the bundled libraries and crash. The app itself is
    unaffected: the dynamic loader read the variable when the process started.
    """
    if not getattr(sys, "frozen", False):
        return None
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        original = env.pop(f"{var}_ORIG", None)
        if original is None:
            env.pop(var, None)
        else:
            env[var] = original
    return env


@contextlib.contextmanager
def _child_environment_applied() -> Iterator[None]:
    """For library calls that start programs without an ``env`` argument (webbrowser)."""
    env = child_environment()
    if env is None:
        yield
        return
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _wait_until_idle(idle_seconds: IdleClock, limit: float, poll: float = 5.0) -> None:
    with contextlib.suppress(KeyboardInterrupt):
        while idle_seconds() < limit:
            time.sleep(poll)


class BrowserAppWindow:
    """Edge/Chrome in app mode, with a profile of its own so that its process is the window."""

    name = "browser app window"

    def __init__(
        self, command: list[str], profile: Path, idle_limit: float = IDLE_LIMIT_SECONDS
    ) -> None:
        self.command = command
        self.profile = profile
        self.idle_limit = idle_limit

    def _launch(self, url: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                *self.command,
                f"--app={url}",
                f"--user-data-dir={self.profile}",
                "--no-first-run",
                "--no-default-browser-check",
                f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=child_environment(),
        )

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        self.profile.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            proc = self._launch(url)
        except OSError as exc:
            raise WindowUnavailable(f"could not start {self.command[0]}: {exc}") from exc
        with contextlib.suppress(KeyboardInterrupt):
            proc.wait()
        if time.monotonic() - started < QUICK_EXIT_SECONDS:
            # the window went to a browser process that was already running (e.g. left over
            # from an earlier start): there is no process to wait for, so wait for idleness
            log.info("the window belongs to another browser process; quitting when idle")
            _wait_until_idle(idle_seconds, self.idle_limit)

    def another(self, url: str) -> None:
        self._launch(url)


class DefaultBrowser:
    """Last resort: a tab in the default browser; the app quits when nobody uses it."""

    name = "default browser"

    def __init__(self, idle_limit: float = IDLE_LIMIT_SECONDS) -> None:
        self.idle_limit = idle_limit

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        with _child_environment_applied():
            opened = webbrowser.open(url)
        if not opened:
            raise WindowUnavailable("no web browser found")
        print(f"{APP_NAME} runs on {url} - press Ctrl+C to quit")
        _wait_until_idle(idle_seconds, self.idle_limit)

    def another(self, url: str) -> None:
        with _child_environment_applied():
            webbrowser.open(url)


class Windows:
    """Tries each way of showing the UI in turn; ``show`` blocks until the window is closed."""

    def __init__(self, backends: list[Callable[[], Any]]) -> None:
        self.backends = backends
        self.active: Any = None

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        for make in self.backends:
            try:
                backend = make()
                self.active = backend
                backend.show(url, idle_seconds)
                return
            except WindowUnavailable as exc:
                log.info("%s", exc)
        raise RuntimeError("found no way to show a window")

    def another(self, url: str) -> None:
        if self.active is not None:
            self.active.another(url)


def default_windows(settings: Settings) -> Windows:
    storage = settings.data_dir / "window"

    def browser_window() -> BrowserAppWindow:
        browser = find_browser()
        if browser is None:
            raise WindowUnavailable("no Edge, Chrome or Chromium found")
        return BrowserAppWindow([browser], storage / "browser")

    return Windows([lambda: WebviewWindow(storage), browser_window, DefaultBrowser])


# --------------------------------------------------------------------------- one instance
class InstanceLock:
    """A second start opens a window of the running app instead of starting a second server."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = secrets.token_hex(16)
        self._server: socket.socket | None = None

    def notify_running(self) -> bool:
        """Ask a running app to open another window. False: no app is running."""
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            with socket.create_connection(("127.0.0.1", int(info["control"])), timeout=3) as s:
                s.sendall(f"{info['token']} open\n".encode())
                return s.recv(16).startswith(b"ok")
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def listen(self, url: str, on_open: Callable[[], None]) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        self._server = server
        info = {"url": url, "control": server.getsockname()[1], "token": self.token}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(info), encoding="utf-8")
        threading.Thread(
            target=self._serve, args=(server, on_open), name="vt-instance", daemon=True
        ).start()

    def _serve(self, server: socket.socket, on_open: Callable[[], None]) -> None:
        expected = f"{self.token} open".encode()
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return  # closed by release()
            with conn:
                try:
                    conn.settimeout(3)
                    if not hmac.compare_digest(conn.recv(256).strip(), expected):
                        continue
                    conn.sendall(b"ok\n")
                except OSError:
                    continue
            try:
                on_open()
            except Exception:
                log.exception("could not open another window")

    def release(self) -> None:
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
            self._server = None
        with contextlib.suppress(OSError, ValueError):
            if json.loads(self.path.read_text(encoding="utf-8")).get("token") == self.token:
                self.path.unlink()


# --------------------------------------------------------------------------- app
def run(settings: Settings, *, windows: Windows | None = None, port: int | None = None) -> int:
    """Start the server, show the window, and stop everything when it is closed."""
    settings.ensure_dirs()
    lock = InstanceLock(settings.data_dir / "desktop.json")
    if lock.notify_running():
        log.info("%s is already running: opened another window", APP_NAME)
        return 0
    from .db.migrate import upgrade_if_needed

    if upgrade_if_needed(settings.db_path):  # nobody runs `vision-tagger migrate` here
        log.info("upgraded the database (a backup copy was made first)")
    server = AppServer(settings, port)
    server.start()
    windows = windows or default_windows(settings)
    lock.listen(server.url, lambda: windows.another(server.url))
    try:
        windows.show(server.url, server.clock.idle_seconds)
    finally:
        lock.release()
        log.info("window closed; stopping (a running pipeline is paused)")
        server.stop()
    return 0


def _has_console() -> bool:
    return sys.stdout is not None and sys.stderr is not None


def setup_logging(settings: Settings) -> Path:
    """Log to ``<data>/logs/desktop.log``; without a console, catch stray output as well."""
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    console = _has_console()
    if not console:  # windowed app: libraries that print must not fail
        stream = (log_dir / "output.log").open("w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    path = log_dir / "desktop.log"
    handlers: list[logging.Handler] = [
        RotatingFileHandler(path, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    ]
    if console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def show_error(message: str) -> None:
    """An error the user can see, also in the windowed app without a console."""
    log.error("%s", message)
    if sys.platform == "win32" and not _has_console():
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)  # type: ignore[attr-defined]
    else:
        print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the installed app and of ``vision-tagger-app``."""
    import multiprocessing

    multiprocessing.freeze_support()  # joblib/loky workers of a frozen build
    args = sys.argv[1:] if argv is None else argv
    if "--self-test" in args:
        return self_test()
    try:
        settings = load_desktop_settings()
    except Exception as exc:
        show_error(f"{APP_NAME} could not read its settings:\n{exc}")
        return 1
    log_path = setup_logging(settings)
    try:
        return run(settings)
    except Exception as exc:
        log.exception("the app failed")
        show_error(f"{APP_NAME} could not start:\n{exc}\n\nDetails: {log_path}")
        return 1


# --------------------------------------------------------------------------- self-test
def self_test() -> int:
    """Check a build (e.g. the frozen Windows app): imports, resources, migrations, pages.

    Returns 0 when everything works. The installer workflow runs it before packaging.
    """
    import tempfile

    import httpx
    from PIL import Image

    def step(name: str) -> None:
        print(f"  ok  {name}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        import cv2
        import insightface
        import onnxruntime
        import sklearn.cluster

        step(
            f"imports: onnxruntime {onnxruntime.__version__} "
            f"({', '.join(onnxruntime.get_available_providers())}), "
            f"insightface {insightface.__version__}, opencv {cv2.__version__}, "
            f"scikit-learn {sklearn.__version__}"
        )
        from .db.migrate import head_revision, upgrade

        result = upgrade(f"sqlite:///{(root / 'migrated.sqlite3').as_posix()}", backup=False)
        assert result.after == head_revision(), result
        step(f"migrations up to {result.after}")

        photos = root / "photos"
        photos.mkdir()
        Image.new("RGB", (640, 480), (40, 120, 200)).save(photos / "a.jpg", quality=90)
        settings = Settings(
            paths={"data_dir": root / "data", "image_roots": [photos]},
            vlm={"backend": "fake"},
        )
        settings.ensure_dirs()
        from .db.session import Database
        from .indexer import Indexer

        db = Database(settings.db_path)
        db.create_all()
        stats = Indexer(db, settings).run()
        db.dispose()
        assert stats.new == 1, stats
        step("indexing (Pillow, HEIF plug-in, hashes, thumbnails)")

        server = AppServer(settings, 0)
        server.start()
        try:
            pages = ("", "import", "pipeline", "images", "faces", "static/app.css", "sw.js")
            for page in pages:
                r = httpx.get(server.url + page, timeout=30)
                assert r.status_code == 200, (page, r.status_code)
            step(f"web UI ({len(pages)} pages and files)")
        finally:
            server.stop()
    print("self-test passed", flush=True)
    return 0
