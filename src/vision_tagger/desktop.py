"""The desktop app (local mode): Mira's web UI in a window of its own.

``vision-tagger desktop``, ``vision-tagger-app`` and the installed app (Windows, macOS, Linux)
start the server on a free local port and show the UI in a native window:

1. pywebview (WebView2 on Windows, WebKit on macOS, GTK/Qt on Linux), from the ``desktop`` extra;
2. otherwise Edge, Chrome or Chromium in app mode (a window without tabs and address bar);
3. otherwise the default browser.

Closing the window stops the server; a running pipeline is paused and continues on the next
start. Starting the app again while it runs opens another window of the running app instead of
a second server. ``Mira --quit`` (or ``vision-tagger-cli quit-app``) closes a running app the
same way, e.g. before an installer replaces its files.

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


def find_desktop_settings() -> Settings | None:
    """The settings the desktop app would use, without creating anything.

    None: there is no library yet (the app never ran, so it has not written its config).
    For installers and the command line (``migrate --desktop``, ``quit-app``).
    """
    from . import config

    config_file = config._resolve_config_file()
    if config_file is None:
        config_file = user_data_dir() / "config.toml"
        if not config_file.is_file():
            return None
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
        self.closing = threading.Event()

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
        if self.closing.is_set():
            return
        self._create(url)
        try:
            # not private: theme, language and the upload queue survive a restart. start()
            # runs the GUI loop in this thread, the main thread, as macOS requires.
            self.webview.start(private_mode=False, storage_path=str(self.storage))
        except Exception as exc:  # e.g. no GUI toolkit on Linux
            raise WindowUnavailable(f"pywebview could not open a window: {exc}") from exc

    def another(self, url: str) -> None:
        self._create(url)

    def quit(self) -> None:
        """Close every window (from any thread); ``show`` then returns."""
        self.closing.set()
        for window in list(self.webview.windows):
            try:
                window.destroy()
            except Exception:  # e.g. a window that never appeared
                log.exception("could not close a window")


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


def _wait_until_idle(
    idle_seconds: IdleClock, limit: float, stop: threading.Event, poll: float = 5.0
) -> None:
    """Until nobody used the app for ``limit`` seconds, Ctrl+C or ``stop`` (a quit request)."""
    with contextlib.suppress(KeyboardInterrupt):
        while idle_seconds() < limit:
            if stop.wait(poll):
                return


class BrowserAppWindow:
    """Edge/Chrome in app mode, with a profile of its own so that its process is the window."""

    name = "browser app window"

    def __init__(
        self, command: list[str], profile: Path, idle_limit: float = IDLE_LIMIT_SECONDS
    ) -> None:
        self.command = command
        self.profile = profile
        self.idle_limit = idle_limit
        self.closing = threading.Event()
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen[bytes]] = []

    def _launch(self, url: str) -> subprocess.Popen[bytes] | None:
        with self._lock:  # a quit request either sees this process or prevents it
            if self.closing.is_set():
                return None
            proc = subprocess.Popen(
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
            self._procs.append(proc)
            return proc

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        self.profile.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            proc = self._launch(url)
        except OSError as exc:
            raise WindowUnavailable(f"could not start {self.command[0]}: {exc}") from exc
        if proc is None:
            return  # asked to quit before the window opened
        with contextlib.suppress(KeyboardInterrupt):
            proc.wait()
        if self.closing.is_set():
            return
        if time.monotonic() - started < QUICK_EXIT_SECONDS:
            # the window went to a browser process that was already running (e.g. left over
            # from an earlier start): there is no process to wait for, so wait for idleness
            log.info("the window belongs to another browser process; quitting when idle")
            _wait_until_idle(idle_seconds, self.idle_limit, self.closing)

    def another(self, url: str) -> None:
        self._launch(url)

    def quit(self) -> None:
        """End the browser processes this app started; ``show`` then returns."""
        with self._lock:
            self.closing.set()
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.terminate()


class DefaultBrowser:
    """Last resort: a tab in the default browser; the app quits when nobody uses it."""

    name = "default browser"

    def __init__(self, idle_limit: float = IDLE_LIMIT_SECONDS) -> None:
        self.idle_limit = idle_limit
        self.closing = threading.Event()

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        with _child_environment_applied():
            opened = webbrowser.open(url)
        if not opened:
            raise WindowUnavailable("no web browser found")
        print(f"{APP_NAME} runs on {url} - press Ctrl+C to quit", flush=True)
        _wait_until_idle(idle_seconds, self.idle_limit, self.closing)

    def another(self, url: str) -> None:
        with _child_environment_applied():
            webbrowser.open(url)

    def quit(self) -> None:
        """Stop waiting; the browser tab stays (it is not the app's to close)."""
        self.closing.set()


class Windows:
    """Tries each way of showing the UI in turn; ``show`` blocks until the window is closed."""

    def __init__(self, backends: list[Callable[[], Any]]) -> None:
        self.backends = backends
        self.active: Any = None
        self.closing = threading.Event()

    def show(self, url: str, idle_seconds: IdleClock) -> None:
        for make in self.backends:
            if self.closing.is_set():
                return
            try:
                backend = make()
                self.active = backend
                if self.closing.is_set():  # asked to quit while the backend was set up
                    self._quit(backend)
                backend.show(url, idle_seconds)
                return
            except WindowUnavailable as exc:
                log.info("%s", exc)
        raise RuntimeError("found no way to show a window")

    def another(self, url: str) -> None:
        if self.active is not None and not self.closing.is_set():
            self.active.another(url)

    def quit(self) -> None:
        """Close the window(s) as if the user did: ``show`` returns and the app shuts down."""
        self.closing.set()
        if self.active is not None:
            self._quit(self.active)

    @staticmethod
    def _quit(backend: Any) -> None:
        stop = getattr(backend, "quit", None)
        if stop is not None:
            stop()


def default_windows(settings: Settings) -> Windows:
    storage = settings.data_dir / "window"

    def browser_window() -> BrowserAppWindow:
        browser = find_browser()
        if browser is None:
            raise WindowUnavailable("no Edge, Chrome or Chromium found")
        return BrowserAppWindow([browser], storage / "browser")

    backends: list[Callable[[], Any]] = [browser_window, DefaultBrowser]
    # the frozen Linux app has no GTK or Qt bindings for pywebview (they come with the system's
    # Python, which it does not use): straight to the browser window, without failed attempts
    if not (getattr(sys, "frozen", False) and sys.platform.startswith("linux")):
        backends.insert(0, lambda: WebviewWindow(storage))
    return Windows(backends)


# --------------------------------------------------------------------------- one instance
class InstanceLock:
    """One app per library: a second start opens a window of the running app instead.

    The running app writes ``desktop.json`` (control port, a secret token, its process id) and
    answers ``<token> open`` and ``<token> quit`` on the control port with ``ok``. After a quit
    it keeps the connection open until it has shut down, so the asking side can wait for that.
    Apps from before ``quit`` existed close the connection without an answer.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = secrets.token_hex(16)
        self._server: socket.socket | None = None
        self._waiting: list[socket.socket] = []  # quit requests, closed by finished()

    def notify_running(self) -> bool:
        """Ask a running app to open another window. False: no app is running."""
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            with socket.create_connection(("127.0.0.1", int(info["control"])), timeout=3) as s:
                s.sendall(f"{info['token']} open\n".encode())
                return s.recv(16).startswith(b"ok")
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def listen(
        self,
        url: str,
        on_open: Callable[[], None],
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        self._server = server
        info = {
            "url": url,
            "control": server.getsockname()[1],
            "token": self.token,
            "pid": os.getpid(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(info), encoding="utf-8")
        threading.Thread(
            target=self._serve, args=(server, on_open, on_quit), name="vt-instance", daemon=True
        ).start()

    def _serve(
        self,
        server: socket.socket,
        on_open: Callable[[], None],
        on_quit: Callable[[], None] | None,
    ) -> None:
        token = self.token.encode()
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return  # closed by release()
            keep = False
            try:
                conn.settimeout(3)
                given, _, command = conn.recv(256).strip().partition(b" ")
                if not hmac.compare_digest(given, token):
                    continue
                if command == b"open":
                    action, what = on_open, "open another window"
                elif command == b"quit" and on_quit is not None:
                    action, what, keep = on_quit, "quit", True
                else:
                    continue  # unknown request: no answer, like an older app
                conn.sendall(b"ok\n")
            except OSError:
                keep = False
                continue
            finally:
                if keep:
                    self._waiting.append(conn)  # closed once the app has shut down
                else:
                    conn.close()
            try:
                action()
            except Exception:
                log.exception("could not %s", what)

    def release(self) -> None:
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
            self._server = None
        with contextlib.suppress(OSError, ValueError):
            if json.loads(self.path.read_text(encoding="utf-8")).get("token") == self.token:
                self.path.unlink()

    def finished(self) -> None:
        """The app has shut down: tell whoever asked it to quit."""
        while self._waiting:
            with contextlib.suppress(OSError):
                self._waiting.pop().close()


#: results of :func:`quit_running`
NOT_RUNNING, QUIT, UNSUPPORTED, STILL_RUNNING = "not running", "quit", "unsupported", "running"


def quit_running(data_dir: Path, timeout: float = 120.0) -> str:
    """Ask the app that runs on this library to quit, as if its window was closed.

    A running pipeline is paused (it continues on the next start). Waits until the app has
    exited, at most ``timeout`` seconds. Returns ``NOT_RUNNING``, ``QUIT``, ``UNSUPPORTED`` (the
    running app is too old to quit on request) or ``STILL_RUNNING`` (it did not exit in time).
    """
    deadline = time.monotonic() + timeout
    try:
        info = json.loads((data_dir / "desktop.json").read_text(encoding="utf-8"))
        port, token, pid = int(info["control"]), str(info["token"]), info.get("pid")
    except (OSError, ValueError, KeyError, TypeError):
        return NOT_RUNNING
    if not isinstance(pid, int) or isinstance(pid, bool) or pid == os.getpid():
        pid = None  # an older app wrote no pid; in tests the app runs in this process
    elif _wait_for_exit(pid, 0):
        return NOT_RUNNING  # desktop.json left behind by an app that crashed
    try:
        conn = socket.create_connection(("127.0.0.1", port), timeout=3)
    except OSError:
        return NOT_RUNNING  # nobody listens: left behind as well
    with conn:
        try:
            conn.sendall(f"{token} quit\n".encode())
            answer = conn.recv(16)
        except OSError:
            answer = b""
        if not answer.startswith(b"ok"):
            return UNSUPPORTED
        while True:  # the app closes the connection once it is done, or by exiting
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return STILL_RUNNING
            conn.settimeout(min(remaining, 1.0))
            try:
                if not conn.recv(16):
                    break
            except TimeoutError:
                continue
            except OSError:
                break
    if pid is not None and not _wait_for_exit(pid, max(0.0, deadline - time.monotonic())):
        return STILL_RUNNING
    return QUIT


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """True once process ``pid`` has exited; False if it still runs after ``timeout``."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            # ERROR_INVALID_PARAMETER: no such process; otherwise it exists but is not ours
            return ctypes.get_last_error() == 87  # type: ignore[attr-defined]
        try:
            return kernel32.WaitForSingleObject(handle, int(timeout * 1000)) == 0
        finally:
            kernel32.CloseHandle(handle)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # it exists, but belongs to another user
        else:
            with contextlib.suppress(OSError, IndexError):  # Linux: an unreaped zombie
                stat = Path(f"/proc/{pid}/stat").read_text()
                if stat.rpartition(")")[2].split()[0] == "Z":
                    return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


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
    lock.listen(server.url, lambda: windows.another(server.url), windows.quit)
    try:
        windows.show(server.url, server.clock.idle_seconds)
    finally:
        lock.release()
        log.info("window closed; stopping (a running pipeline is paused)")
        try:
            server.stop()
        finally:
            lock.finished()
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
    if "--quit" in args:
        return quit_app()
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


#: exit codes of ``Mira --quit`` and ``vision-tagger-cli quit-app``
QUIT_EXIT_CODES = {NOT_RUNNING: 0, QUIT: 0, STILL_RUNNING: 1, UNSUPPORTED: 3}


def quit_app(timeout: float = 120.0, echo: Callable[[str], Any] = print) -> int:
    """Quit the desktop app running on the user's library (for installers); see QUIT_EXIT_CODES."""
    try:
        settings = find_desktop_settings()
    except Exception as exc:
        echo(f"could not read the settings: {exc}")
        return 1
    result = NOT_RUNNING if settings is None else quit_running(settings.data_dir, timeout)
    echo(
        {
            NOT_RUNNING: f"{APP_NAME} is not running",
            QUIT: f"{APP_NAME} has quit (a running pipeline is paused until the next start)",
            STILL_RUNNING: f"{APP_NAME} is still running after {timeout:g} s",
            UNSUPPORTED: f"the running {APP_NAME} is too old to quit on request: close it",
        }[result]
    )
    return QUIT_EXIT_CODES[result]


# --------------------------------------------------------------------------- self-test
def self_test() -> int:
    """Check a build (e.g. the frozen app): imports, resources, migrations, pages.

    Returns 0 when everything works. The desktop workflow runs it before packaging; the first
    line, ``Mira <version>``, shows that the build has the release's version.
    """
    import tempfile

    import httpx
    from PIL import Image

    from . import __version__

    def step(name: str) -> None:
        print(f"  ok  {name}", flush=True)

    print(f"{APP_NAME} {__version__}", flush=True)

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
