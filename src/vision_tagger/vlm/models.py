"""The Ollama models Mira needs, and downloading them through Ollama.

Ollama keeps the models for properties and search; Mira asks it to download them
(``POST /api/pull``) instead of telling the user to run ``ollama pull`` in a terminal: from
the web UI (Pipeline page, Home), from ``vision-tagger models pull`` and, when a pipeline run
needs a model, automatically (``vlm.auto_pull``). With Docker or on a server it is the
server's Ollama that downloads.

The download answers with one JSON object per line::

    {"status": "pulling manifest"}
    {"status": "pulling 1703…", "digest": "sha256:1703…", "total": 4683075271, "completed": 2419}
    {"status": "verifying sha256 digest"}
    {"status": "writing manifest"}
    {"status": "success"}

or ``{"error": "…"}``. Ollama keeps the layers it has downloaded, so a cancelled download
continues where it stopped.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import queue
import socket
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..i18n import N_, Lazy, LazyJoin
from .backend import VlmError

log = logging.getLogger(__name__)

DOWNLOAD_URL = "https://ollama.com/download"
CONNECT_TIMEOUT = 10.0  # connecting to Ollama; the download itself has no read timeout
STATUS_TIMEOUT = 3.0  # version and model list: Ollama answers at once or not at all

#: what a model is used for, by the key forms use to name it (translated when shown)
PURPOSES = {
    "vlm": N_("Properties"),
    "embed": N_("Search index"),
    "search": N_("Search queries"),
}

# words in Ollama's error messages (Go error texts on Linux, macOS and Windows)
_UNKNOWN = ("file does not exist", "manifest unknown", "not found", "status code 404")
_DISK_FULL = ("no space left", "not enough space", "disk full", "disk quota")
_OFFLINE = (
    "dial tcp",
    "no such host",
    "i/o timeout",
    "tls handshake",
    "network is unreachable",
    "connection refused",
    "connection reset",
    "temporary failure in name resolution",
)


class PullError(VlmError):
    """A model could not be downloaded; ``text`` says why (translated when shown)."""

    def __init__(self, text: Lazy) -> None:
        super().__init__(text.message)
        self.text = text

    def __str__(self) -> str:
        return str(self.text)


class OllamaUnreachable(PullError):
    """Ollama does not answer at all (not installed, not started, wrong ``vlm.base_url``)."""


class PullCancelled(Exception):
    """The download was cancelled (pause, stop, the Cancel button); Ollama keeps the layers."""


# --------------------------------------------------------------------------- required models
@dataclass(frozen=True)
class RequiredModel:
    name: str  # as Ollama lists it: "nomic-embed-text:latest"
    key: str  # "vlm", "embed" or "search": how forms refer to it (never by name)
    purposes: tuple[str, ...]  # N_() texts


def normalize(name: str) -> str:
    """Ollama's full name of a model: ``nomic-embed-text`` -> ``nomic-embed-text:latest``.

    Only the last path part carries the tag (``host:5000/ns/model`` has none).
    """
    name = name.strip()
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def required_models(settings: Any) -> list[RequiredModel]:
    """The Ollama models this configuration needs, each once (none for the fake backend)."""
    if settings.vlm.backend != "ollama":
        return []
    entries: dict[str, tuple[str, list[str]]] = {}
    wanted = [
        ("vlm", settings.vlm.model),
        ("embed", settings.search.embed_model),
        ("search", settings.search.model or settings.vlm.model),
    ]
    for key, raw in wanted:
        if not raw or not raw.strip():
            continue
        name = normalize(raw)
        if name in entries:
            if PURPOSES[key] not in entries[name][1]:
                entries[name][1].append(PURPOSES[key])
        else:
            entries[name] = (key, [PURPOSES[key]])
    return [RequiredModel(name, key, tuple(p)) for name, (key, p) in entries.items()]


def missing_models(
    required: Iterable[RequiredModel], installed: Iterable[str]
) -> list[RequiredModel]:
    have = {normalize(n) for n in installed}
    return [m for m in required if m.name not in have]


# --------------------------------------------------------------------------- status
@dataclass
class OllamaStatus:
    url: str
    enabled: bool = True  # False: the fake backend, no Ollama needed
    reachable: bool = False
    version: str | None = None
    error: str | None = None
    installed: frozenset[str] = frozenset()
    required: list[RequiredModel] = field(default_factory=list)

    @property
    def missing(self) -> list[RequiredModel]:
        return missing_models(self.required, self.installed) if self.reachable else []

    def is_installed(self, model: RequiredModel) -> bool:
        return model.name in self.installed


def _client(base_url: str, timeout: httpx.Timeout | float) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)


def installed_models(http: httpx.Client) -> frozenset[str]:
    """The models Ollama has (``GET /api/tags``), normalised."""
    resp = http.get("/api/tags", timeout=STATUS_TIMEOUT)
    resp.raise_for_status()
    tags = resp.json().get("models") or []
    return frozenset(normalize(n) for m in tags for n in (m.get("name"), m.get("model")) if n)


def ollama_status(settings: Any, client: httpx.Client | None = None) -> OllamaStatus:
    """Is Ollama there, and which of the required models does it have? Never raises."""
    status = OllamaStatus(url=settings.vlm.base_url, required=required_models(settings))
    if settings.vlm.backend != "ollama":
        status.enabled = False
        return status
    own = client is None
    http = client or _client(settings.vlm.base_url, STATUS_TIMEOUT)
    try:
        resp = http.get("/api/version", timeout=STATUS_TIMEOUT)
        resp.raise_for_status()
        status.version = resp.json().get("version")
        status.installed = installed_models(http)
        status.reachable = True
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        status.error = f"{type(exc).__name__}: {exc}"
    finally:
        if own:
            http.close()
    return status


# --------------------------------------------------------------------------- download
@dataclass
class PullProgress:
    """Progress of one download, added up over its layers."""

    model: str
    status: str = ""  # Ollama's last status line ("pulling 170370233dd5")
    phase: str = "manifest"  # manifest, download, verify, write, done
    layers: dict[str, tuple[int, int]] = field(default_factory=dict)  # digest: (done, total)

    @property
    def completed(self) -> int:
        return sum(done for done, _total in self.layers.values())

    @property
    def total(self) -> int:
        """Bytes of the layers seen so far (Ollama announces them one after another)."""
        return sum(total for _done, total in self.layers.values())

    @property
    def percent(self) -> int | None:
        total = self.total
        return min(100, 100 * self.completed // total) if total else None

    def update(self, line: dict[str, Any]) -> None:
        status = str(line.get("status") or "")
        self.status = status
        digest = line.get("digest")
        if digest:
            total = int(line.get("total") or 0)
            done = int(line.get("completed") or 0)
            old_done, old_total = self.layers.get(digest, (0, 0))
            total = max(total, old_total)
            self.layers[digest] = (max(old_done, min(done, total) if total else done), total)
            self.phase = "download"
            return
        low = status.lower()
        if low.startswith("pulling manifest"):
            self.phase = "manifest"
        elif low.startswith("verifying"):
            self.phase = "verify"
            self.layers = {d: (t, t) for d, (_c, t) in self.layers.items()}
        elif low.startswith(("writing", "removing")):
            self.phase = "write"
        elif low == "success":
            self.phase = "done"
            self.layers = {d: (t, t) for d, (_c, t) in self.layers.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "phase": self.phase,
            "completed": self.completed,
            "total": self.total,
            "percent": self.percent,
        }


def unreachable_text(url: str) -> Lazy:
    return Lazy(
        "Ollama is not reachable at %(url)s. Install it from %(link)s and start it; Mira then "
        "downloads the models itself.",
        url=url,
        link=DOWNLOAD_URL,
    )


def error_text(model: str, error: str) -> Lazy:
    """A readable reason for Ollama's error message."""
    low = error.lower()
    if any(word in low for word in _DISK_FULL):
        return Lazy(
            "Not enough disk space for %(model)s. Free up space on the drive where Ollama keeps "
            "its models, then try again.",
            model=model,
        )
    if any(word in low for word in _OFFLINE):
        return Lazy(
            "Ollama cannot reach the model library to download %(model)s (%(error)s). Check the "
            "internet connection, then try again.",
            model=model,
            error=error,
        )
    if any(word in low for word in _UNKNOWN):
        return Lazy(
            "Ollama does not know the model %(model)s. Check the model name in config.toml.",
            model=model,
        )
    return Lazy("Ollama could not download %(model)s: %(error)s", model=model, error=error)


def _error_of(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error") or resp.text[:300])
    except (ValueError, AttributeError):
        return f"HTTP {resp.status_code}: {resp.text[:300]}"


def _abort(resp: httpx.Response) -> None:
    """Wake up a read that waits for the next line by ending the connection's socket.

    ``shutdown`` wakes a blocked read on Linux and macOS; on Windows it doesn't, there only
    closing the socket aborts it.
    """
    stream = resp.extensions.get("network_stream")
    sock = stream.get_extra_info("socket") if stream is not None else None
    if isinstance(sock, socket.socket):
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        if sys.platform == "win32":
            with contextlib.suppress(OSError):
                sock.close()


_END = object()  # the reader's last item: the response ended normally


def _read_lines(resp: httpx.Response, lines: queue.Queue[object]) -> None:
    """Reader thread: hands every line, then ``_END`` or the exception, to the download loop."""
    try:
        for raw in resp.iter_lines():
            lines.put(raw)
    except BaseException as exc:  # handed over, raised in the download loop
        lines.put(exc)
    else:
        lines.put(_END)


def _close(resp: httpx.Response, reader: threading.Thread | None, own: httpx.Client | None) -> None:
    """End the response without ever blocking the caller on a read that can't be woken."""
    if reader is not None and reader.is_alive():
        _abort(resp)
        reader.join(timeout=2)
    if reader is not None and reader.is_alive():
        # the read is stuck somewhere no abort reaches: let it end in the background
        def later() -> None:
            reader.join()
            _close(resp, None, own)

        threading.Thread(target=later, name="vision-tagger-pull-close", daemon=True).start()
        return
    with contextlib.suppress(Exception):
        resp.close()
    if own is not None:
        with contextlib.suppress(Exception):
            own.close()


def _is_cancelled(cancel: Callable[[], bool] | None) -> bool:
    if cancel is None:
        return False
    try:
        return bool(cancel())
    except Exception as exc:  # a failing check must not end the download
        log.debug("cancel check failed: %s", exc)
        return False


def pull(
    name: str,
    *,
    base_url: str = "http://localhost:11434",
    client: httpx.Client | None = None,
    on_progress: Callable[[PullProgress], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    connect_timeout: float = CONNECT_TIMEOUT,
    poll: float = 0.25,
) -> PullProgress:
    """Let Ollama download ``name``; returns the final progress.

    ``on_progress`` gets the progress after every line, ``cancel`` is asked at least every
    ``poll`` seconds (also while Ollama is silent: it verifies a big layer for minutes without a
    line). The lines are read in a thread of their own, so a cancel never waits for the next one,
    on every system. Raises :class:`OllamaUnreachable`, :class:`PullError` or
    :class:`PullCancelled`.
    """
    progress = PullProgress(model=name)
    if _is_cancelled(cancel):
        raise PullCancelled(name)
    url = str(client.base_url).rstrip("/") if client is not None else base_url
    # no read timeout: verifying a layer of several GB can take minutes without a line
    timeout = httpx.Timeout(None, connect=connect_timeout)
    own = None if client is not None else _client(base_url, timeout)
    http = client or own
    assert http is not None
    try:
        request = http.build_request(
            "POST", "/api/pull", json={"model": name, "stream": True}, timeout=timeout
        )
        resp = http.send(request, stream=True)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        if own is not None:
            own.close()
        raise OllamaUnreachable(unreachable_text(url)) from exc
    except httpx.HTTPError as exc:
        if own is not None:
            own.close()
        raise _interrupted(name, exc) from exc
    reader: threading.Thread | None = None
    cancelled = False
    try:
        if resp.status_code != 200:
            resp.read()
            raise PullError(error_text(name, _error_of(resp)))
        lines: queue.Queue[object] = queue.Queue()
        reader = threading.Thread(
            target=_read_lines, args=(resp, lines), name="vision-tagger-pull-read", daemon=True
        )
        reader.start()
        while True:
            try:
                item = lines.get(timeout=poll)
            except queue.Empty:
                item = None
            if _is_cancelled(cancel):
                cancelled = True
                raise PullCancelled(name)
            if item is None:
                continue
            if item is _END:
                break
            if isinstance(item, BaseException):
                raise item
            raw = str(item)
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
            except ValueError:
                log.debug("ignoring a line of the download of %s: %r", name, raw[:200])
                continue
            if not isinstance(line, dict):
                continue
            if line.get("error"):
                raise PullError(error_text(name, str(line["error"])))
            progress.update(line)
            if on_progress is not None:
                on_progress(progress)
    except httpx.HTTPError as exc:
        if cancelled or _is_cancelled(cancel):
            raise PullCancelled(name) from exc
        raise _interrupted(name, exc) from exc
    finally:
        _close(resp, reader, own)
    if progress.phase != "done":
        if _is_cancelled(cancel):
            raise PullCancelled(name)
        raise PullError(
            Lazy(
                "The download of %(model)s ended early. Try again; it continues where it stopped.",
                model=name,
            )
        )
    return progress


def _interrupted(name: str, exc: Exception) -> PullError:
    return PullError(
        Lazy(
            "The download of %(model)s was interrupted (%(error)s). Try again; it continues "
            "where it stopped.",
            model=name,
            error=f"{type(exc).__name__}: {exc}",
        )
    )


def pull_missing(
    settings: Any,
    *,
    keys: Iterable[str] | None = None,
    on_progress: Callable[[int, list[str], PullProgress], None] | None = None,
    on_start: Callable[[int, list[str]], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    client: httpx.Client | None = None,
) -> list[str]:
    """Download the required models Ollama does not have yet, one after another.

    ``keys`` limits it to some of :func:`required_models` (``"embed"``). The callbacks get the
    index of the current model in the list of models to download. Returns that list.
    """
    status = ollama_status(settings, client)
    if not status.enabled:
        return []
    if not status.reachable:
        raise OllamaUnreachable(unreachable_text(status.url))
    wanted = set(keys) if keys is not None else None
    todo = [m.name for m in status.missing if wanted is None or m.key in wanted]
    for index, name in enumerate(todo):
        if on_start is not None:
            on_start(index, todo)
        report = functools.partial(on_progress, index, todo) if on_progress else None
        log.info("downloading the Ollama model %s (%d/%d)", name, index + 1, len(todo))
        pull(name, base_url=status.url, client=client, on_progress=report, cancel=cancel)
        log.info("downloaded %s", name)
    return todo


# --------------------------------------------------------------------------- display
def fmt_size(n: int | float) -> str:
    """Decimal units, as Ollama shows them: ``4.7 GB``, ``274 MB``."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e3:.0f} KB"


def phase_text(progress: dict[str, Any]) -> Lazy:
    """How far a download is (``progress`` as :meth:`PullProgress.as_dict` makes it)."""
    phase = progress.get("phase")
    total = int(progress.get("total") or 0)
    if phase == "download" and total:
        return Lazy(
            "%(done)s of %(total)s (%(percent)s %%)",
            done=fmt_size(int(progress.get("completed") or 0)),
            total=fmt_size(total),
            percent=progress.get("percent") or 0,
        )
    if phase == "verify":
        return Lazy("checking the download")
    if phase == "write":
        return Lazy("finishing")
    if phase == "done":
        return Lazy("ready")
    return Lazy("starting the download")


def pull_message(progress: dict[str, Any]) -> Lazy:
    """``1/2 · qwen2.5vl:7b · 1.2 GB of 6.0 GB (20 %)``: translated when shown."""
    parts: list[Any] = [str(progress.get("model", "")), phase_text(progress)]
    count = int(progress.get("count") or 1)
    if count > 1:
        parts.insert(0, f"{int(progress.get('index') or 0) + 1}/{count}")
    return LazyJoin(" · ", parts)
