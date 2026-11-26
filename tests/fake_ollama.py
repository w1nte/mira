"""A small stand-in for Ollama on localhost: ``/api/version``, ``/api/tags`` and a streaming
``/api/pull`` that adds the model to the list once it is done (no network, no models)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from vision_tagger.vlm.models import normalize

UNKNOWN_ERROR = "pull model manifest: file does not exist"


class FakeOllama:
    """``with FakeOllama(installed=["qwen2.5vl:7b"]) as ollama: ... ollama.url``

    * ``layers``: sizes of the layers every model consists of;
    * ``delay``: seconds between two progress lines;
    * ``silence``: seconds without a line before "verifying" (Ollama checks a big layer);
    * ``errors``: model -> error line sent after the first layer; models starting with
      ``unknown`` are unknown.
    """

    def __init__(
        self,
        installed: tuple[str, ...] | list[str] = (),
        *,
        layers: tuple[int, ...] = (3_000_000_000, 1_000_000),
        delay: float = 0.0,
        silence: float = 0.0,
        errors: dict[str, str] | None = None,
    ) -> None:
        self.installed = {normalize(n) for n in installed}
        self.layers = layers
        self.delay = delay
        self.silence = silence
        self.errors = {normalize(k): v for k, v in (errors or {}).items()}
        self.pulls: list[str] = []  # in the order they started
        self.finished: list[str] = []
        self.aborted: list[str] = []  # the client went away during the download
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------ server
    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeOllama:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # a streamed body ends when the connection closes

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/api/version":
                    self._json({"version": "0.12.0"})
                elif self.path == "/api/tags":
                    with fake._lock:
                        names = sorted(fake.installed)
                    self._json({"models": [{"name": n, "model": n} for n in names]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/pull":
                    fake._pull(self, normalize(str(body.get("model", ""))))
                elif self.path == "/api/generate":
                    self._json({"done": True})
                else:
                    self._json({"error": "not found"}, 404)

            def _json(self, data: Any, status: int = 200) -> None:
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    # ------------------------------------------------------------------ pull
    def lines(self, model: str) -> list[dict[str, Any]]:
        """The progress lines of a download (without the pauses)."""
        out: list[dict[str, Any]] = [{"status": "pulling manifest"}]
        for i, size in enumerate(self.layers):
            digest = "sha256:" + hashlib.sha256(f"{model}/{i}".encode()).hexdigest()
            for done in (0, size // 2, size):
                out.append(
                    {"status": f"pulling {digest[7:19]}", "digest": digest, "total": size}
                    | ({"completed": done} if done else {})
                )
        out += [
            {"status": "verifying sha256 digest"},
            {"status": "writing manifest"},
            {"status": "success"},
        ]
        return out

    def _pull(self, handler: BaseHTTPRequestHandler, model: str) -> None:
        with self._lock:
            self.pulls.append(model)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "application/x-ndjson")
            handler.end_headers()
            lines = self.lines(model)
            if model.startswith("unknown"):
                lines = [lines[0], {"error": UNKNOWN_ERROR}]
            elif model in self.errors:
                lines = [*lines[:4], {"error": self.errors[model]}]
            for line in lines:
                if line.get("status", "").startswith("verifying") and self.silence:
                    time.sleep(self.silence)
                handler.wfile.write(json.dumps(line).encode() + b"\n")
                handler.wfile.flush()
                if line.get("status") == "success":
                    with self._lock:
                        self.installed.add(model)
                        self.finished.append(model)
                time.sleep(self.delay)
        except (BrokenPipeError, ConnectionResetError):
            self.aborted.append(model)
        finally:
            with self._lock:
                self.active -= 1
