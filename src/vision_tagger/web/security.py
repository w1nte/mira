"""Protection for a web UI that listens on localhost.

* **DNS rebinding:** a malicious site can make its own domain resolve to 127.0.0.1 and then
  read the UI from the browser. Only requests whose ``Host`` header names this machine are
  served.
* **CSRF:** any site open in the browser could otherwise submit forms to the UI (start an
  export to a network share, delete persons, ...). State-changing requests must come from the
  UI itself, checked with ``Origin`` / ``Sec-Fetch-Site`` / ``Referer``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _hostname(host_header: str) -> str:
    """``"[::1]:8765"`` -> ``"::1"``, ``"localhost:8765"`` -> ``"localhost"``."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host.strip("[]")
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def allowed_hostnames(bind_host: str, extra: Iterable[str]) -> set[str]:
    hosts = set(LOCAL_HOSTS) | {h.lower() for h in extra}
    if bind_host not in ("0.0.0.0", "::", ""):
        hosts.add(bind_host.lower().strip("[]"))
    return hosts


def same_origin(request: Request) -> bool:
    host = request.headers.get("host", "").lower()
    origin = request.headers.get("origin")
    if origin is not None:
        return origin != "null" and urlsplit(origin).netloc.lower() == host
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None:
        return fetch_site in ("same-origin", "none")
    referer = request.headers.get("referer")
    if referer is not None:
        return urlsplit(referer).netloc.lower() == host
    return True  # no browser headers at all: CLI tools / scripts on this machine


CallNext = Callable[[Request], Awaitable[Response]]


def guard(allowed: set[str]) -> Callable[[Request, CallNext], Awaitable[Response]]:
    """``allowed`` may contain ``"*"`` (remote mode behind a login: any host name)."""
    any_host = "*" in allowed

    async def middleware(request: Request, call_next: CallNext) -> Response:
        if not any_host and _hostname(request.headers.get("host", "")) not in allowed:
            return PlainTextResponse("invalid Host header", status_code=400)
        if request.method not in SAFE_METHODS and not same_origin(request):
            return PlainTextResponse("cross-site request blocked", status_code=403)
        return await call_next(request)

    return middleware
