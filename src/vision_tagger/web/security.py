"""Protection for a web UI that listens on localhost.

* **DNS rebinding:** a malicious site can make its own domain resolve to 127.0.0.1 and then
  read the UI from the browser. Only requests whose ``Host`` header names this machine are
  served.
* **CSRF:** any site open in the browser could otherwise submit forms to the UI (start an
  export to a network share, delete persons, ...). State-changing requests must come from the
  UI itself, checked with ``Origin`` / ``Sec-Fetch-Site`` / ``Referer``.
* **This computer only:** what acts on the server's desktop (opening Explorer or Finder) is
  offered only to requests made on the server's own computer (:func:`from_this_computer`).
"""

from __future__ import annotations

import ipaddress
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


#: headers a reverse proxy adds: the request was not made on this computer, even if the proxy is
PROXY_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")


def _loopback(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)  # ::ffff:127.0.0.1
    return ip.is_loopback or (mapped is not None and mapped.is_loopback)


def is_loopback_host(host: str) -> bool:
    """A bind address or host name that only accepts connections from this computer."""
    name = host.strip().strip("[]").lower()
    return name == "localhost" or _loopback(name)


def exposure_problem(remote_mode: bool, host: str, password: str) -> str | None:
    """Why serving on ``host`` without ``password`` would be unsafe; ``None`` if it is fine.

    Without a password every request is trusted. That is right only on a loopback address in
    local mode: the Host check below blocks DNS rebinding from browsers, but anyone on the
    network can send any Host header. So remote mode, and any address that other computers can
    reach, needs a password (``vision-tagger serve`` refuses to start without one).
    """
    if password:
        return None
    if remote_mode:
        return (
            "remote mode needs a password: without one, anyone who reaches the server sees "
            "every photo."
        )
    if not is_loopback_host(host):
        return (
            f"{host} can be reached from other computers: without a password, anyone there "
            "sees every photo and can browse the folders of this computer."
        )
    return None


def from_this_computer(request: Request) -> bool:
    """The request comes from a browser on the computer the server runs on: from a loopback
    address (127.0.0.1, ::1), under a local host name, and not through a reverse proxy."""
    client = request.client.host if request.client is not None else ""
    if not _loopback(client):
        return False
    if _hostname(request.headers.get("host", "")) not in LOCAL_HOSTS:
        return False
    return not any(h in request.headers for h in PROXY_HEADERS)


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
