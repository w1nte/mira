"""Optional password protection (``web.password``), meant for remote mode.

A successful login sets a signed, HTTP-only session cookie (valid for 30 days). Without it,
pages redirect to ``/login`` and API calls get ``401``. Wrong passwords are slowed down, and an
address that keeps guessing is locked out for a few minutes.

The cookie is signed (HMAC-SHA256) with a key derived from a random secret kept in the data
folder *and* the password, so logins can be ended:

* changing the password ends every login made with the old one;
* "Log out everywhere" (:meth:`Auth.end_all`) replaces the secret, which ends every login.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

COOKIE = "vt_session"
MAX_AGE = 30 * 24 * 3600
PUBLIC = ("/login", "/static/", "/healthz", "/manifest.webmanifest", "/sw.js", "/favicon")
MAX_FAILURES = 10
LOCKOUT_SECONDS = 300


def _write_secret(path: Path, key: bytes) -> None:
    """Replace the key file atomically; readable by this user only (where supported)."""
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def new_secret(data_dir: Path) -> bytes:
    """A new random secret in place of the old one: every login signed with it ends."""
    key = secrets.token_bytes(32)
    _write_secret(data_dir / "secret.key", key)
    return key


def load_secret(data_dir: Path) -> bytes:
    path = data_dir / "secret.key"
    try:
        key = path.read_bytes()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    with contextlib.suppress(OSError):  # not supported everywhere (e.g. some Windows shares)
        path.chmod(0o600)
    return key


class Auth:
    def __init__(self, password: str, secret: bytes) -> None:
        self.enabled = bool(password)
        self._digest = hashlib.sha256(password.encode()).digest()
        self._key = self._signing_key(secret)
        self._failures: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def _signing_key(self, secret: bytes) -> bytes:
        """Bound to the secret and the password: a change of either ends every login."""
        return hmac.new(secret, b"vt-session\x00" + self._digest, hashlib.sha256).digest()

    def end_all(self, data_dir: Path) -> None:
        """Log out everywhere: a new secret, so no cookie signed before is valid any more."""
        self._key = self._signing_key(new_secret(data_dir))

    def token(self, now: float | None = None) -> str:
        expires = int((now or time.time()) + MAX_AGE)
        sig = hmac.new(self._key, str(expires).encode(), hashlib.sha256).hexdigest()
        return f"{expires}.{sig}"

    def valid(self, token: str | None, now: float | None = None) -> bool:
        if not token or "." not in token:
            return False
        expires, sig = token.split(".", 1)
        expected = hmac.new(self._key, expires.encode(), hashlib.sha256).hexdigest()
        return (
            hmac.compare_digest(sig, expected)
            and expires.isdigit()
            and int(expires) > (now or time.time())
        )

    def locked(self, client: str) -> bool:
        with self._lock:
            count, since = self._failures.get(client, (0, 0.0))
            if count >= MAX_FAILURES and time.time() - since < LOCKOUT_SECONDS:
                return True
            if count >= MAX_FAILURES:
                self._failures.pop(client, None)
            return False

    def check(self, password: str, client: str) -> bool:
        ok = hmac.compare_digest(hashlib.sha256(password.encode()).digest(), self._digest)
        with self._lock:
            if ok:
                self._failures.pop(client, None)
            else:
                count, _since = self._failures.get(client, (0, 0.0))
                self._failures[client] = (count + 1, time.time())
        return ok

    def middleware(self) -> Callable[[Request, Callable], Awaitable[Response]]:
        async def mw(request: Request, call_next: Callable) -> Response:
            if not self.enabled or request.url.path.startswith(PUBLIC):
                return await call_next(request)
            if self.valid(request.cookies.get(COOKIE)):
                return await call_next(request)
            wants_page = request.method == "GET" and "text/html" in request.headers.get(
                "accept", ""
            )
            if wants_page and not request.headers.get("hx-request"):
                target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
                from urllib.parse import quote

                return RedirectResponse(f"/login?next={quote(target)}", status_code=303)
            return JSONResponse({"detail": "login required"}, status_code=401)

        return mw
