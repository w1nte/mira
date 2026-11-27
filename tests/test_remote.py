from __future__ import annotations

import io
import json
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from vision_tagger.config import Settings
from vision_tagger.indexer import Indexer
from vision_tagger.web import auth as A
from vision_tagger.web.app import create_app

from .conftest import make_image

HTML = {"Accept": "text/html"}


@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setattr("vision_tagger.web.routes.auth.FAIL_DELAY", 0)
    s = Settings(
        mode="remote",
        remote={"upload_dir": tmp_path / "library", "auto_process": False},
        paths={"data_dir": tmp_path / "data"},
        vlm={"backend": "fake"},
        web={"password": "s3cret"},
    )
    s.ensure_dirs()
    for i in range(2):
        make_image(s.upload_dir / "trip" / f"p{i}.jpg", seed=i)
    return s


def test_login_protects_everything_but_public_paths(remote):
    app = create_app(remote)
    with TestClient(app) as c:
        r = c.get("/", headers=HTML, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login?next=/"
        assert c.get("/api/status").status_code == 401
        assert c.get("/healthz").json() == {"ok": True}
        assert c.get("/static/app.css").status_code == 200
        assert "Password" in c.get("/login").text

        r = c.post("/login", data={"password": "nope", "next": "/images"})
        assert "Wrong password" in r.text and A.COOKIE not in r.cookies
        r = c.post(
            "/login",
            data={"password": "s3cret", "next": "//evil.example/x"},
            follow_redirects=False,
        )
        assert r.status_code == 303 and r.headers["location"] == "/"  # no open redirect
        assert "httponly" in r.headers["set-cookie"].lower()
        assert c.get("/", headers=HTML).status_code == 200
        assert c.get("/api/status").status_code == 200
        assert "Log out" in c.get("/", headers=HTML).text
        c.post("/logout")
        assert c.get("/api/status").status_code == 401


def test_login_answers_cannot_be_framed(remote):
    with TestClient(create_app(remote)) as c:
        for r in (
            c.get("/", headers=HTML, follow_redirects=False),  # redirect to the login
            c.get("/api/status"),  # 401
            c.get("/login"),
        ):
            assert r.headers["x-frame-options"] == "DENY", r.status_code
            assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_exposure_needs_a_password():
    from vision_tagger.web.security import exposure_problem, is_loopback_host

    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2", "LOCALHOST"):
        assert is_loopback_host(host), host
        assert exposure_problem(False, host, "") is None, host
    for host in ("0.0.0.0", "::", "", "192.168.1.20", "my-pc"):
        assert not is_loopback_host(host), host
        assert "other computers" in exposure_problem(False, host, ""), host
        assert exposure_problem(False, host, "pw") is None
    assert "remote mode needs a password" in exposure_problem(True, "127.0.0.1", "")
    assert exposure_problem(True, "0.0.0.0", "pw") is None


def test_tokens_and_lockout(tmp_path):
    auth = A.Auth("pw", A.load_secret(tmp_path))
    assert A.load_secret(tmp_path) == A.load_secret(tmp_path)  # stable key
    token = auth.token(now=1000.0)
    assert auth.valid(token, now=2000.0)
    assert not auth.valid(token, now=1000.0 + A.MAX_AGE + 1)  # expired
    expires, sig = token.split(".")
    assert not auth.valid(f"{int(expires) + 99}.{sig}", now=2000.0)  # tampered
    assert not auth.valid("garbage") and not auth.valid(None)
    for _ in range(A.MAX_FAILURES):
        assert not auth.check("wrong", "1.2.3.4")
    assert auth.locked("1.2.3.4") and not auth.locked("5.6.7.8")


def test_logins_end_with_a_new_password_or_everywhere(tmp_path):
    secret = A.load_secret(tmp_path)
    auth = A.Auth("old password", secret)
    token = auth.token()
    assert auth.valid(token)
    assert not A.Auth("new password", secret).valid(token)  # the password changed
    assert A.Auth("old password", A.load_secret(tmp_path)).valid(token)  # a restart keeps it
    auth.end_all(tmp_path)
    assert not auth.valid(token)
    assert auth.valid(auth.token())
    assert A.load_secret(tmp_path) != secret  # also after a restart
    assert not A.Auth("old password", A.load_secret(tmp_path)).valid(token)
    if os.name != "nt":
        assert (tmp_path / "secret.key").stat().st_mode & 0o777 == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["secret.key"]  # no temporary file left


def test_secret_key_is_private_binary_and_made_once(tmp_path, monkeypatch):
    import threading

    keys: list[bytes] = []
    start = threading.Barrier(8)

    def load() -> None:
        start.wait()
        keys.append(A.load_secret(tmp_path))

    threads = [threading.Thread(target=load) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stored = (tmp_path / "secret.key").read_bytes()
    assert len(set(keys)) == 1 and keys[0] == stored  # one key, also for parallel starts
    if os.name != "nt":
        assert (tmp_path / "secret.key").stat().st_mode & 0o777 == 0o600
    # newlines stay as they are (Windows text mode would write \r\n)
    monkeypatch.setattr(A.secrets, "token_bytes", lambda n: b"\n\r" * (n // 2))
    other = tmp_path / "other"
    other.mkdir()
    assert A.load_secret(other) == (other / "secret.key").read_bytes() == b"\n\r" * 16
    assert A.new_secret(other) == (other / "secret.key").read_bytes()


def test_damaged_or_open_secret_keys_are_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "KEY_WAIT", 0.05)
    path = tmp_path / "secret.key"
    path.write_bytes(b"short")  # cut off
    key = A.load_secret(tmp_path)
    assert len(key) == A.KEY_BYTES and path.read_bytes() == key
    if os.name != "nt":
        path.chmod(0o644)  # as older versions left it
        assert A.load_secret(tmp_path) == key
        assert path.stat().st_mode & 0o777 == 0o600


def test_log_out_everywhere(remote):
    with TestClient(create_app(remote)) as c:
        c.post("/login", data={"password": "s3cret"})
        old = c.cookies.get(A.COOKIE)
        page = c.get("/database", headers=HTML).text
        assert 'action="/logout/everywhere"' in page and "Log out everywhere" in page
        r = c.post("/logout/everywhere", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login"
        assert c.get("/api/status").status_code == 401
        c.cookies.set(A.COOKIE, old)  # a copy of the cookie (another device) is no use either
        assert c.get("/api/status").status_code == 401
        c.post("/login", data={"password": "s3cret"})
        assert c.get("/api/status").status_code == 200
    with TestClient(create_app(remote)) as c:  # after a restart, too
        c.cookies.set(A.COOKIE, old)
        assert c.get("/api/status").status_code == 401


def test_lockout_counts_ipv6_networks_not_single_addresses(tmp_path):
    auth = A.Auth("pw", A.load_secret(tmp_path))
    for i in range(A.MAX_FAILURES):  # a new address of the same /64 for every guess
        assert not auth.locked(f"2001:db8:1:2::{i + 1:x}")
        auth.check("wrong", f"2001:db8:1:2::{i + 1:x}")
    assert auth.locked("2001:db8:1:2:ffff::1")
    assert not auth.locked("2001:db8:1:3::1") and not auth.locked("1.2.3.4")
    assert A.client_key("::ffff:10.0.0.7") == "10.0.0.7" == A.client_key("10.0.0.7")
    assert A.client_key("testclient") == "testclient"


def test_many_addresses_guessing_lock_every_new_login(tmp_path, monkeypatch):
    auth = A.Auth("pw", A.load_secret(tmp_path))
    now = [1000.0]
    monkeypatch.setattr(A.time, "time", lambda: now[0])
    for i in range(A.GLOBAL_MAX_FAILURES):
        assert not auth.locked(f"10.0.{i // 250}.{i % 250}")
        auth.check("wrong", f"10.0.{i // 250}.{i % 250}")  # one guess per address
    assert auth.locked("192.0.2.1")  # nobody may try for a while
    assert not auth.check("wrong", "192.0.2.1") and auth.locked("192.0.2.1")
    now[0] += A.LOCKOUT_SECONDS + 1
    assert not auth.locked("192.0.2.1")
    assert auth.check("pw", "192.0.2.1")


def test_failed_logins_use_bounded_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "MAX_TRACKED", 50)
    auth = A.Auth("pw", A.load_secret(tmp_path))
    for i in range(500):
        auth.check("wrong", f"10.1.{i // 250}.{i % 250}")
    assert len(auth._failures) <= 50


def test_parallel_guesses_are_counted_before_the_delay(remote, monkeypatch):
    """The login is async: a burst of parallel guesses cannot all pass the lockout first."""
    import asyncio

    import httpx

    monkeypatch.setattr("vision_tagger.web.routes.auth.FAIL_DELAY", 0.2)
    app = create_app(remote)

    async def burst() -> list[str]:
        transport = httpx.ASGITransport(app=app, client=("198.51.100.7", 1234))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            replies = await asyncio.gather(
                *(c.post("/login", data={"password": f"guess{i}"}) for i in range(25))
            )
        return [r.text for r in replies]

    with TestClient(app):  # runs the lifespan (app state)
        texts = asyncio.run(burst())
    assert sum("Wrong password" in t for t in texts) == A.MAX_FAILURES
    assert sum("Too many attempts" in t for t in texts) == 25 - A.MAX_FAILURES


def test_wrong_passwords_do_not_hold_worker_threads(remote, monkeypatch):
    """The delay after a wrong password is awaited: guessing cannot use up the thread pool that
    serves every other page."""
    import asyncio
    import time

    import anyio
    import httpx

    monkeypatch.setattr("vision_tagger.web.routes.auth.FAIL_DELAY", 0.4)
    app = create_app(remote)

    async def burst() -> float:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2  # this loop only
        started = time.monotonic()
        replies = []
        for i in range(10):  # each from its own address: no lockout
            transport = httpx.ASGITransport(app=app, client=(f"198.51.100.{i + 1}", 1234))
            client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
            replies.append(client.post("/login", data={"password": "wrong"}))
        texts = [r.text for r in await asyncio.gather(*replies)]
        assert all("Wrong password" in t for t in texts)
        return time.monotonic() - started

    with TestClient(app):
        elapsed = asyncio.run(burst())
    assert elapsed < 1.6, elapsed  # blocking: 10 delays on 2 threads would take 2 s or more


def test_password_rules():
    from vision_tagger.web.security import password_problem, weak_password

    assert "example" in password_problem("choose-a-good-one")
    assert password_problem(" Choose-A-Good-One ") is not None
    assert password_problem("") is None and password_problem("something else") is None
    assert weak_password("short") and not weak_password("") and not weak_password("x" * 12)


def test_remote_hosts_and_zip_export(remote):
    remote.web.password = ""
    app = create_app(remote)
    with TestClient(app, base_url="http://nas.local:8765") as c:
        Indexer(app.state.ctx.db, remote).run()
        assert c.get("/images").status_code == 200  # any host name in remote mode
        r = c.post(
            "/images/export", data={"target": "/tmp/x"}, headers={"Origin": "http://evil.example"}
        )
        assert r.status_code == 403  # still no cross-site writes
        r = c.post("/images/export", data={"target": "/etc/x"})
        assert r.status_code == 403  # remote users may not write to server folders
        page = c.get("/images").text
        assert "/images/download.zip" in page and 'action="/images/export"' not in page
        r = c.get("/images/download.zip")
        assert r.headers["content-type"] == "application/zip"
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = sorted(zf.namelist())
        assert names == ["p0.jpg", "p0.jpg.xmp", "p1.jpg", "p1.jpg.xmp", "vision-tagger.json"]
        assert zf.read("p0.jpg") == (remote.upload_dir / "trip" / "p0.jpg").read_bytes()
        assert len(json.loads(zf.read("vision-tagger.json"))["photos"]) == 2


def test_local_mode_still_checks_hosts(settings):
    app = create_app(settings)
    with TestClient(app, base_url="http://evil.example") as c:
        assert c.get("/").status_code == 400


def test_docker_env_file_instructions_work_on_windows():
    """Windows PowerShell 5.1 writes UTF-16 with `echo … > .env`, which docker compose rejects."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    example = (root / ".env.example").read_bytes()
    assert not example.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"))  # no BOM
    assert b"\nVT_PASSWORD=" in example and example.decode("ascii")  # plain text
    for doc in ("README.md", "docs/remote.md", "docker-compose.yml"):
        assert 'echo "VT_PASSWORD' not in (root / doc).read_text(encoding="utf-8"), doc
    for ignore in (".gitignore", ".dockerignore"):  # the password never lands in git or images
        assert ".env" in (root / ignore).read_text(encoding="utf-8").splitlines(), ignore


def test_no_logins_card_without_a_password(settings):
    with TestClient(create_app(settings)) as c:
        assert "logout/everywhere" not in c.get("/database").text
        before = (settings.data_dir / "secret.key").read_bytes()
        r = c.post("/logout/everywhere", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
        assert (settings.data_dir / "secret.key").read_bytes() == before
