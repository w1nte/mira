from __future__ import annotations

import io
import json
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
    monkeypatch.setattr("vision_tagger.web.routes.auth.time.sleep", lambda s: None)
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


def test_traefik_compose_file_is_complete_and_documented():
    """docker-compose.traefik.yml: HTTPS in front of Mira, configured only through .env."""
    import re
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent
    text = (root / "docker-compose.traefik.yml").read_text(encoding="utf-8")

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!reset", lambda loader, node: [])  # Compose's "drop the base value"
    services = yaml.load(text, Loader=Loader)["services"]  # a SafeLoader subclass

    # every variable it reads is explained in .env.example, which .env is copied from
    example = (root / ".env.example").read_text(encoding="utf-8")
    used = set(re.findall(r"\$\{(\w+)", text))
    assert used >= {"MIRA_DOMAIN", "ACME_EMAIL"}
    for name in used | {"COMPOSE_FILE"}:
        assert re.search(rf"^(# )?{name}=", example, re.M), name
    assert "COMPOSE_FILE=docker-compose.yml:docker-compose.traefik.yml" in example

    mira = services["mira"]
    assert mira["ports"] == []  # only reachable through Traefik
    # the visitor's address, not Traefik's, counts for the login lock
    assert mira["environment"]["FORWARDED_ALLOW_IPS"] == "*"
    labels = dict(label.split("=", 1) for label in mira["labels"])
    assert labels["traefik.enable"] == "true"
    assert "${MIRA_DOMAIN" in labels["traefik.http.routers.mira.rule"]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    port = labels["traefik.http.services.mira.loadbalancer.server.port"]
    assert f"EXPOSE {port}" in dockerfile and f"VT_WEB__PORT={port}" in dockerfile

    command = " ".join(services["traefik"]["command"])
    assert "--providers.docker.exposedbydefault=false" in command  # not Ollama
    assert "readTimeout=10m" in command  # 8 MB upload pieces on slow connections
    resolver = labels["traefik.http.routers.mira.tls.certresolver"]
    assert f"--certificatesresolvers.{resolver}.acme.storage=" in command

    for doc in ("README.md", "docs/remote.md", "docker-compose.yml"):
        assert "docker-compose.traefik.yml" in (root / doc).read_text(encoding="utf-8"), doc
