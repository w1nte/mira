from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image

from vision_tagger.web.app import create_app
from vision_tagger.web.common import STATIC_DIR


def test_manifest_service_worker_and_icons(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        r = c.get("/manifest.webmanifest")
        assert r.headers["content-type"].startswith("application/manifest+json")
        m = r.json()
        assert m["display"] == "standalone" and m["start_url"] == "/"
        sizes = {i["sizes"] for i in m["icons"]}
        assert {"192x192", "512x512"} <= sizes
        assert any(i.get("purpose") == "maskable" for i in m["icons"])
        for icon in m["icons"]:
            assert c.get(icon["src"]).status_code == 200
            w, h = icon["sizes"].split("x")
            with Image.open(STATIC_DIR / icon["src"].removeprefix("/static/")) as im:
                assert im.size == (int(w), int(h))

        sw = c.get("/sw.js")
        assert sw.headers["service-worker-allowed"] == "/"
        assert "__VERSION__" not in sw.text and "vt-shell-" in sw.text
        assert c.get("/static/offline.html").status_code == 200

        page = c.get("/").text
        assert '<link rel="manifest" href="/manifest.webmanifest">' in page
        assert "data-install" in page and 'rel="apple-touch-icon"' in page


def test_pwa_files_are_public_with_a_password(settings):
    settings.web.password = "pw"
    app = create_app(settings)
    with TestClient(app) as c:
        for path in ("/manifest.webmanifest", "/sw.js", "/static/offline.html", "/healthz"):
            assert c.get(path).status_code == 200, path
        assert c.get("/api/status").status_code == 401
